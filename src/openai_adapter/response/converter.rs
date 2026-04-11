//! OpenAI Chunk 生成器 —— 将 DsFrame 映射为 ChatCompletionChunk

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use log::debug;

use crate::openai_adapter::OpenAIAdapterError;
use crate::openai_adapter::types::{ChatCompletionChunk, ChunkChoice, Delta, Usage};

use super::state::DsFrame;
use super::{next_chatcmpl_id, now_secs};

fn make_usage_chunk(usage: Usage, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: next_chatcmpl_id(),
        object: "chat.completion.chunk",
        created: now_secs(),
        model: model.to_string(),
        choices: vec![],
        usage: Some(usage),
        service_tier: None,
        system_fingerprint: None,
    }
}

fn make_chunk(model: &str, delta: Delta, finish: Option<&'static str>) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: next_chatcmpl_id(),
        object: "chat.completion.chunk",
        created: now_secs(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: finish,
            logprobs: None,
        }],
        usage: None,
        service_tier: None,
        system_fingerprint: None,
    }
}

pin_project! {
    #[allow(unused_doc_comments)]
    /// 将 DsFrame 增量帧映射为 OpenAI ChatCompletionChunk 的流转换器
    pub struct ConverterStream<S> {
        #[pin]
        inner: S,
        model: String,
        include_usage: bool,
        include_obfuscation: bool,
        finished: bool,
        usage_value: Option<u32>,
    }
}

impl<S> ConverterStream<S> {
    /// 创建 Chunk 转换流
    pub fn new(inner: S, model: String, include_usage: bool, include_obfuscation: bool) -> Self {
        Self {
            inner,
            model,
            include_usage,
            include_obfuscation,
            finished: false,
            usage_value: None,
        }
    }
}

impl<S> Stream for ConverterStream<S>
where
    S: Stream<Item = Result<DsFrame, OpenAIAdapterError>>,
{
    type Item = Result<ChatCompletionChunk, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // 如果已结束且有待发 usage，优先发送
        if *this.finished
            && *this.include_usage
            && let Some(u) = this.usage_value.take()
        {
            let usage = Usage {
                prompt_tokens: 0,
                completion_tokens: u,
                total_tokens: u,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            };
            return Poll::Ready(Some(Ok(make_usage_chunk(usage, this.model))));
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(frame))) => match frame {
                    DsFrame::Role => {
                        debug!(target: "adapter", "converter 生成 role chunk");
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta {
                                role: Some("assistant"),
                                ..Default::default()
                            },
                            None,
                        ))));
                    }
                    DsFrame::ThinkDelta(text) => {
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta {
                                reasoning_content: Some(text),
                                ..Default::default()
                            },
                            None,
                        ))));
                    }
                    DsFrame::ContentDelta(text) => {
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta {
                                content: Some(text),
                                ..Default::default()
                            },
                            None,
                        ))));
                    }
                    DsFrame::Status(status) if status == "FINISHED" && !*this.finished => {
                        debug!(target: "adapter", "converter 收到 FINISHED 状态");
                        *this.finished = true;
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta::default(),
                            Some("stop"),
                        ))));
                    }
                    DsFrame::Status(_) => {}
                    DsFrame::Usage(u) => {
                        *this.usage_value = Some(u);
                        if *this.finished && *this.include_usage {
                            return Poll::Ready(Some(Ok(make_usage_chunk(
                                Usage {
                                    prompt_tokens: 0,
                                    completion_tokens: u,
                                    total_tokens: u,
                                    prompt_tokens_details: None,
                                    completion_tokens_details: None,
                                },
                                this.model,
                            ))));
                        }
                    }
                    DsFrame::Finish if !*this.finished => {
                        debug!(target: "adapter", "converter 收到 finish 事件");
                        *this.finished = true;
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta::default(),
                            Some("stop"),
                        ))));
                    }
                    DsFrame::Finish => {}
                },
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    if *this.finished
                        && *this.include_usage
                        && let Some(u) = this.usage_value.take()
                    {
                        let usage = Usage {
                            prompt_tokens: 0,
                            completion_tokens: u,
                            total_tokens: u,
                            prompt_tokens_details: None,
                            completion_tokens_details: None,
                        };
                        return Poll::Ready(Some(Ok(make_usage_chunk(usage, this.model))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use crate::openai_adapter::response::state::DsFrame;

    use super::*;

    #[tokio::test]
    async fn converter_emits_role_and_content() {
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::Role),
            Ok(DsFrame::ContentDelta("hello".into())),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-default".into(), false, false);
        let chunk1 = conv.next().await.unwrap().unwrap();
        assert_eq!(chunk1.choices[0].delta.role, Some("assistant"));
        let chunk2 = conv.next().await.unwrap().unwrap();
        assert_eq!(chunk2.choices[0].delta.content.as_deref(), Some("hello"));
    }
}
