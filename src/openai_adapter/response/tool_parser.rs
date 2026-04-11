//! 工具调用解析 —— 把 prompt 中约束的 XML <tool_calls> 转换为结构化 tool_calls

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use log::debug;

use crate::openai_adapter::OpenAIAdapterError;
use crate::openai_adapter::types::{
    ChatCompletionChunk, ChunkChoice, Delta, FunctionCall, ToolCall,
};

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
const MAX_XML_BUF_LEN: usize = 64 * 1024;

fn next_call_id() -> String {
    let n = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{:016x}", n)
}

/// 解析 XML 工具调用文本，返回结构化 ToolCall 列表
///
/// 输入示例:
/// `<tool_calls><tool_call name="get_weather" arguments="{&quot;city&quot;:&quot;北京&quot;}" /></tool_calls>`
pub fn parse_tool_calls(xml: &str) -> Option<Vec<ToolCall>> {
    let start = xml.find("<tool_calls>")?;
    let end = xml.find("</tool_calls>")? + "</tool_calls>".len();
    let inner = &xml[start + "<tool_calls>".len()..end - "</tool_calls>".len()];

    let mut calls = Vec::new();
    let mut search_start = 0;
    while let Some(pos) = inner[search_start..].find("<tool_call ") {
        let tag_start = search_start + pos;
        let tag_end = inner[tag_start..].find("/>")? + tag_start + 2;
        let tag = &inner[tag_start..tag_end];
        search_start = tag_end;

        let name = extract_attr(tag, "name")?;
        let args_raw = extract_attr(tag, "arguments")?;
        let args_json = args_raw.replace("&quot;", "\"");
        let arguments = serde_json::from_str::<serde_json::Value>(&args_json)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| args_json);

        calls.push(ToolCall {
            id: next_call_id(),
            ty: "function".to_string(),
            function: Some(FunctionCall { name, arguments }),
            custom: None,
            index: calls.len() as u32,
        });
    }

    if calls.is_empty() { None } else { Some(calls) }
}

fn extract_attr(tag: &str, key: &str) -> Option<String> {
    let dq = format!("{}=\"", key);
    let sq = format!("{}='", key);
    let (start, quote) = if let Some(pos) = tag.find(&dq) {
        (pos + dq.len(), '"')
    } else if let Some(pos) = tag.find(&sq) {
        (pos + sq.len(), '\'')
    } else {
        return None;
    };
    let end = tag[start..].find(quote)? + start;
    Some(tag[start..end].to_string())
}

fn make_end_chunk(model: &str, delta: Delta, finish_reason: &'static str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: "chatcmpl-end".to_string(),
        object: "chat.completion.chunk",
        created: 0,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage: None,
        service_tier: None,
        system_fingerprint: None,
    }
}

#[derive(Debug)]
enum ToolParseState {
    Detecting,
    PlainText,
    CollectingXml(String),
    ToolCallsEmitted,
}

pin_project! {
    #[allow(unused_doc_comments)]
    /// 在 content delta 中检测并解析 XML <tool_calls> 的流转换器
    pub struct ToolCallStream<S> {
        #[pin]
        inner: S,
        state: ToolParseState,
        model: String,
    }
}

impl<S> ToolCallStream<S> {
    /// 创建工具调用解析流
    pub fn new(inner: S, model: String) -> Self {
        Self {
            inner,
            state: ToolParseState::Detecting,
            model,
        }
    }
}

impl<S> Stream for ToolCallStream<S>
where
    S: Stream<Item = Result<ChatCompletionChunk, OpenAIAdapterError>>,
{
    type Item = Result<ChatCompletionChunk, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(mut chunk))) => {
                    let choice = match chunk.choices.first_mut() {
                        Some(c) => c,
                        None => return Poll::Ready(Some(Ok(chunk))),
                    };

                    if let Some(content) = choice.delta.content.take() {
                        if content.is_empty() {
                            choice.delta.content = Some(content);
                            return Poll::Ready(Some(Ok(chunk)));
                        }

                        match &mut this.state {
                            ToolParseState::Detecting => {
                                if content.trim_start().starts_with('<') {
                                    debug!(target: "adapter", "tool_parser 开始收集 XML 工具调用");
                                    *this.state = ToolParseState::CollectingXml(content);
                                    continue;
                                } else {
                                    *this.state = ToolParseState::PlainText;
                                    choice.delta.content = Some(content);
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                            }
                            ToolParseState::PlainText => {
                                choice.delta.content = Some(content);
                                return Poll::Ready(Some(Ok(chunk)));
                            }
                            ToolParseState::CollectingXml(buf) => {
                                buf.push_str(&content);
                                if buf.len() > MAX_XML_BUF_LEN {
                                    debug!(target: "adapter", "tool_parser XML 缓冲超过上限，回退为纯文本");
                                    let flushed = std::mem::take(buf);
                                    *this.state = ToolParseState::PlainText;
                                    choice.delta.content = Some(flushed);
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                if let Some(end_pos) = buf.find("</tool_calls>") {
                                    let end_pos = end_pos + "</tool_calls>".len();
                                    let collected = buf[..end_pos].to_string();
                                    let tail = buf.split_off(end_pos);
                                    *this.state = ToolParseState::PlainText;

                                    if let Some(calls) = parse_tool_calls(&collected) {
                                        debug!(
                                            target: "adapter",
                                            "tool_parser 解析出 {} 个工具调用",
                                            calls.len()
                                        );
                                        let has_tail = !tail.trim().is_empty();
                                        choice.delta.content =
                                            if has_tail { Some(tail) } else { None };
                                        choice.delta.tool_calls = Some(calls);
                                        if choice.finish_reason == Some("stop") {
                                            choice.finish_reason = Some("tool_calls");
                                        }
                                        *this.state = if has_tail {
                                            ToolParseState::PlainText
                                        } else {
                                            ToolParseState::ToolCallsEmitted
                                        };
                                    } else {
                                        debug!(target: "adapter", "tool_parser XML 解析失败，回退为纯文本");
                                        let mut flushed = collected;
                                        flushed.push_str(&tail);
                                        choice.delta.content = Some(flushed);
                                        *this.state = ToolParseState::PlainText;
                                    }
                                    return Poll::Ready(Some(Ok(chunk)));
                                } else {
                                    continue;
                                }
                            }
                            ToolParseState::ToolCallsEmitted => {
                                choice.delta.content = Some(content);
                                return Poll::Ready(Some(Ok(chunk)));
                            }
                        }
                    } else if matches!(this.state, ToolParseState::CollectingXml(_))
                        || matches!(this.state, ToolParseState::ToolCallsEmitted)
                    {
                        if let ToolParseState::CollectingXml(buf) =
                            std::mem::replace(this.state, ToolParseState::PlainText)
                        {
                            if let Some(calls) = parse_tool_calls(&buf) {
                                choice.delta.tool_calls = Some(calls);
                                if choice.finish_reason == Some("stop") {
                                    choice.finish_reason = Some("tool_calls");
                                }
                            } else {
                                choice.delta.content = Some(buf);
                            }
                        } else if choice.finish_reason == Some("stop") {
                            choice.finish_reason = Some("tool_calls");
                        }
                        return Poll::Ready(Some(Ok(chunk)));
                    } else {
                        return Poll::Ready(Some(Ok(chunk)));
                    }
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    if let ToolParseState::CollectingXml(buf) =
                        std::mem::replace(this.state, ToolParseState::PlainText)
                    {
                        // 流结束时必须清空缓冲，避免未发射的 XML 内容丢失
                        if let Some(calls) = parse_tool_calls(&buf) {
                            let chunk = make_end_chunk(
                                this.model,
                                Delta {
                                    tool_calls: Some(calls),
                                    ..Default::default()
                                },
                                "tool_calls",
                            );
                            return Poll::Ready(Some(Ok(chunk)));
                        } else {
                            let chunk = make_end_chunk(
                                this.model,
                                Delta {
                                    content: Some(buf),
                                    ..Default::default()
                                },
                                "stop",
                            );
                            return Poll::Ready(Some(Ok(chunk)));
                        }
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

    use crate::openai_adapter::types::{ChunkChoice, Delta};

    use super::*;

    #[test]
    fn parse_simple_xml_tool() {
        let xml = r#"<tool_calls><tool_call name="get_weather" arguments="{&quot;city&quot;:&quot;北京&quot;}" /></tool_calls>"#;
        let calls = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(
            calls[0].function.as_ref().unwrap().arguments,
            r#"{"city":"北京"}"#
        );
    }

    #[test]
    fn parse_single_quote_args() {
        let xml = r#"<tool_calls><tool_call name="get_weather" arguments='{&quot;city&quot;:&quot;北京&quot;}' /></tool_calls>"#;
        let calls = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(
            calls[0].function.as_ref().unwrap().arguments,
            r#"{"city":"北京"}"#
        );
    }

    #[test]
    fn parse_multiple_tools_with_index() {
        let xml = r#"<tool_calls><tool_call name="get_weather" arguments="{}" /><tool_call name="get_time" arguments="{&quot;tz&quot;:&quot;bj&quot;}" /></tool_calls>"#;
        let calls = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(calls[1].index, 1);
        assert_eq!(calls[1].function.as_ref().unwrap().name, "get_time");
    }

    #[tokio::test]
    async fn tool_stream_detect_and_burst() {
        let chunks = vec![Ok(ChatCompletionChunk {
            id: "1".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some(
                        r#"<tool_calls><tool_call name="f" arguments="{}" /></tool_calls>"#.into(),
                    ),
                    ..Default::default()
                },
                finish_reason: Some("stop"),
                logprobs: None,
            }],
            usage: None,
            service_tier: None,
            system_fingerprint: None,
        })];
        let mut ts = ToolCallStream::new(futures::stream::iter(chunks), "m".into());
        let out = ts.next().await.unwrap().unwrap();
        assert!(out.choices[0].delta.content.is_none());
        assert_eq!(out.choices[0].finish_reason, Some("tool_calls"));
        assert_eq!(out.choices[0].delta.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn tool_stream_plain_text_fallback() {
        let chunks = vec![Ok(ChatCompletionChunk {
            id: "1".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            service_tier: None,
            system_fingerprint: None,
        })];
        let mut ts = ToolCallStream::new(futures::stream::iter(chunks), "m".into());
        let out = ts.next().await.unwrap().unwrap();
        assert_eq!(out.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(out.choices[0].delta.tool_calls.is_none());
    }

    #[tokio::test]
    async fn tool_stream_xml_collected_then_finish() {
        // 模拟真实流：content 和 finish_reason 分在两个 chunk
        let chunks = vec![
            Ok(ChatCompletionChunk {
                id: "1".into(),
                object: "chat.completion.chunk",
                created: 0,
                model: "m".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        content: Some(
                            r#"<tool_calls><tool_call name="f" arguments="{}" /></tool_calls>"#
                                .into(),
                        ),
                        ..Default::default()
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            }),
            Ok(ChatCompletionChunk {
                id: "2".into(),
                object: "chat.completion.chunk",
                created: 0,
                model: "m".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: Some("stop"),
                    logprobs: None,
                }],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            }),
        ];
        let mut ts = ToolCallStream::new(futures::stream::iter(chunks), "m".into());
        // 第一个 content chunk 被 swallow，直接拿到 finish chunk 转成的 tool_calls
        let out = ts.next().await.unwrap().unwrap();
        assert!(out.choices[0].delta.content.is_none());
        assert_eq!(out.choices[0].finish_reason, Some("tool_calls"));
        assert_eq!(out.choices[0].delta.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn tool_stream_fragmented_xml() {
        // 模拟碎片化 XML：第一个 chunk 只有 "<" 也能被捕获
        let chunks = vec![
            Ok(ChatCompletionChunk {
                id: "1".into(),
                object: "chat.completion.chunk",
                created: 0,
                model: "m".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        content: Some(r#"<tool_calls><tool_call name="get_weather" arguments='{"city":"北京"}' />"#.into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            }),
            Ok(ChatCompletionChunk {
                id: "2".into(),
                object: "chat.completion.chunk",
                created: 0,
                model: "m".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        content: Some("</tool_calls>".into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            }),
            Ok(ChatCompletionChunk {
                id: "3".into(),
                object: "chat.completion.chunk",
                created: 0,
                model: "m".into(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: Some("stop"),
                    logprobs: None,
                }],
                usage: None,
                service_tier: None,
                system_fingerprint: None,
            }),
        ];
        let mut ts = ToolCallStream::new(futures::stream::iter(chunks), "m".into());
        let out = ts.next().await.unwrap().unwrap();
        assert!(out.choices[0].delta.content.is_none());
        assert_eq!(out.choices[0].finish_reason, None);
        let calls = out.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(
            calls[0].function.as_ref().unwrap().arguments,
            r#"{"city":"北京"}"#
        );

        let out2 = ts.next().await.unwrap().unwrap();
        assert_eq!(out2.choices[0].finish_reason, Some("tool_calls"));
    }
}
