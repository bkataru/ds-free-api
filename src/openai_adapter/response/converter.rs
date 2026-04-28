//! Streaming chunk projector — converts `DsFrame`s into OpenAI-compatible `chat.completion.chunk` payloads.
//!
//! Ordering rules (tools_present parity):
//!
//! - `tools_present=false`: stream `reasoning_content` deltas verbatim (Anthropic-compatible thinking channel).
//!
//! - `tools_present=true`: buffer THINK deltas into `buffered_reasoning`. When RESPONSE text arrives or the stream finishes,
//!   merge buffered reasoning ahead of ordinary assistant content so tooling requests never emit an empty `choices` shard
//!   ahead of streamed JSON tool calls downstream.
//!
//! - `pending_reasoning_flush` coordinates the rare case where `include_usage=true` forces a usage-only chunk (`choices=[]`)
//!   before the merged reasoning/content pair is safe to flush. Finish handling sets this latch when usage preempts deltas.
//!
//! - Finalization always prefers draining buffered reasoning (if any), then emits `finish_reason`, then optional trailing usage.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use log::trace;

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

fn make_usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    }
}

pub(crate) fn make_chunk(model: &str, delta: Delta, finish: Option<&'static str>) -> ChatCompletionChunk {
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
    /// Adapter stream that folds `DsFrame` emissions into chunked OpenAI deltas
    pub struct ConverterStream<S> {
        #[pin]
        inner: S,
        model: String,
        include_usage: bool,
        include_obfuscation: bool,
        prompt_tokens: u32,
        finished: bool,
        usage_value: Option<u32>,
        tools_present: bool,
        buffered_reasoning: String,
        pending_finish: bool,
        pending_reasoning_flush: bool,
    }
}

impl<S> ConverterStream<S> {
    /// Build a chunked encoder around an upstream DeepSeek patch stream.
    pub fn new(
        inner: S,
        model: String,
        include_usage: bool,
        include_obfuscation: bool,
        prompt_tokens: u32,
        tools_present: bool,
    ) -> Self {
        Self {
            inner,
            model,
            include_usage,
            include_obfuscation,
            prompt_tokens,
            finished: false,
            usage_value: None,
            tools_present,
            buffered_reasoning: String::new(),
            pending_reasoning_flush: false,
            pending_finish: false,
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

        // When upstream already signaled completion but usage waited, flush usage chunks first.
        if *this.finished
            && *this.include_usage
            && let Some(u) = this.usage_value.take()
        {
            return Poll::Ready(Some(Ok(make_usage_chunk(
                make_usage(*this.prompt_tokens, u),
                this.model,
            ))));
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(frame))) => match frame {
                    DsFrame::Role => {
                        trace!(target: "adapter", ">>> conv: role=assistant");
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
                        trace!(target: "adapter", ">>> conv: think delta len={} tools_present={}", text.len(), *this.tools_present);
                        if *this.tools_present {
                            // Buffer reasoning for later prepending to content
                            this.buffered_reasoning.push_str(&text);
                            continue;
                        }
                        // No tools: emit as reasoning_content
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
                        trace!(target: "adapter", ">>> conv: content delta len={} buf_len={}", text.len(), this.buffered_reasoning.len());
                        let content = if !this.buffered_reasoning.is_empty() {
                            let combined = format!("{}{}", *this.buffered_reasoning, text);
                            this.buffered_reasoning.clear();
                            combined
                        } else {
                            text
                        };
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta {
                                content: Some(content),
                                ..Default::default()
                            },
                            None,
                        ))));
                    }
                    DsFrame::Status(status) if status == "FINISHED" && !*this.finished => {
                        trace!(target: "adapter", ">>> conv: finish=stop status buf_len={}", this.buffered_reasoning.len());
                        // Flush any buffered reasoning as content before finishing
                        if !this.buffered_reasoning.is_empty() {
                            let buffered = std::mem::take(this.buffered_reasoning);
                            *this.finished = true;
                            return Poll::Ready(Some(Ok(make_chunk(
                                this.model,
                                Delta {
                                    content: Some(buffered),
                                    ..Default::default()
                                },
                                Some("stop"),
                            ))));
                        }
                        *this.finished = true;
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta::default(),
                            Some("stop"),
                        ))));
                    }
                    DsFrame::Status(_) => {}
                    DsFrame::Usage(u) => {
                        trace!(target: "adapter", ">>> conv: usage={} finished={} include_usage={}", u, *this.finished, *this.include_usage);
                        *this.usage_value = Some(u);
                        if *this.finished && *this.include_usage {
                            return Poll::Ready(Some(Ok(make_usage_chunk(
                                make_usage(*this.prompt_tokens, u),
                                this.model,
                            ))));
                        }
                    }
                    DsFrame::Finish if !*this.finished => {
                        trace!(target: "adapter", ">>> conv: finish=stop pending_flush={} tools_present={}", *this.pending_reasoning_flush, *this.tools_present);
                        *this.finished = true;
                        // Emit pending usage before finishing
                        if *this.include_usage {
                            if let Some(u) = this.usage_value.take() {
                                // Don't return yet - set flag to flush reasoning next
                                *this.pending_reasoning_flush = true;
                                return Poll::Ready(Some(Ok(make_usage_chunk(
                                    make_usage(*this.prompt_tokens, u),
                                    this.model,
                                ))));
                            }
                        }
                        // Flush buffered reasoning as content when tools_present
                        if *this.tools_present && !this.buffered_reasoning.is_empty() {
                            *this.pending_finish = true;
                            let reasoning = std::mem::replace(this.buffered_reasoning, String::new());
                            return Poll::Ready(Some(Ok(make_chunk(
                                this.model,
                                Delta {
                                    content: Some(reasoning),
                                    ..Default::default()
                                },
                                None,
                            ))));
                        }
                        // Default: emit finish chunk
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
                    trace!(target: "adapter", ">>> conv: stream_end pending_flush={} pending_finish={} buf_len={}", *this.pending_reasoning_flush, *this.pending_finish, this.buffered_reasoning.len());
                    // First, check if we need to flush reasoning (after usage was emitted)
                    if *this.pending_reasoning_flush {
                        *this.pending_reasoning_flush = false;
                        *this.pending_finish = true;
                        // Flush any remaining buffered reasoning
                        if !this.buffered_reasoning.is_empty() {
                            let buffered = std::mem::take(this.buffered_reasoning);
                            return Poll::Ready(Some(Ok(make_chunk(
                                this.model,
                                Delta {
                                    content: Some(buffered),
                                    ..Default::default()
                                },
                                None,
                            ))));
                        }
                    }
                    // Emit pending finish chunk if we were waiting to flush reasoning
                    if *this.pending_finish {
                        *this.pending_finish = false;
                        return Poll::Ready(Some(Ok(make_chunk(this.model, Delta::default(), Some("stop")))));
                    }
                    if *this.finished
                        && *this.include_usage
                        && let Some(u) = this.usage_value.take()
                    {
                        return Poll::Ready(Some(Ok(make_usage_chunk(
                            make_usage(*this.prompt_tokens, u),
                            this.model,
                        ))));
                    }
                    // Flush any remaining buffered reasoning
                    if !this.buffered_reasoning.is_empty() {
                        let buffered = std::mem::take(this.buffered_reasoning);
                        return Poll::Ready(Some(Ok(make_chunk(
                            this.model,
                            Delta {
                                content: Some(buffered),
                                ..Default::default()
                            },
                            None,
                        ))));
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

    use super::super::state::DsFrame;

    use super::*;

    #[tokio::test]
    async fn converter_emits_role_and_content() {
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::Role),
            Ok(DsFrame::ContentDelta("hello".into())),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-default".into(), false, false, 0, false);
        let chunk1 = conv.next().await.unwrap().unwrap();
        assert_eq!(chunk1.choices[0].delta.role, Some("assistant"));
        let chunk2 = conv.next().await.unwrap().unwrap();
        assert_eq!(chunk2.choices[0].delta.content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn converter_buffers_reasoning_when_tools_present() {
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::ThinkDelta("Thinking... ".into())),
            Ok(DsFrame::ContentDelta("Hello".into())),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-expert".into(), false, false, 0, true);
        let chunk = conv.next().await.unwrap().unwrap();
        // First chunk should have combined content (reasoning buffered and prepended)
        assert_eq!(chunk.choices[0].delta.reasoning_content, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Thinking... Hello"));
    }

    #[tokio::test]
    async fn converter_emits_reasoning_when_no_tools() {
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::ThinkDelta("Thinking...".into())),
            Ok(DsFrame::ContentDelta("Hello".into())),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-expert".into(), false, false, 0, false);
        let chunk = conv.next().await.unwrap().unwrap();
        // First chunk should have reasoning_content
        assert_eq!(chunk.choices[0].delta.reasoning_content, Some("Thinking...".into()));
    }
    
    #[tokio::test]
    async fn converter_flush_buffered_reasoning_with_finish() {
        // Tools present, Finish should flush buffered reasoning as content
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::ThinkDelta("Thinking...".into())),
            Ok(DsFrame::Finish),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-expert".into(), false, false, 0, true);
        let chunk1 = conv.next().await.unwrap().unwrap();
        // Should have combined content (reasoning flushed)
        assert_eq!(chunk1.choices[0].delta.content.as_deref(), Some("Thinking..."));
        let chunk2 = conv.next().await.unwrap().unwrap();
        // Should have finish
        assert_eq!(chunk2.choices[0].finish_reason, Some("stop".into()));
    }
    
    #[tokio::test]
    async fn converter_finish_with_buffered_reasoning_and_usage() {
        // Tools present, include_usage, Finish should emit usage then flush reasoning
        let frames = futures::stream::iter(vec![
            Ok(DsFrame::ThinkDelta("Thinking...".into())),
            Ok(DsFrame::Usage(42)),
            Ok(DsFrame::Finish),
        ]);
        let mut conv = ConverterStream::new(frames, "deepseek-expert".into(), true, false, 100, true);
        let chunk1 = conv.next().await.unwrap().unwrap();
        // Should emit usage first
        assert_eq!(chunk1.usage.unwrap().total_tokens, 142);
        let chunk2 = conv.next().await.unwrap().unwrap();
        // Then flush reasoning
        assert_eq!(chunk2.choices[0].delta.content.as_deref(), Some("Thinking..."));
        let chunk3 = conv.next().await.unwrap().unwrap();
        // Then finish
        assert_eq!(chunk3.choices[0].finish_reason, Some("stop".into()));
    }
    }
