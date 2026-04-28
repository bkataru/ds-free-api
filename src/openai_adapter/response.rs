//! Adapter response pipeline — convert DeepSeek SSE into OpenAI-shaped chunks / completions.
//!
//! Pipeline: `sse_parser` → `state` → `converter` → `tool_parser`
//! - Only `THINK` / `RESPONSE` fragments become user-visible text.
//! - Length obfuscation inserts dynamic padding at the final SSE serialization boundary.

mod converter;
mod sse_parser;
mod state;
mod tool_parser;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use log::{debug, error};
use pin_project_lite::pin_project;
use rand::RngExt;

use crate::openai_adapter::{
    OpenAIAdapterError, StreamResponse,
    types::{ChatCompletion, ChatCompletionChunk, ChunkChoice, Choice, Delta, MessageResponse, ToolCall},
};

static CHATCMPL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_chatcmpl_id() -> String {
    let n = CHATCMPL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("chatcmpl-{:016x}", n)
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

const OBFUSCATION_TARGET_LEN: usize = 512;
const OBFUSCATION_MIN_PAD: usize = 16;
const FINISH_STOP: &str = "stop";
const FINISH_TOOL_CALLS: &str = "tool_calls";

fn random_padding(len: usize) -> String {
    if len == 0 {
        return String::new();
    }
    let byte_len = (len * 3).div_ceil(4);
    let mut bytes = vec![0u8; byte_len];
    rand::rng().fill(&mut bytes);
    let s = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
    s[..len].to_string()
}

fn chunk_to_bytes(
    mut chunk: ChatCompletionChunk,
    include_obfuscation: bool,
) -> Result<Bytes, OpenAIAdapterError> {
    if include_obfuscation && !chunk.choices.is_empty() {
        let without = serde_json::to_string(&chunk).map_err(OpenAIAdapterError::from)?;
        let overhead = r#","obfuscation":"""#.len();
        let pad_len = if without.len() + overhead < OBFUSCATION_TARGET_LEN {
            OBFUSCATION_TARGET_LEN - without.len() - overhead
        } else {
            OBFUSCATION_MIN_PAD
        };
        if let Some(choice) = chunk.choices.first_mut() {
            choice.delta.obfuscation = Some(random_padding(pad_len));
        }
    }
    let json_text = serde_json::to_string(&chunk).map_err(OpenAIAdapterError::from)?;
    Ok(Bytes::from(format!("data: {}\n\n", json_text)))
}

fn find_stop_pos(content: &str, stop: &[String]) -> Option<usize> {
    stop.iter().filter_map(|s| content.find(s)).min()
}

/// RepairStream inner stream type
#[allow(dead_code)]
type ChunkStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, OpenAIAdapterError>> + Send>>;

/// Async closure type: takes raw XML text, returns repaired tool_calls via a fresh ds_core call.
pub type RepairFn = Arc<
    dyn Fn(
            String,
        )
            -> Pin<Box<dyn Future<Output = Result<Vec<ToolCall>, OpenAIAdapterError>> + Send>>
        + Send
        + Sync,
>;

/// Execute a repair: consume ds_core byte stream, extract text from ContentDelta frames,
/// wrap in <tool_calls> if missing, parse to Vec<ToolCall>.
pub(crate) async fn execute_tool_repair(
    ds_stream: Pin<Box<dyn Stream<Item = Result<Bytes, crate::ds_core::CoreError>> + Send>>,
) -> Result<Vec<ToolCall>, OpenAIAdapterError> {
    let sse = sse_parser::SseStream::new(ds_stream);
    let state_stream = state::StateStream::new(sse);
    futures::pin_mut!(state_stream);

    let mut text = String::new();
    while let Some(frame) = state_stream.next().await {
        if let state::DsFrame::ContentDelta(t) = frame? {
            text.push_str(&t);
            if text.len() > tool_parser::MAX_XML_BUF_LEN {
                return Err(OpenAIAdapterError::Internal(
                    "repair model output exceeds buffer limit, abandoning repair".into(),
                ));
            }
        }
    }

    let wrapped = if text.contains("<tool_calls>") {
        text.trim().to_string()
    } else {
        format!("<tool_calls>{}</tool_calls>", text.trim())
    };

    let (calls, _) = tool_parser::parse_tool_calls(&wrapped).ok_or_else(|| {
        OpenAIAdapterError::Internal(format!(
            "repair model returned unparseable content: {}",
            &text[..text.len().min(200)]
        ))
    })?;

    Ok(calls)
}

/// RepairStream: catches ToolCallRepairNeeded errors from the upstream stream,
/// calls the repair closure to get corrected tool_calls, then resumes forwarding.
enum RepairState {
    Forwarding,
    Repairing {
        future: Pin<Box<dyn Future<Output = Result<Vec<ToolCall>, OpenAIAdapterError>> + Send>>,
    },
    RepairFailed(String),
    Done,
}

pin_project! {
    struct RepairStream<S> {
        #[pin]
        inner: S,
        repair_fn: RepairFn,
        state: RepairState,
    }
}

impl<S> RepairStream<S> {
    fn new(inner: S, repair_fn: RepairFn) -> Self {
        Self {
            inner,
            repair_fn,
            state: RepairState::Forwarding,
        }
    }
}

impl<S, E> Stream for RepairStream<S>
where
    S: Stream<Item = Result<ChatCompletionChunk, E>>,
    E: Into<OpenAIAdapterError>,
{
    type Item = Result<ChatCompletionChunk, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.state {
                RepairState::Forwarding => {
                    match this.inner.as_mut().poll_next(cx) {
                        Poll::Ready(Some(Ok(chunk))) => return Poll::Ready(Some(Ok(chunk))),
                        Poll::Ready(Some(Err(e))) => {
                            let e = e.into();
                            if let OpenAIAdapterError::ToolCallRepairNeeded(raw_xml) = e {
                                debug!(target: "adapter", "RepairStream catching repair signal");
                                let repair_fn = this.repair_fn.clone();
                                let future = (repair_fn)(raw_xml);
                                *this.state = RepairState::Repairing { future };
                                continue;
                            }
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(None) => {
                            *this.state = RepairState::Done;
                            return Poll::Ready(None);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                RepairState::Repairing { future } => {
                    match future.as_mut().poll(cx) {
                        Poll::Ready(Ok(calls)) => {
                            debug!(target: "adapter", "RepairStream repair succeeded with {} calls", calls.len());
                            // Emit repair result as tool_calls chunk then resume forwarding
                            let chunk = make_repair_chunk(calls);
                            *this.state = RepairState::Forwarding;
                            return Poll::Ready(Some(Ok(chunk)));
                        }
                        Poll::Ready(Err(e)) => {
                            error!(target: "adapter", "RepairStream repair failed: {}", e);
                            *this.state = RepairState::RepairFailed(e.to_string());
                            continue;
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                RepairState::RepairFailed(msg) => {
                    let msg = std::mem::take(msg);
                    return Poll::Ready(Some(Err(OpenAIAdapterError::Internal(msg))));
                }
                RepairState::Done => return Poll::Ready(None),
            }
        }
    }
}

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Build a synthetic tool_calls chunk from repaired call list.
fn make_repair_chunk(calls: Vec<ToolCall>) -> ChatCompletionChunk {
    let model = calls.first()
        .and_then(|c| c.function.as_ref())
        .map(|_| "".to_string())
        .unwrap_or_default();
    ChatCompletionChunk {
        id: format!("chatcmpl-repair-{}", CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)),
        object: "chat.completion.chunk",
        created: 0,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
                tool_calls: Some(calls),
                reasoning_content: None,
                ..Default::default()
            },
            finish_reason: Some("tool_calls"),
            logprobs: None,
        }],
        usage: None,
        service_tier: None,
        system_fingerprint: None,
    }
}

pin_project! {
    struct StopStream<S> {
        #[pin]
        inner: S,
        stop: Vec<String>,
        stopped: bool,
        sent_len: usize,
        buffer: String,
        include_obfuscation: bool,
    }
}

impl<S> Stream for StopStream<S>
where
    S: Stream<Item = Result<ChatCompletionChunk, OpenAIAdapterError>> + Unpin,
{
    type Item = Result<Bytes, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(mut chunk))) => {
                    if *this.stopped {
                        if chunk.choices.is_empty() && chunk.usage.is_some() {
                            return Poll::Ready(Some(chunk_to_bytes(
                                chunk,
                                *this.include_obfuscation,
                            )));
                        }
                        // Allow upgrading `finish_reason` from `stop` to `tool_calls` after parsing tools
                        if let Some(choice) = chunk.choices.first_mut()
                            && choice.delta.content.is_none()
                            && choice.delta.reasoning_content.is_none()
                            && choice.delta.tool_calls.is_none()
                            && choice.finish_reason == Some(FINISH_TOOL_CALLS)
                        {
                            return Poll::Ready(Some(chunk_to_bytes(
                                chunk,
                                *this.include_obfuscation,
                            )));
                        }
                        continue;
                    }

                    if let Some(choice) = chunk.choices.first_mut()
                        && let Some(ref content) = choice.delta.content
                    {
                        this.buffer.push_str(content);
                        if let Some(pos) = find_stop_pos(this.buffer, this.stop) {
                            let truncated = &this.buffer[*this.sent_len..pos];
                            if truncated.is_empty() {
                                choice.delta.content = None;
                            } else {
                                choice.delta.content = Some(truncated.to_string());
                            }
                            choice.finish_reason = Some(FINISH_STOP);
                            *this.stopped = true;
                            this.buffer.clear();
                            *this.sent_len = pos;
                        } else {
                            *this.sent_len = this.buffer.len();
                        }
                    }
                    return Poll::Ready(Some(chunk_to_bytes(chunk, *this.include_obfuscation)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Streaming path: convert the `ds_core` byte stream into OpenAI-style `data: {json}` SSE
pub(crate) fn stream<S>(
    ds_stream: S,
    model: String,
    include_usage: bool,
    include_obfuscation: bool,
    stop: Vec<String>,
    prompt_tokens: u32,
    tools_present: bool,
    repair_fn: Option<RepairFn>,
) -> StreamResponse
where
    S: Stream<Item = Result<Bytes, crate::ds_core::CoreError>> + Send + 'static,
{
    debug!(
        target: "adapter",
        "building streaming response: model={}, include_usage={}, include_obfuscation={}, stop_count={}, repair={}",
        model, include_usage, include_obfuscation, stop.len(), repair_fn.is_some()
    );
    let sse = sse_parser::SseStream::new(ds_stream);
    let state_stream = state::StateStream::new(sse);
    let converted = converter::ConverterStream::new(
        state_stream,
        model.clone(),
        include_usage,
        include_obfuscation,
        prompt_tokens,
       tools_present,
    );
    let tool_parsed = tool_parser::ToolCallStream::new(converted, model);

    // Wrap with RepairStream if repair function is provided
    let inner_stream: Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, OpenAIAdapterError>> + Send>> =
        if let Some(rf) = repair_fn {
            Box::pin(RepairStream::new(tool_parsed, rf))
        } else {
            Box::pin(tool_parsed)
        };

    let stop_stream = StopStream {
        inner: inner_stream,
        stop,
        stopped: false,
        sent_len: 0,
        buffer: String::new(),
        include_obfuscation,
    };
    Box::pin(stop_stream)
}

/// Buffered path: fuse the internal SSE stream down to one `chat.completion` payload
pub(crate) async fn aggregate<S>(
    ds_stream: S,
    model: String,
    stop: Vec<String>,
    prompt_tokens: u32,
    tools_present: bool,
) -> Result<Vec<u8>, OpenAIAdapterError>
where
    S: Stream<Item = Result<Bytes, crate::ds_core::CoreError>> + Send,
{
    debug!(target: "adapter", "building buffered response: model={}, stop_count={}", model, stop.len());
    let sse = sse_parser::SseStream::new(ds_stream);
    let state_stream = state::StateStream::new(sse);
    let converted =
       converter::ConverterStream::new(state_stream, model.clone(), true, false, prompt_tokens, tools_present);

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = None;
    let mut finish_reason = None;

    futures::pin_mut!(converted);
    while let Some(res) = converted.next().await {
        let chunk = res?;
        if let Some(u) = chunk.usage {
            usage = Some(u);
        }
        if let Some(choice) = chunk.choices.into_iter().next() {
            if finish_reason.is_none() {
                finish_reason = choice.finish_reason.map(|s| s.to_string());
            }
            if let Some(c) = choice.delta.content {
                content.push_str(&c);
            }
            if let Some(r) = choice.delta.reasoning_content {
                reasoning.push_str(&r);
            }
        }
    }

    let stop_pos = if !stop.is_empty() {
        find_stop_pos(&content, &stop)
    } else {
        None
    };

    let parsed = tool_parser::parse_tool_calls(&content);

    // Stop-sequence trimming (ignored once tool calls are materialized)
    if let Some(pos) = stop_pos
        && parsed.is_none()
    {
        content.truncate(pos);
        finish_reason = Some(FINISH_STOP.to_string());
    }

    let (message_content, tool_calls) = if let Some((calls, remaining)) = parsed {
        let tail = remaining.trim();
        if tail.is_empty() {
            (None, Some(calls))
        } else {
            (Some(tail.to_string()), Some(calls))
        }
    } else {
        let c = if content.is_empty() {
            None
        } else {
            Some(content)
        };
        (c, None)
    };

    let final_reason: Option<&'static str> = if tool_calls.is_some() {
        Some(FINISH_TOOL_CALLS)
    } else if finish_reason.as_deref() == Some(FINISH_STOP) {
        Some(FINISH_STOP)
    } else {
        None
    };

    let completion = ChatCompletion {
        id: next_chatcmpl_id(),
        object: "chat.completion",
        created: now_secs(),
        model,
        choices: vec![Choice {
            index: 0,
            message: MessageResponse {
                role: "assistant",
                content: message_content,
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
                refusal: None,
                annotations: None,
                audio: None,
                function_call: None,
                tool_calls,
            },
            finish_reason: final_reason,
            logprobs: None,
        }],
        usage,
        service_tier: None,
        system_fingerprint: None,
    };

    let json = serde_json::to_vec(&completion)?;
    debug!(
        target: "adapter",
        "buffered completion synthesized: finish_reason={:?}, has_tool_calls={}, usage={:?}",
        completion.choices[0].finish_reason,
        completion.choices[0].message.tool_calls.is_some(),
        completion.usage
    );
    Ok(json)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::StreamExt;

    use super::*;

    fn sse_bytes(body: &str) -> Result<Bytes, crate::ds_core::CoreError> {
        Ok(Bytes::from(body.to_string()))
    }

    #[tokio::test]
    async fn aggregate_plain_text() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"message_id\":2,\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"hello\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\" world\"}\n\n\
            data: {\"p\":\"response\",\"o\":\"BATCH\",\"v\":[{\"p\":\"accumulated_token_usage\",\"v\":41},{\"p\":\"quasi_status\",\"v\":\"FINISHED\"}]}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-default".into(), vec![], 0, false)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&json).unwrap();
        println!("\n=== AGGREGATED RESPONSE (plain_text) ===");
        println!("{}", serde_json::to_string_pretty(&completion).unwrap());
        println!("=========================================\n");
        assert_eq!(completion["object"], "chat.completion");
        assert_eq!(completion["model"], "deepseek-default");
        assert_eq!(
            completion["choices"][0]["message"]["content"],
            "hello world"
        );
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
        assert_eq!(completion["usage"]["completion_tokens"], 41);
    }

    #[tokio::test]
    async fn aggregate_thinking() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"message_id\":2,\"fragments\":[{\"type\":\"THINK\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"thinking\"}\n\n\
            data: {\"p\":\"response/fragments/-1/elapsed_secs\",\"o\":\"SET\",\"v\":0.95}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"answer\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-expert".into(), vec![], 0, false)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&json).unwrap();
        println!("\n=== AGGREGATED RESPONSE (thinking) ===");
        println!("{}", serde_json::to_string_pretty(&completion).unwrap());
        println!("=======================================\n");
        assert_eq!(
            completion["choices"][0]["message"]["reasoning_content"],
            "thinking"
        );
        assert_eq!(completion["choices"][0]["message"]["content"], "answer");
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn aggregate_tool_calls() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"beijing\\\"}}]</tool_calls>\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-default".into(), vec![], 0, false)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&json).unwrap();
        println!("\n=== AGGREGATED RESPONSE (tool_calls) ===");
        println!("{}", serde_json::to_string_pretty(&completion).unwrap());
        println!("=========================================\n");
        assert!(completion["choices"][0]["message"]["content"].is_null());
        let calls = completion["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"beijing"}"#);
        assert_eq!(completion["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn aggregate_tool_calls_with_trailing_text() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {}}]</tool_calls> trailing text\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-default".into(), vec![], 0, false)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&json).unwrap();
        println!("\n=== AGGREGATED RESPONSE (tool_calls + trailing text) ===");
        println!("{}", serde_json::to_string_pretty(&completion).unwrap());
        println!("========================================================\n");
        assert_eq!(
            completion["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            "trailing text"
        );
        let calls = completion["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(completion["choices"][0]["finish_reason"], "tool_calls");
    }

    async fn collect_chunks(st: StreamResponse) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        let mut st = st;
        while let Some(res) = st.next().await {
            let text = String::from_utf8(res.unwrap().to_vec()).unwrap();
            let json = text
                .strip_prefix("data: ")
                .unwrap()
                .strip_suffix("\n\n")
                .unwrap();
            out.push(serde_json::from_str(json).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn stream_plain_text() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"hi\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (plain_text) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("===================================\n");
        // With W=19, the two-byte "hi" lingers until the rolling window releases it,
        // so terminal metadata may piggyback that same chunk.
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // Reassembled assistant text should read "hi".
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(all_content, "hi");
        // Final finish reason after coalescing deltas.
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[tokio::test]
    async fn stream_include_usage() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"x\"}\n\n\
            data: {\"p\":\"response\",\"o\":\"BATCH\",\"v\":[{\"p\":\"accumulated_token_usage\",\"v\":12}]}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            true,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (include_usage) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("======================================\n");
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // Reassembled assistant text should read "x".
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(all_content, "x");
        // usage chunk
        let usage_chunk = chunks
            .iter()
            .find(|c| c["usage"]["completion_tokens"].as_i64() == Some(12));
        assert!(usage_chunk.is_some(), "should have usage chunk");
        // `finish_reason` rides on the final chunk that still reports `choices`.
        let finish_chunk = chunks.iter().rev().find(|c| {
            c["choices"].as_array().map_or(false, |a| !a.is_empty())
                && c["choices"][0]["finish_reason"].as_str().is_some()
        });
        assert_eq!(finish_chunk.unwrap()["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn stream_tool_calls() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"f\\\", \\\"arguments\\\": {}}]</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (tool_calls) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("===================================\n");
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // At least one surfaced chunk should emit `tool_calls`.
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk");
        // Assistant text must not leak raw `<tool_calls>` markup.
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert!(
            !all_content.contains("<tool_calls>"),
            "content should not contain tool_calls tags"
        );
        // finish
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[tokio::test]
    async fn stream_fragmented_tool_calls_with_thinking() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"Thinking...\"}]}}}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"get_\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"Beijing\\\"}}]\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (fragmented_tool_calls_with_thinking) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("============================================================\n");
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // `reasoning_content` aggregates THINK fragments verbatim.
        let all_reasoning: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str())
            .collect();
        assert!(all_reasoning.contains("Thinking..."), "expected THINK fragment `Thinking...`");
        // At least one surfaced chunk should emit `tool_calls`.
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk");
        let tc_chunk = chunks
            .iter()
            .find(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some())
            .unwrap();
        let calls = tc_chunk["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Beijing"}"#);
        // finish
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[tokio::test]
    async fn stream_with_tool_search_and_open() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"Thinking\"}]}}}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"id\":3,\"type\":\"TOOL_SEARCH\",\"content\":null,\"queries\":[{\"query\":\"q\"}],\"results\":[],\"stage_id\":1}]}\n\n\
            data: {\"p\":\"response/fragments/-2/results\",\"o\":\"SET\",\"v\":[{\"url\":\"https://example.com\",\"title\":\"ex\",\"snippet\":\"snip\"}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"id\":4,\"type\":\"TOOL_OPEN\",\"status\":\"WIP\",\"result\":{\"url\":\"https://open.com\",\"title\":\"open\",\"snippet\":\"open-snippet\"},\"reference\":{\"id\":3,\"type\":\"TOOL_SEARCH\"},\"stage_id\":1}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"THINK\",\"content\":\"Continuing\"}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"hello\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (tool_search_and_open) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("=============================================\n");
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // Concatenated reasoning should include both THINK snippets.
        let all_reasoning: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str())
            .collect();
        assert!(all_reasoning.contains("Thinking"), "expected substring `Thinking`");
        assert!(all_reasoning.contains("Continuing"), "expected substring `Continuing`");
        // Assistant text should concatenate to "hello".
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(all_content, "hello");
        // finish_reason
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[tokio::test]
    async fn stream_include_obfuscation() {
        // Use a long fragment (>W chars) to exercise obfuscation + buffering
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"This is a sufficiently long test string for obfuscation testing purposes\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            true,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (include_obfuscation) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!(
                "chunk[{i}] len={}:\n{}",
                serde_json::to_string(c).unwrap().len(),
                serde_json::to_string_pretty(c).unwrap()
            );
        }
        println!("============================================\n");
        assert!(chunks.len() >= 2);
        // Every chunk with both `choices` and `content` should hover near the padding target length
        for c in &chunks {
            if c["choices"][0]["delta"]["content"].as_str().is_some()
                || c["choices"][0]["finish_reason"].as_str().is_some()
            {
                assert!(
                    c["choices"][0]["delta"]["obfuscation"].as_str().is_some(),
                    "chunk with content or finish_reason should have obfuscation"
                );
                let len = serde_json::to_string(c).unwrap().len();
                assert!(
                    len >= 490 && len <= 530,
                    "chunk len {} out of expected 490..=530 range",
                    len
                );
            }
        }
        // Preserve full assistant text after stripping padding.
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert!(
            all_content.contains("sufficiently long test string"),
            "should contain expected text, got {all_content:?}"
        );
        // finish_reason
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[tokio::test]
    async fn aggregate_tool_calls_with_leading_text() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"Alright, I will help you.\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"beijing\\\"}}]</tool_calls>\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-default".into(), vec![], 0, false)
            .await
            .unwrap();
        let completion: serde_json::Value = serde_json::from_slice(&json).unwrap();
        println!("\n=== AGGREGATED RESPONSE (tool_calls with leading text) ===");
        println!("{}", serde_json::to_string_pretty(&completion).unwrap());
        println!("===========================================================\n");
        // Leading conversational text streams as `content`, then tooling JSON follows
        assert_eq!(
            completion["choices"][0]["message"]["content"],
            "Alright, I will help you."
        );
        let calls = completion["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"beijing"}"#);
        assert_eq!(completion["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_tool_calls_with_leading_text_fragmented() {
            // Scenario: leading natural language + chunked `<tool_calls>` JSON
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"Alright, I will help you generate the image.\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<too\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"l_calls>[{\\\"name\\\": \\\"astrbot_execute_shell\\\", \\\"arguments\\\": {\\\"command\\\": \\\"cat /data/astrbot/skills/doubao-image-gen/SKILL.md\\\"}}]</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (tool_calls with leading text, fragmented) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("====================================================================\n");
        // Verify we surface leading text, parsed tool calls, and final finish reasons
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // Reassembled text should still include the conversational prefix.
        let all_content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert!(
            all_content.contains("Alright, I will help you generate the image"),
            "should contain leading text, got {all_content:?}"
        );
        // At least one surfaced chunk should emit `tool_calls`.
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk");
        let tc_chunk = chunks
            .iter()
            .find(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some())
            .unwrap();
        let calls = tc_chunk["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "astrbot_execute_shell");
        // finish
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_tool_calls_with_leading_text_multi_chunk_fragments() {
        // Heavier fragmentation splits JSON across deltas.
        // chunk 1: leading text
        // chunk 2: <tool_calls>[{"name": "f", "arguments": {}}
        // chunk 3: ]
        // chunk 4: </tool_calls>
        // chunk 5: FINISHED status
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"Let me look it up.\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"f\\\", \\\"arguments\\\": {}}\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"]\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (leading text + multi-chunk JSON fragments) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("=============================================================\n");
        // Expect role chunk, leading text, parsed tool calls, then finish metadata
        for (i, c) in chunks.iter().enumerate() {
            eprintln!(
                "chunk[{}] content={:?} tool_calls={:?} finish={:?}",
                i,
                c["choices"][0]["delta"]["content"],
                c["choices"][0]["delta"]["tool_calls"],
                c["choices"][0]["finish_reason"]
            );
        }
        // A dedicated chunk must advertise `tool_calls`.
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk but didn't");
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_tool_calls_with_thinking_then_leading_text_then_fragmented_json() {
            // Full production mix: THINK → leading content → fragmented `<tool_calls>`
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"User wants weather info, I need to call the tool\"}]}}}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"Alright, let me look it up\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"...\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"_calls>[{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"beijing\\\"}}\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"]</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (thinking + leading + fragmented JSON) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("=============================================================\n");
        for (i, c) in chunks.iter().enumerate() {
            eprintln!(
                "chunk[{}] content={:?} reasoning={:?} tool_calls={:?} finish={:?}",
                i,
                c["choices"][0]["delta"]["content"],
                c["choices"][0]["delta"]["reasoning_content"],
                c["choices"][0]["delta"]["tool_calls"],
                c["choices"][0]["finish_reason"]
            );
        }
        // A dedicated chunk must advertise `tool_calls`.
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk but didn't");
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_tool_calls_json_split_right_after_tag() {
            // `<tool_calls>` opens early while JSON arguments finish in later deltas
        // chunk 1: leading text
        // chunk 2: <tool_calls>[{"name": "f", "arguments": {}}]
            // Third chunk may only carry the closing `</tool_calls>` sentinel
        // chunk 4: FINISHED
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"Alright.\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"f\\\", \\\"arguments\\\": {}}]\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "m".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (JSON split right after tool_call) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("=============================================================\n");
        let has_tool_calls = chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some());
        assert!(has_tool_calls, "should have a tool_calls chunk but didn't");
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_tool_calls_no_leading_text() {
            // Common degenerate case — model jumps straight into `<tool_calls>`
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls>[{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"beijing\\\"}}]</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(
            bytes_stream,
            "deepseek-default".into(),
            false,
            false,
            vec![],
            0,
            false,
            None,
        ))
        .await;
        println!("\n=== STREAM CHUNKS (tool_calls, no leading text) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("===================================================\n");
        // Expect role scaffolding, tool deltas, then canonical finish chunk
        for (i, c) in chunks.iter().enumerate() {
            eprintln!(
                "chunk[{}] content={:?} tool_calls={:?} finish={:?}",
                i,
                c["choices"][0]["delta"]["content"],
                c["choices"][0]["delta"]["tool_calls"],
                c["choices"][0]["finish_reason"]
            );
        }
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        // Locate the delta that actually lists `tool_calls`.
        let tc_idx = chunks
            .iter()
            .position(|c| c["choices"][0]["delta"]["tool_calls"].as_array().is_some())
            .expect("should have a chunk with tool_calls");
        let tc_chunk = &chunks[tc_idx];
        let calls = tc_chunk["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"beijing"}"#);
        // Final finish reason should advertise `tool_calls` when tools win
        let last = chunks.last().unwrap();
        assert_eq!(
            last["choices"][0]["finish_reason"], "tool_calls",
            "finish_reason should be tool_calls, got {:?}",
            last["choices"][0]["finish_reason"]
        );
    }
}
