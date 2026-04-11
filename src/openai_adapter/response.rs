//! OpenAI 响应转换 —— 将 DeepSeek SSE 流映射为 OpenAI 响应格式
//!
//! 数据流：sse_parser -> state -> converter -> tool_parser
//! - 仅 THINK / RESPONSE 片段映射到用户可见文本
//! - obfuscation 在最终 SSE 序列化阶段动态注入

pub mod converter;
pub mod sse_parser;
pub mod state;
pub mod tool_parser;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use log::debug;

use crate::openai_adapter::{
    OpenAIAdapterError, StreamResponse,
    types::{ChatCompletion, ChatCompletionChunk, Choice, MessageResponse},
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
    let bytes: Vec<u8> = (0..byte_len).map(|_| rand::random()).collect();
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

/// 流式响应：把 ds_core 字节流转换为 OpenAI SSE 字节流
pub fn stream<S>(
    ds_stream: S,
    model: String,
    include_usage: bool,
    include_obfuscation: bool,
) -> StreamResponse
where
    S: Stream<Item = Result<Bytes, crate::ds_core::CoreError>> + Send + 'static,
{
    debug!(
        target: "adapter",
        "构建流式响应: model={}, include_usage={}, include_obfuscation={}",
        model, include_usage, include_obfuscation
    );
    let sse = sse_parser::SseStream::new(ds_stream);
    let state_stream = state::StateStream::new(sse);
    let converted = converter::ConverterStream::new(
        state_stream,
        model.clone(),
        include_usage,
        include_obfuscation,
    );
    let tool_parsed = tool_parser::ToolCallStream::new(converted, model);
    Box::pin(tool_parsed.map(move |res| match res {
        Ok(chunk) => chunk_to_bytes(chunk, include_obfuscation),
        Err(e) => Err(e),
    }))
}

/// 非流式响应：聚合 SSE 流为单个 ChatCompletion JSON
pub async fn aggregate<S>(ds_stream: S, model: String) -> Result<Vec<u8>, OpenAIAdapterError>
where
    S: Stream<Item = Result<Bytes, crate::ds_core::CoreError>> + Send,
{
    debug!(target: "adapter", "构建非流式响应: model={}", model);
    let sse = sse_parser::SseStream::new(ds_stream);
    let state_stream = state::StateStream::new(sse);
    let converted = converter::ConverterStream::new(state_stream, model.clone(), true, false);

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

    let (message_content, tool_calls) = if let Some(calls) = tool_parser::parse_tool_calls(&content)
    {
        (None, Some(calls))
    } else {
        let c = if content.is_empty() {
            None
        } else {
            Some(content)
        };
        (c, None)
    };

    let is_stop = finish_reason.as_deref() == Some(FINISH_STOP);
    let final_reason: Option<&'static str> = if tool_calls.is_some() {
        Some(FINISH_TOOL_CALLS)
    } else if is_stop {
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
        "非流式响应聚合完成: finish_reason={:?}, has_tool_calls={}, usage={:?}",
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
        let json = aggregate(stream, "deepseek-default".into()).await.unwrap();
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
        let json = aggregate(stream, "deepseek-expert".into()).await.unwrap();
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
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls><tool_call name=\\\"get_weather\\\" arguments=\\\"{&quot;city&quot;:&quot;beijing&quot;}\\\" /></tool_calls>\"}\n\n\
            event: finish\ndata: {}\n\n";
        let stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let json = aggregate(stream, "deepseek-default".into()).await.unwrap();
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
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), false, false)).await;
        println!("\n=== STREAM CHUNKS (plain_text) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("===================================\n");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "hi");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
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
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), true, false)).await;
        println!("\n=== STREAM CHUNKS (include_usage) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("======================================\n");
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "x");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        assert!(chunks[3]["choices"].as_array().unwrap().is_empty());
        assert_eq!(chunks[3]["usage"]["completion_tokens"], 12);
    }

    #[tokio::test]
    async fn stream_tool_calls() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls><tool_call name=\\\"f\\\" arguments=\\\"{}\\\" /></tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), false, false)).await;
        println!("\n=== STREAM CHUNKS (tool_calls) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("===================================\n");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(chunks[1]["choices"][0]["delta"]["content"].is_null());
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_fragmented_tool_calls_with_thinking() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"思考中\"}]}}}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"<tool_calls><tool_call name=\\\"get_\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"weather\\\" arguments='{\\\"city\\\":\\\"北京\\\"}' />\"}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"</tool_calls>\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), false, false)).await;
        println!("\n=== STREAM CHUNKS (fragmented_tool_calls_with_thinking) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("============================================================\n");
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning_content"],
            "思考中"
        );
        assert!(chunks[2]["choices"][0]["delta"]["content"].is_null());
        let calls = chunks[2]["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"北京"}"#);
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn stream_with_tool_search_and_open() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"思考\"}]}}}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"id\":3,\"type\":\"TOOL_SEARCH\",\"content\":null,\"queries\":[{\"query\":\"q\"}],\"results\":[],\"stage_id\":1}]}\n\n\
            data: {\"p\":\"response/fragments/-2/results\",\"o\":\"SET\",\"v\":[{\"url\":\"https://example.com\",\"title\":\"ex\",\"snippet\":\"snip\"}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"id\":4,\"type\":\"TOOL_OPEN\",\"status\":\"WIP\",\"result\":{\"url\":\"https://open.com\",\"title\":\"open\",\"snippet\":\"open-snippet\"},\"reference\":{\"id\":3,\"type\":\"TOOL_SEARCH\"},\"stage_id\":1}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"THINK\",\"content\":\"继续\"}]}\n\n\
            data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"hello\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), false, false)).await;
        println!("\n=== STREAM CHUNKS (tool_search_and_open) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!("chunk[{i}]:\n{}", serde_json::to_string_pretty(c).unwrap());
        }
        println!("=============================================\n");
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning_content"],
            "思考"
        );
        assert_eq!(
            chunks[2]["choices"][0]["delta"]["reasoning_content"],
            "继续"
        );
        assert_eq!(chunks[3]["choices"][0]["delta"]["content"], "hello");
        assert_eq!(chunks[4]["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn stream_include_obfuscation() {
        let fixture = "event: ready\ndata: {}\n\n\
            data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"RESPONSE\",\"content\":\"\"}]}}}\n\n\
            data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\"x\"}\n\n\
            data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n\
            event: finish\ndata: {}\n\n";
        let bytes_stream = futures::stream::iter(vec![sse_bytes(fixture)]);
        let chunks = collect_chunks(super::stream(bytes_stream, "m".into(), false, true)).await;
        println!("\n=== STREAM CHUNKS (include_obfuscation) ===");
        for (i, c) in chunks.iter().enumerate() {
            println!(
                "chunk[{i}] len={}:\n{}",
                serde_json::to_string(c).unwrap().len(),
                serde_json::to_string_pretty(c).unwrap()
            );
        }
        println!("============================================\n");
        assert_eq!(chunks.len(), 3);
        // 所有含 choices 的 chunk 都应被动态 padding 到目标长度附近
        for c in &chunks {
            assert!(c["choices"][0]["delta"]["obfuscation"].as_str().is_some());
            let len = serde_json::to_string(c).unwrap().len();
            assert!(
                len >= 490 && len <= 530,
                "chunk len {} out of expected 490..=530 range",
                len
            );
        }
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "x");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    }
}
