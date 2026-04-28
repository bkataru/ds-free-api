//! Anthropic → OpenAI request mapping for `/v1/messages`.
//!
//! Unsupported knobs (`top_k`, `cache_control`, …) deserialize for compatibility but are dropped before the core pipeline.

#![allow(dead_code)]

use log::debug;

use serde::Deserialize;
use serde_json::json;

use crate::anthropic_compat::AnthropicCompatError;

// ============================================================================
// Anthropic Message API wire types
// ============================================================================

/// Incoming `POST /v1/messages` document.
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<MessageParam>,
    pub max_tokens: u32,

    #[serde(default)]
    pub system: Option<SystemContent>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub tools: Option<Vec<ToolUnion>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub metadata: Option<Metadata>,
    #[serde(default)]
    pub output_config: Option<OutputConfig>,
    /// Web search controls (Anthropic extension; forwarded as OpenAI `web_search_options` when present).
    #[serde(default)]
    pub web_search_options: Option<serde_json::Value>,

    // Accepted for forward compatibility — parsed but intentionally ignored downstream.
    #[serde(default)]
    pub cache_control: Option<CacheControlEphemeral>,
    #[serde(default)]
    pub container: Option<String>,
    #[serde(default)]
    pub inference_geo: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,

    // Passthrough bucket for unknown keys.
    #[serde(flatten)]
    pub _extra: serde_json::Value,
}

/// One role/content pair from the Anthropic `messages` array.
#[derive(Debug, Deserialize, Clone)]
pub struct MessageParam {
    pub role: String,
    pub content: MessageContent,
}

/// Either a plain string or a structured block list.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Anthropic multimodal / tool blocks.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    // Everything else (documents, search results, server tools, …) is ignored.
    #[serde(other)]
    Other,
}

/// Inline image payload.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { data: String, media_type: String },
    #[serde(rename = "url")]
    Url { url: String },
}

/// `tool_result` body: string or nested blocks.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// `system` prompt: string or structured blocks.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemTextBlock>),
}

/// Structured `system` block — we only keep `text` (ignores cache/citations metadata).
#[derive(Debug, Deserialize, Clone)]
pub struct SystemTextBlock {
    pub text: String,
    #[serde(rename = "type")]
    pub ty: String,
}

/// Union of custom tools vs server-defined tools.
#[derive(Debug, Clone)]
pub enum ToolUnion {
    Custom {
        name: String,
        description: Option<String>,
        input_schema: serde_json::Value,
        strict: Option<bool>,
    },
    // Server-managed tools (bash, code execution, web search, …) are ignored here.
    Other,
}

impl<'de> serde::Deserialize<'de> for ToolUnion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("tool must be an object"))?;

        match obj.get("type").and_then(|v| v.as_str()) {
            Some("custom") | None => {
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?;
                let description = obj
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let input_schema = obj.get("input_schema").cloned().unwrap_or_default();
                let strict = obj.get("strict").and_then(|v| v.as_bool());
                Ok(ToolUnion::Custom {
                    name,
                    description,
                    input_schema,
                    strict,
                })
            }
            Some(_) => Ok(ToolUnion::Other),
        }
    }
}

/// Tool routing configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    #[serde(rename = "any")]
    Any {
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    #[serde(rename = "tool")]
    Tool {
        name: String,
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    #[serde(rename = "none")]
    None,
}

/// Extended thinking budget / display hints.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ThinkingConfig {
    #[serde(rename = "enabled")]
    Enabled {
        budget_tokens: u32,
        #[serde(default)]
        display: Option<String>,
    },
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "adaptive")]
    Adaptive {
        #[serde(default)]
        display: Option<String>,
    },
}

/// Optional request metadata.
#[derive(Debug, Deserialize, Clone)]
pub struct Metadata {
    #[serde(default)]
    pub user_id: Option<String>,
}

/// Response shaping hints (effort, JSON schema, …).
#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub format: Option<JsonOutputFormat>,
}

/// JSON schema contract for structured outputs.
#[derive(Debug, Deserialize, Clone)]
pub struct JsonOutputFormat {
    pub schema: serde_json::Value,
    #[serde(rename = "type")]
    pub ty: String,
}

/// `cache_control` (accepted for compatibility only).
#[derive(Debug, Deserialize, Clone)]
pub struct CacheControlEphemeral {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub ttl: Option<String>,
}

// ============================================================================
// Mapping entry points
// ============================================================================

/// Convert an Anthropic Messages JSON payload into an OpenAI chat-completions JSON payload.
pub fn to_openai_request(body: &[u8]) -> Result<Vec<u8>, AnthropicCompatError> {
    let req: MessagesRequest = serde_json::from_slice(body)
        .map_err(|e| AnthropicCompatError::BadRequest(format!("bad request: {}", e)))?;
    debug!(target: "anthropic_compat::request", "parsed request: model={}, messages={}, stream={}", req.model, req.messages.len(), req.stream);

    let mut openai = serde_json::Map::new();

    openai.insert("model".to_string(), json!(req.model));
    openai.insert("max_tokens".to_string(), json!(req.max_tokens));

    // Prepend optional `system` content, then map each Anthropic message.
    let mut messages = Vec::new();
    if let Some(system) = req.system {
        messages.push(system_to_openai(&system));
    }
    for msg in &req.messages {
        messages.extend(message_param_to_openai(msg));
    }
    openai.insert("messages".to_string(), json!(messages));

    // stream
    if req.stream {
        openai.insert("stream".to_string(), json!(true));
    }

    // stop_sequences -> stop
    if let Some(stop) = req.stop_sequences
        && !stop.is_empty()
    {
        openai.insert("stop".to_string(), json!(stop));
    }

    // temperature
    if let Some(t) = req.temperature {
        openai.insert("temperature".to_string(), json!(t));
    }

    // top_p
    if let Some(p) = req.top_p {
        openai.insert("top_p".to_string(), json!(p));
    }

    // `top_k` has no OpenAI equivalent — drop it.

    // tools
    let mut parallel_tool_calls_disabled = false;
    if let Some(tools) = req.tools {
        let openai_tools: Vec<serde_json::Value> =
            tools.iter().filter_map(tool_union_to_openai).collect();
        if !openai_tools.is_empty() {
            openai.insert("tools".to_string(), json!(openai_tools));
        }
    }

    // tool_choice
    if let Some(tc) = req.tool_choice {
        parallel_tool_calls_disabled = tc.disable_parallel();
        openai.insert("tool_choice".to_string(), tc.to_openai());
    }

    if parallel_tool_calls_disabled {
        openai.insert("parallel_tool_calls".to_string(), json!(false));
    }

    // thinking -> reasoning_effort
    if let Some(thinking) = req.thinking {
        let effort = match thinking {
            ThinkingConfig::Enabled { .. } | ThinkingConfig::Adaptive { .. } => "high",
            ThinkingConfig::Disabled => "none",
        };
        openai.insert("reasoning_effort".to_string(), json!(effort));
    }

    // output_config.format -> response_format
    if let Some(output_config) = req.output_config
        && let Some(fmt) = output_config.format
    {
        openai.insert(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": fmt.schema
            }),
        );
    }

    // web_search_options -> web_search_options
    if let Some(opts) = req.web_search_options {
        openai.insert("web_search_options".to_string(), opts);
    }

    serde_json::to_vec(&openai)
        .map_err(|e| AnthropicCompatError::Internal(format!("json error: {}", e)))
}

// ============================================================================
// Helpers
// ============================================================================

fn system_to_openai(system: &SystemContent) -> serde_json::Value {
    let text = match system {
        SystemContent::Text(t) => t.clone(),
        SystemContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| b.text.clone())
            .collect::<Vec<_>>()
            .join("\n"),
    };
    json!({"role": "system", "content": text})
}

fn message_param_to_openai(msg: &MessageParam) -> Vec<serde_json::Value> {
    let blocks = match &msg.content {
        MessageContent::Text(t) => {
            return vec![json!({"role": msg.role, "content": t})];
        }
        MessageContent::Blocks(b) => b,
    };

    match msg.role.as_str() {
        "assistant" => assistant_blocks_to_openai(blocks),
        "user" => user_blocks_to_openai(blocks),
        _ => {
            // Non-standard roles stringify into generic user/assistant content.
            let text = extract_text_from_blocks(blocks);
            vec![json!({"role": msg.role, "content": text})]
        }
    }
}

/// Map assistant blocks into a single OpenAI assistant message.
/// - `text` aggregates into `content`
/// - `tool_use` becomes parallel `tool_calls`
/// - Thinking / images / tool results / unknown blocks are ignored
fn assistant_blocks_to_openai(blocks: &[ContentBlock]) -> Vec<serde_json::Value> {
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => texts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string()
                    }
                }));
            }
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolResult { .. }
            | ContentBlock::Other => {}
        }
    }

    let mut msg = serde_json::Map::new();
    msg.insert("role".to_string(), json!("assistant"));

    let content = if texts.is_empty() {
        json!(null)
    } else {
        json!(texts.join("\n"))
    };
    msg.insert("content".to_string(), content);

    if !tool_calls.is_empty() {
        msg.insert("tool_calls".to_string(), json!(tool_calls));
    }

    vec![json!(msg)]
}

/// Map user blocks into one or more OpenAI-compatible messages.
/// - Inline text merges into `content`
/// - Images become `image_url` parts when present
/// - Tool results synthesize standalone `tool` role messages
/// - Thinking and unknown blocks are ignored
fn user_blocks_to_openai(blocks: &[ContentBlock]) -> Vec<serde_json::Value> {
    let mut text_parts = Vec::new();
    let mut image_parts = Vec::new();
    let mut tool_results = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::Image { source } => {
                let url = match source {
                    ImageSource::Base64 { data, media_type } => {
                        format!("data:{};base64,{}", media_type, data)
                    }
                    ImageSource::Url { url } => url.clone(),
                };
                image_parts.push(url);
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                let text = match content {
                    Some(ToolResultContent::Text(t)) => t.clone(),
                    Some(ToolResultContent::Blocks(b)) => extract_text_from_blocks(b),
                    None => String::new(),
                };
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": text
                }));
            }
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::ToolUse { .. }
            | ContentBlock::Other => {}
        }
    }

    let mut result = Vec::new();

    // Fold text plus inline images into a single user message.
    if !text_parts.is_empty() || !image_parts.is_empty() {
        if image_parts.is_empty() {
            // Pure text shards merge via `join("\n")`.
            result.push(json!({"role": "user", "content": text_parts.join("\n")}));
        } else {
            // Images require a multimodal parts array.
            let mut parts = Vec::new();
            for text in &text_parts {
                parts.push(json!({"type": "text", "text": text}));
            }
            for url in &image_parts {
                parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
            result.push(json!({"role": "user", "content": parts}));
        }
    }

    // Each `tool_result` becomes its own `role: "tool"` message.
    result.extend(tool_results);

    result
}

fn extract_text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_union_to_openai(tool: &ToolUnion) -> Option<serde_json::Value> {
    match tool {
        ToolUnion::Custom {
            name,
            description,
            input_schema,
            strict,
        } => Some(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description.as_deref().unwrap_or(""),
                "parameters": input_schema,
                "strict": strict.unwrap_or(false)
            }
        })),
        ToolUnion::Other => None,
    }
}

impl ToolChoice {
    fn disable_parallel(&self) -> bool {
        match self {
            ToolChoice::Auto {
                disable_parallel_tool_use,
            }
            | ToolChoice::Any {
                disable_parallel_tool_use,
            }
            | ToolChoice::Tool {
                disable_parallel_tool_use,
                ..
            } => *disable_parallel_tool_use,
            ToolChoice::None => false,
        }
    }

    fn to_openai(&self) -> serde_json::Value {
        match self {
            ToolChoice::Auto { .. } => json!("auto"),
            ToolChoice::Any { .. } => json!("required"),
            ToolChoice::Tool { name, .. } => json!({
                "type": "function",
                "function": { "name": name }
            }),
            ToolChoice::None => json!("none"),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_openai(json: &[u8]) -> serde_json::Value {
        serde_json::from_slice(json).unwrap()
    }

    #[test]
    fn basic_user_message() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["model"], "deepseek-default");
        assert_eq!(openai["max_tokens"], 1024);
        assert_eq!(openai["messages"].as_array().unwrap().len(), 1);
        assert_eq!(openai["messages"][0]["role"], "user");
        assert_eq!(openai["messages"][0]["content"], "Hello");
    }

    #[test]
    fn system_as_string() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "system": "You are a helpful assistant."
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let msgs = openai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn system_as_blocks() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Sys1"}, {"type": "text", "text": "Sys2"}]
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let msgs = openai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Sys1\nSys2");
    }

    #[test]
    fn user_with_text_blocks() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]}
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        // Collapse multiple plaintext blocks before routing.
        assert_eq!(openai["messages"][0]["content"], "Hello\nWorld");
    }

    #[test]
    fn assistant_with_tool_use() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Beijing"}}
                    ]
                }
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let msg = &openai["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "Let me check");
        let tool_calls = msg["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "toolu_01");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
        assert_eq!(
            tool_calls[0]["function"]["arguments"],
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn user_with_tool_result() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "25C"}
                    ]
                }
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let msgs = openai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "toolu_01");
        assert_eq!(msgs[0]["content"], "25C");
    }

    #[test]
    fn stream_and_stop_sequences() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "stream": true,
            "stop_sequences": ["STOP", "HALT"],
            "temperature": 0.7,
            "top_p": 0.9
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["stream"], true);
        let stop = openai["stop"].as_array().unwrap();
        assert_eq!(stop.len(), 2);
        assert_eq!(stop[0], "STOP");
        let temp = openai["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature mismatch: {}", temp);
        let top_p = openai["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.001, "top_p mismatch: {}", top_p);
    }

    #[test]
    fn tools_mapping() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "tools": [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            ],
            "tool_choice": {"type": "auto"}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let tools = openai["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(openai["tool_choice"], "auto");
    }

    #[test]
    fn tool_choice_named_tool() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "tools": [{"type": "custom", "name": "get_weather", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let tc = &openai["tool_choice"];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
    }

    #[test]
    fn tool_choice_disable_parallel() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "tools": [{"type": "custom", "name": "f", "input_schema": {}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["parallel_tool_calls"], false);
    }

    #[test]
    fn thinking_enabled() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["reasoning_effort"], "high");
    }

    #[test]
    fn thinking_disabled() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "thinking": {"type": "disabled"}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["reasoning_effort"], "none");
    }

    #[test]
    fn output_config_json_schema() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["response_format"]["type"], "json_schema");
        assert_eq!(openai["response_format"]["json_schema"]["type"], "object");
    }

    #[test]
    fn unknown_content_blocks_skipped() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "document", "source": {"type": "url", "url": "http://example.com"}}
                    ]
                }
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert_eq!(openai["messages"][0]["content"], "Hello");
    }

    #[test]
    fn top_k_not_mapped() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "top_k": 40
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        assert!(openai.get("top_k").is_none());
    }

    #[test]
    fn server_tools_ignored() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "tools": [
                {"type": "custom", "name": "my_tool", "input_schema": {}},
                {"type": "bash_20250124", "name": "bash"}
            ]
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let tools = openai["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "my_tool");
    }

    #[test]
    fn image_base64_mapped() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"}}
                    ]
                }
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let content = openai["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/jpeg;base64,abc123"
        );
    }

    #[test]
    fn image_url_mapped() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 1024
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let content = openai["messages"][0]["content"].as_array().unwrap();
        assert_eq!(
            content[0]["image_url"]["url"],
            "https://example.com/img.jpg"
        );
    }

    #[test]
    fn web_search_options_mapped() {
        let body = br#"{
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "latest news"}],
            "max_tokens": 1024,
            "web_search_options": {"search_context_size": "high"}
        }"#;

        let openai = parse_openai(&to_openai_request(body).unwrap());
        let opts = &openai["web_search_options"];
        assert_eq!(opts["search_context_size"], "high");
    }

    #[test]
    fn malformed_json_error() {
        let body = b"not-json";
        let err = to_openai_request(body).unwrap_err();
        assert!(matches!(err, AnthropicCompatError::BadRequest(_)));
    }
}
