//! Anthropic response mapping — convert OpenAI chat completions back into Anthropic `Message`s.
//!
//! Facade module: shared structs plus submodule wiring.

mod aggregate;
mod stream;

pub(crate) use aggregate::from_chat_completion_bytes;
pub(crate) use stream::from_chat_completion_stream;

use serde::{Deserialize, Serialize};

/// Canonical Anthropic assistant payload (`message`). Reused by `message_start` when streaming too.
#[derive(Debug, Serialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Serialized `content` union variants.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Prompt/completion token accounting.
#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ============================================================================
// Shared OpenAI completion mirror types (minimal deserialization surface)
// ============================================================================

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiCompletion {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChoice {
    #[serde(default)]
    pub finish_reason: Option<String>,
    pub message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct OpenAiToolCall {
    pub id: String,
    #[serde(default)]
    pub function: Option<OpenAiFunctionCall>,
    #[serde(default)]
    pub custom: Option<OpenAiCustomToolCall>,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct OpenAiCustomToolCall {
    pub name: String,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub(crate) struct OpenAiUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

// ============================================================================
// Shared helpers
// ============================================================================

pub(crate) fn finish_reason_map(reason: &str) -> String {
    match reason {
        "stop" => "end_turn".to_string(),
        "tool_calls" => "tool_use".to_string(),
        _ => reason.to_string(),
    }
}

/// Rewrite OpenAI ids (`chatcmpl-*`) into Anthropic-ish `msg_*` handles.
pub(crate) fn map_id(openai_id: &str) -> String {
    if let Some(hex) = openai_id.strip_prefix("chatcmpl-") {
        format!("msg_{}", hex)
    } else if let Some(suffix) = openai_id.strip_prefix("call_") {
        format!("toolu_{}", suffix)
    } else {
        format!("msg_{}", openai_id)
    }
}
