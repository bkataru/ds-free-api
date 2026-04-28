//! OpenAI-shaped request/response scaffolding (serde-tolerant superset).
//!
//! Shape matches the upstream API; fields we cannot honor are still parsed and dropped later.

#![allow(dead_code)]
// NOTE: This file intentionally mirrors OpenAI's sprawling surface area. The following
// symbols are actually exercised by the adapter layers today:
//
// request path:
//   ChatCompletionRequest.{model,messages,stream,stop,tools,tool_choice,
//   parallel_tool_calls,web_search_options,reasoning_effort}
//   plus nested types: Message, MessageContent, ContentPart, StopSequence, Tool,
//   FunctionDefinition, CustomTool(+CustomToolFormat, GrammarDefinition), ToolChoice(+variants),
//   FunctionCallOption, ResponseFormat, StreamOptions, WebSearchOptions, …
//
// response path:
//   ChatCompletion/{Choice,MessageResponse}, ChatCompletionChunk/{ChunkChoice,Delta}, Usage,
//   ToolCall, FunctionCall, Model, ModelList

use serde::{Deserialize, Serialize};

// ============================================================================
// Request payloads
// ============================================================================

/// `POST /v1/chat/completions` request envelope
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(default)]
    pub stream: bool,

    // Parsed for forward compatibility even when the adapter ignores them today
    #[serde(default)]
    pub audio: Option<AudioRequest>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub function_call: Option<FunctionCallOption>,
    #[serde(default)]
    pub functions: Option<Vec<FunctionDefinition>>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    #[serde(default)]
    pub n: Option<u8>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub prediction: Option<Prediction>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub prompt_cache_retention: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub safety_identifier: Option<String>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub verbosity: Option<String>,
    #[serde(default)]
    pub web_search_options: Option<WebSearchOptions>,

    // Swallow future/unknown fields untouched
    #[serde(flatten)]
    pub _extra: serde_json::Value,
}

/// Legacy `audio` generation hints (top-level field)
#[derive(Debug, Deserialize, Clone)]
pub struct AudioRequest {
    pub format: String,
    pub voice: serde_json::Value,
}

/// Speculative decoding hints (`prediction`)
#[derive(Debug, Deserialize, Clone)]
pub struct Prediction {
    #[serde(rename = "type")]
    pub ty: String,
    pub content: String,
}

/// Hosted web search controls
#[derive(Debug, Deserialize, Clone)]
pub struct WebSearchOptions {
    #[serde(default)]
    pub search_context_size: Option<String>,
    #[serde(default)]
    pub user_location: Option<UserLocation>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub ty: String,
    pub approximate: ApproximateLocation,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApproximateLocation {
    #[serde(default)]
    pub city: Option<String>,
    #[serde(default)]
    pub country: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub timezone: Option<String>,
}

/// One chat turn (OpenAI multimodal capable)
#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub function_call: Option<FunctionCall>,
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    #[serde(default)]
    pub refusal: Option<String>,
}

/// Either a plain string or structured multimodal fragments
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Multimodal part (text, image, audio, file, …)
#[derive(Debug, Deserialize, Clone)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ImageUrlContent>,
    #[serde(default)]
    pub input_audio: Option<InputAudioContent>,
    #[serde(default)]
    pub file: Option<FileContent>,
    #[serde(default)]
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ImageUrlContent {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InputAudioContent {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FileContent {
    #[serde(default)]
    pub file_data: Option<String>,
    #[serde(default)]
    pub file_id: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
}

/// Stop sequence can be a lone string or heterogeneous list
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

/// Tool invocation entry attached to assistant chatter
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub function: Option<FunctionCall>,
    #[serde(default)]
    pub custom: Option<CustomToolCall>,
    #[serde(default)]
    pub index: u32,
}

/// Structured function call payload
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool catalog entry (function or vendor custom)
#[derive(Debug, Deserialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub function: Option<FunctionDefinition>,
    #[serde(default)]
    pub custom: Option<CustomTool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CustomTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub format: Option<CustomToolFormat>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum CustomToolFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "grammar")]
    Grammar { grammar: GrammarDefinition },
}

#[derive(Debug, Deserialize, Clone)]
pub struct GrammarDefinition {
    pub definition: String,
    pub syntax: String,
}

/// Deprecated `function_call` union from legacy clients
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum FunctionCallOption {
    Mode(String),             // "none" | "auto"
    Named(FunctionCallNamed), // { "name": "..." }
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionCallNamed {
    pub name: String,
}

/// `tool_choice` union (mode, named, custom, allowed_tools)
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String), // "none" | "auto" | "required"
    AllowedTools(AllowedToolsChoice),
    Named(NamedToolChoice), // { "type": "function", "function": { "name": "..." } }
    Custom(NamedCustomChoice), // { "type": "custom", "custom": { "name": "..." } }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AllowedToolsChoice {
    #[serde(rename = "type")]
    pub ty: String,
    pub allowed_tools: AllowedTools,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AllowedTools {
    pub mode: String,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: NamedFunction,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamedFunction {
    pub name: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamedCustomChoice {
    #[serde(rename = "type")]
    pub ty: String,
    pub custom: NamedCustom,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamedCustom {
    pub name: String,
}

/// `response_format` guidance block
#[derive(Debug, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

pub(crate) fn default_true() -> bool {
    true
}

/// `stream_options` flags for streaming completions
#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
    #[serde(default = "crate::openai_adapter::types::default_true")]
    pub include_obfuscation: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            include_usage: false,
            include_obfuscation: true,
        }
    }
}

// ============================================================================
// Response payloads
// ============================================================================

/// Non-streaming `/v1/chat/completions` reply
#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u8,
    pub message: MessageResponse,
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// `/v1/chat/completions` streamed chunk wrapper
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u8,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub obfuscation: Option<String>,
}

/// Token accounting block
#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Individual `/v1/models` entry
#[derive(Debug, Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// Preferred max input tokens (when known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u32>,
    /// Preferred max output/completion tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Compatibility alias for `max_input_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Compatibility alias for `max_input_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Compatibility alias for `max_input_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    /// Compatibility alias for `max_output_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Compatibility alias for `max_output_tokens`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
}

/// `/v1/models` list payload
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<Model>,
}

/// Optional audio artifact metadata
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioResponse {
    pub id: String,
    pub data: String,
    pub expires_at: u64,
    pub transcript: String,
}

/// Web search citation annotations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Annotation {
    #[serde(rename = "type")]
    pub ty: String,
    pub url_citation: UrlCitation,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UrlCitation {
    pub end_index: u32,
    pub start_index: u32,
    #[serde(default)]
    pub title: Option<String>,
    pub url: String,
}

/// Logprob payload (OpenAI native shape)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Logprobs {
    #[serde(default)]
    pub content: Option<Vec<TokenLogprob>>,
    #[serde(default)]
    pub refusal: Option<Vec<TokenLogprob>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

/// Custom/non-function tool payload mirror
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CustomToolCall {
    pub name: String,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
}

/// Detailed prompt token accounting
#[derive(Debug, Serialize, Clone)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Detailed completion token accounting
#[derive(Debug, Serialize, Clone)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}
