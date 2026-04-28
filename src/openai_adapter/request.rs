//! OpenAI request parsing — down-level `ChatCompletion` bodies into `ds_core::ChatRequest`.
//!
//! Constraints:
//! - Multi-round chats collapse into one ChatML-shaped prompt string.
//! - Tool specs land in their own `<|im_start|>reminder` block before `<|im_start|>assistant`.

use log::debug;

use crate::ds_core::ChatRequest;
use crate::openai_adapter::OpenAIAdapterError;
use crate::openai_adapter::types::{
    ChatCompletionRequest, FunctionCallOption, NamedFunction, NamedToolChoice, Tool, ToolChoice,
};

mod normalize;
mod prompt;
mod resolver;
mod tools;

/// Parsed adapter envelope handed to transports
#[derive(Debug)]
pub struct AdapterRequest {
    pub model: String,
    pub ds_req: ChatRequest,
    /// Exposed so the HTTP facade can branch on streaming semantics.
    #[allow(dead_code)]
    pub stream: bool,
    pub include_usage: bool,
    pub include_obfuscation: bool,
    pub stop: Vec<String>,
    pub prompt_tokens: u32,
    pub tools_present: bool,
}

/// Parse JSON payloads, coerce defaults, and resolve upstream capabilities
pub fn parse(
    body: &[u8],
    registry: &std::collections::HashMap<String, String>,
) -> Result<AdapterRequest, OpenAIAdapterError> {
    let mut req: ChatCompletionRequest = serde_json::from_slice(body)
        .map_err(|e| OpenAIAdapterError::BadRequest(format!("bad request: {}", e)))?;

    debug!(target: "adapter", "parsed chat completion payload model={}", req.model);

    // Legacy shim translating `functions` / `function_call` into modern tool fields
    if req.tools.as_ref().map(|t| t.is_empty()).unwrap_or(true)
        && let Some(functions) = req.functions.clone()
        && !functions.is_empty()
    {
        req.tools = Some(
            functions
                .into_iter()
                .map(|f| Tool {
                    ty: "function".to_string(),
                    function: Some(f),
                    custom: None,
                })
                .collect(),
        );
    }
    if req.tool_choice.is_none()
        && let Some(fc) = req.function_call.clone()
    {
        req.tool_choice = Some(match fc {
            FunctionCallOption::Mode(mode) => ToolChoice::Mode(mode),
            FunctionCallOption::Named(named) => ToolChoice::Named(NamedToolChoice {
                ty: "function".to_string(),
                function: NamedFunction { name: named.name },
            }),
        });
    }

    let norm = normalize::apply(&req).map_err(OpenAIAdapterError::BadRequest)?;

    let tool_ctx = tools::extract(&req).map_err(OpenAIAdapterError::BadRequest)?;
    let prompt = prompt::build(&req, &tool_ctx);
    let model_res = resolver::resolve(
        registry,
        &req.model,
        req.reasoning_effort.as_deref(),
        req.web_search_options.as_ref(),
    )
    .map_err(OpenAIAdapterError::BadRequest)?;

    let prompt_tokens = tiktoken_rs::cl100k_base()
        .map(|bpe| bpe.encode_with_special_tokens(&prompt).len() as u32)
        .unwrap_or(0);

    let tools_present = req.tools.as_ref().map_or(false, |t| !t.is_empty());

    debug!(
        target: "adapter",
        "routing flags: thinking_enabled={}, search_enabled={}",
        model_res.thinking_enabled,
        model_res.search_enabled
    );

    Ok(AdapterRequest {
        model: req.model,
        ds_req: ChatRequest {
            prompt,
            thinking_enabled: model_res.thinking_enabled,
            search_enabled: model_res.search_enabled,
            model_type: model_res.model_type,
            files: vec![],
        },
        stream: norm.stream,
        include_usage: norm.include_usage,
        include_obfuscation: norm.include_obfuscation,
        stop: norm.stop,
        prompt_tokens,
        tools_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_registry() -> std::collections::HashMap<String, String> {
        crate::config::DeepSeekConfig::default().model_registry()
    }

    fn parse_json(val: serde_json::Value) -> Result<AdapterRequest, OpenAIAdapterError> {
        let req = parse(val.to_string().as_bytes(), &default_registry())?;
        println!("\n=== PARSED REQUEST ===");
        println!("prompt:\n{}", req.ds_req.prompt);
        println!("adapter: {req:#?}");
        println!("======================\n");
        Ok(req)
    }

    #[test]
    fn basic_chat() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": "Hello" }
            ]
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.prompt.is_empty());
    }

    #[test]
    fn multimodal_user() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "system", "content": "Analyze images and audio." },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Look at this image and listen to this audio." },
                        { "type": "image_url", "image_url": { "url": "data:image/png;base64,abc", "detail": "high" } },
                        { "type": "input_audio", "input_audio": { "data": "base64...", "format": "mp3" } },
                        { "type": "file", "file": { "filename": "report.pdf" } }
                    ]
                }
            ]
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.prompt.is_empty());
    }

    #[test]
    fn tool_conversation() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "What is the weather in Beijing?" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": { "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}" }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "Beijing is sunny today, 25°C."
                },
                { "role": "user", "content": "Thanks" }
            ]
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("get_weather"));
    }

    #[test]
    fn tools_injection() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "system", "content": "You can use tools." },
                { "role": "user", "content": "Check the weather in Beijing." }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                    }
                }
            ],
            "tool_choice": "auto"
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("get_weather"));
    }

    #[test]
    fn reasoning_and_search_flags() {
        let body = serde_json::json!({
            "model": "deepseek-expert",
            "messages": [
                { "role": "user", "content": "Explain quantum computing" }
            ],
            "reasoning_effort": "high",
            "web_search_options": { "search_context_size": "high" }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.thinking_enabled);
        assert!(req.ds_req.search_enabled);
    }

    // Normalization failure coverage
    #[test]
    fn missing_model() {
        let body = serde_json::json!({
            "messages": [{ "role": "user", "content": "Hello" }]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn missing_messages() {
        let body = serde_json::json!({
            "model": "deepseek-default"
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("messages"));
    }

    #[test]
    fn tool_missing_tool_call_id() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "tool", "content": "result" }
            ]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("tool_call_id"));
    }

    #[test]
    fn function_missing_name() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "function", "content": "result" }
            ]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("name"));
    }

    // Registry failures + reasoning/search toggles
    #[test]
    fn unsupported_model() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("unsupported model"));
    }

    #[test]
    fn reasoning_effort_variants() {
        for (effort, expected) in [
            ("minimal", true),
            ("low", true),
            ("medium", true),
            ("high", true),
            ("xhigh", true),
            ("unknown", true),
            ("", true),
        ] {
            let body = serde_json::json!({
                "model": "deepseek-default",
                "messages": [{ "role": "user", "content": "hi" }],
                "reasoning_effort": effort
            });
            let req = parse_json(body).unwrap();
            assert_eq!(
                req.ds_req.thinking_enabled, expected,
                "reasoning_effort={}",
                effort
            );
        }

        // Omitting reasoning_effort still defaults to reasoning-capable tiers
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_json(body).unwrap();
        assert!(
            req.ds_req.thinking_enabled,
            "reasoning_effort absent should default to high"
        );
    }

    #[test]
    fn search_disabled_without_web_search_options() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.search_enabled);
    }

    // Stop payloads + streaming option defaults

    #[test]
    fn stop_single() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "stop": "EOF"
        });
        let req = parse_json(body).unwrap();
        assert_eq!(req.stop, vec!["EOF"]);
    }

    #[test]
    fn stop_multiple() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "stop": ["STOP", "HALT"]
        });
        let req = parse_json(body).unwrap();
        assert_eq!(req.stop, vec!["STOP", "HALT"]);
    }

    #[test]
    fn stream_options_defaults() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = parse_json(body).unwrap();
        assert_eq!(req.stream, false);
        assert_eq!(req.include_usage, false);
        assert_eq!(req.include_obfuscation, true);
    }

    #[test]
    fn stream_options_explicit() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream_options": { "include_usage": true, "include_obfuscation": false }
        });
        let req = parse_json(body).unwrap();
        assert_eq!(req.include_usage, true);
        assert_eq!(req.include_obfuscation, false);
    }

    #[test]
    fn stream_true() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true
        });
        let req = parse_json(body).unwrap();
        assert!(req.stream);
    }

    // Tool-schema validation plus reminder payloads

    #[test]
    fn tool_choice_none_ignores_tools() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "function",
                    "function": { "name": "f", "parameters": {} }
                }
            ],
            "tool_choice": "none"
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.prompt.contains("You can use the following tools"));
    }

    #[test]
    fn tool_choice_required_instruction() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "function",
                    "function": { "name": "f" }
                }
            ],
            "tool_choice": "required"
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("You must call one or more tools."));
    }

    #[test]
    fn parallel_tool_calls_false_instruction() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "f" } }
            ],
            "parallel_tool_calls": false
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("You may only call a single tool invocation."));
    }

    #[test]
    fn tool_choice_named_function() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "get_weather" } }
            ],
            "tool_choice": { "type": "function", "function": { "name": "get_weather" } }
        });
        let req = parse_json(body).unwrap();
        assert!(
            req.ds_req
                .prompt
                .contains("You must call the 'get_weather' tool.")
        );
    }

    #[test]
    fn tool_choice_allowed_tools() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "get_weather" } },
                { "type": "function", "function": { "name": "get_time" } }
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [
                        { "type": "function", "function": { "name": "get_weather" } }
                    ]
                }
            }
        });
        let req = parse_json(body).unwrap();
        assert!(
            req.ds_req
                .prompt
                .contains("You may only choose among these allowed tools: get_weather.")
        );
        assert!(req.ds_req.prompt.contains("You must call one or more tools."));
    }

    #[test]
    fn tool_choice_custom() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "custom",
                    "custom": { "name": "my_custom", "format": { "type": "text" } }
                }
            ],
            "tool_choice": { "type": "custom", "custom": { "name": "my_custom" } }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("- my_custom (custom):"));
        assert!(
            req.ds_req
                .prompt
                .contains("You must call the custom tool 'my_custom'.")
        );
    }

    #[test]
    fn custom_tool_grammar_format() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "grammar_tool",
                        "description": " grammar based tool",
                        "format": {
                            "type": "grammar",
                            "grammar": {
                                "definition": "start: word+",
                                "syntax": "lark"
                            }
                        }
                    }
                }
            ]
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("grammar(syntax: lark)"));
    }

    #[test]
    fn custom_tool_missing_format() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                {
                    "type": "custom",
                    "custom": { "name": "no_format" }
                }
            ]
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("format: unconstrained"));
    }

    #[test]
    fn tool_empty_name() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "" } }
            ]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("name"));
    }

    #[test]
    fn tool_choice_required_without_tools() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": "required"
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
    }

    #[test]
    fn allowed_tools_bad_mode() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "f" } }
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": { "mode": "invalid", "tools": [] }
            }
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
    }

    // Reminder insertion always precedes the final assistant scaffold

    #[test]
    fn tools_as_reminder_before_assistant() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "First question" },
                { "role": "assistant", "content": "Answer" },
                { "role": "user", "content": "Second question" }
            ],
            "tools": [
                { "type": "function", "function": { "name": "calc" } }
            ]
        });
        let req = parse_json(body).unwrap();
        let prompt = &req.ds_req.prompt;
        // Definitions must live in standalone reminder shards before assistants
        assert!(
            !prompt.contains("<|im_start|>user\nSecond question\n\nYou can use the following tools"),
            "tool definitions must not trail immediately after user shards"
        );
        assert!(
            prompt.contains("<|im_start|>reminder") && prompt.contains("You can use the following tools"),
            "tool catalogs must reside within reminder shards"
        );
        // Reminder scaffolding must precede the final `<|im_start|>assistant` sentinel
        let reminder_pos = prompt.find("<|im_start|>reminder").unwrap();
        let assistant_pos = prompt.rfind("<|im_start|>assistant").unwrap();
        assert!(
            reminder_pos < assistant_pos,
            "reminder must appear before assistant staging"
        );
    }

    #[test]
    fn tools_after_tool_role_message() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "Weather in Beijing?" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "get_weather", "arguments": "{\"city\":\"Beijing\"}" }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "Sunny, 25°C"
                }
            ],
            "tools": [
                { "type": "function", "function": { "name": "get_weather", "description": "Get weather" } }
            ],
            "tool_choice": "auto"
        });
        let req = parse_json(body).unwrap();
        let prompt = &req.ds_req.prompt;
        // Trailing tool messages still demand reminder scaffolding before assistants
        assert!(
            prompt.contains("<|im_start|>reminder") && prompt.contains("You can use the following tools"),
            "tool catalogs must reside within reminder shards"
        );
        let reminder_pos = prompt.find("<|im_start|>reminder").unwrap();
        let assistant_pos = prompt.rfind("<|im_start|>assistant").unwrap();
        assert!(reminder_pos < assistant_pos);
    }

    // Compatibility path for deprecated function_call structures

    #[test]
    fn functions_legacy_to_tools() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "Weather in Beijing?" }],
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            ],
            "function_call": "auto"
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("get_weather"));
        assert!(req.ds_req.prompt.contains("You can use the following tools"));
    }

    #[test]
    fn function_call_named_legacy() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "Check weather" }],
            "functions": [
                { "name": "get_weather", "parameters": {} }
            ],
            "function_call": { "name": "get_weather" }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("You must call the 'get_weather' tool."));
    }

    #[test]
    fn tools_priority_over_functions() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [
                { "type": "function", "function": { "name": "tool_a", "parameters": {} } }
            ],
            "functions": [
                { "name": "func_b", "parameters": {} }
            ],
            "tool_choice": "auto",
            "function_call": { "name": "func_b" }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("tool_a"));
        assert!(!req.ds_req.prompt.contains("func_b"));
    }

    #[test]
    fn function_call_none_ignores_functions() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "functions": [
                { "name": "get_weather", "parameters": {} }
            ],
            "function_call": "none"
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.prompt.contains("You can use the following tools"));
    }

    // Folding structured output hints into reminders

    #[test]
    fn response_format_json_object() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "Output JSON" }],
            "response_format": { "type": "json_object" }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("Return a single valid JSON object only"));
    }

    #[test]
    fn response_format_json_schema() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "Structured output" }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": { "type": "object", "properties": { "name": { "type": "string" } } }
                }
            }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.prompt.contains("JSON Schema"));
        assert!(req.ds_req.prompt.contains("person"));
    }

    #[test]
    fn response_format_text_no_injection() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [{ "role": "user", "content": "hi" }],
            "response_format": { "type": "text" }
        });
        let req = parse_json(body).unwrap();
        assert!(!req.ds_req.prompt.contains("Respond using the"));
        assert!(!req.ds_req.prompt.contains("JSON"));
    }
}
