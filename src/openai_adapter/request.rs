//! OpenAI 请求解析 —— 将 OpenAI ChatCompletion 请求降级为 ds_core::ChatRequest
//!
//! 当前限制：
//! - 多轮对话通过 ChatML 格式压缩为单个 prompt 字符串
//! - tool 定义以自然语言注入到最后一条 user 消息中

use log::debug;

use crate::ds_core::ChatRequest;
use crate::openai_adapter::OpenAIAdapterError;
use crate::openai_adapter::types::ChatCompletionRequest;

mod model;
mod normalize;
pub mod prompt;
pub mod tools;

/// 解析并降级后的请求上下文
#[derive(Debug)]
pub struct AdapterRequest {
    pub model: String,
    pub ds_req: ChatRequest,
    pub stream: bool,
    pub include_usage: bool,
    pub include_obfuscation: bool,
    pub stop: Vec<String>,
}

/// 解析 JSON 请求体，执行校验、默认值收敛和能力标志解析
pub fn parse(body: &[u8]) -> Result<AdapterRequest, OpenAIAdapterError> {
    let req: ChatCompletionRequest = serde_json::from_slice(body)
        .map_err(|e| OpenAIAdapterError::BadRequest(format!("bad request: {}", e)))?;

    debug!(target: "adapter", "解析 OpenAI 请求: model={}", req.model);

    let norm = normalize::apply(&req).map_err(OpenAIAdapterError::BadRequest)?;

    let tool_ctx = tools::extract(&req).map_err(OpenAIAdapterError::BadRequest)?;
    let prompt = prompt::build(&req, &tool_ctx);
    let model_res = model::resolve(
        &req.model,
        req.reasoning_effort.as_deref(),
        req.web_search_options.as_ref(),
    )
    .map_err(OpenAIAdapterError::BadRequest)?;

    debug!(target: "adapter", "模型解析结果: thinking={}, search={}", model_res.thinking_enabled, model_res.search_enabled);

    Ok(AdapterRequest {
        model: req.model,
        ds_req: ChatRequest {
            prompt,
            thinking_enabled: model_res.thinking_enabled,
            search_enabled: model_res.search_enabled,
        },
        stream: norm.stream,
        include_usage: norm.include_usage,
        include_obfuscation: norm.include_obfuscation,
        stop: norm.stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_json(val: serde_json::Value) -> Result<AdapterRequest, OpenAIAdapterError> {
        let req = parse(val.to_string().as_bytes())?;
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
                { "role": "system", "content": "你是一个有帮助的助手。" },
                { "role": "user", "content": "你好" }
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
                { "role": "system", "content": "分析图片和音频。" },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "看看这张图片，听听这段音频。" },
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
                { "role": "user", "content": "北京天气怎么样？" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": { "name": "get_weather", "arguments": "{\"city\":\"北京\"}" }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "北京今天晴，25°C。"
                },
                { "role": "user", "content": "谢谢" }
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
                { "role": "system", "content": "你可以使用工具。" },
                { "role": "user", "content": "帮我查一下北京天气。" }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取指定城市的天气",
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
                { "role": "user", "content": "分析一下量子计算" }
            ],
            "reasoning_effort": "high",
            "web_search_options": { "search_context_size": "high" }
        });
        let req = parse_json(body).unwrap();
        assert!(req.ds_req.thinking_enabled);
        assert!(req.ds_req.search_enabled);
    }

    // normalize 错误场景
    #[test]
    fn missing_model() {
        let body = serde_json::json!({
            "messages": [{ "role": "user", "content": "你好" }]
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

    // model 解析错误与能力标志
    #[test]
    fn unsupported_model() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let err = parse_json(body).unwrap_err();
        assert!(matches!(err, OpenAIAdapterError::BadRequest(_)));
        assert!(err.to_string().contains("不支持"));
    }

    #[test]
    fn reasoning_effort_variants() {
        for (effort, expected) in [
            ("minimal", true),
            ("low", true),
            ("medium", true),
            ("high", true),
            ("xhigh", true),
            ("unknown", false),
            ("", false),
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

    // stop 序列与 stream_options 默认值

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

    // tools 校验与注入

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
        assert!(!req.ds_req.prompt.contains("你可以使用以下工具"));
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
        assert!(req.ds_req.prompt.contains("注意：你必须调用一个或多个工具"));
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
        assert!(req.ds_req.prompt.contains("注意：一次只能调用一个工具"));
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
                .contains("注意：你必须调用 'get_weather' 工具")
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
                .contains("注意：你只能从以下允许的工具中选择：get_weather")
        );
        assert!(req.ds_req.prompt.contains("注意：你必须调用一个或多个工具"));
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
                .contains("注意：你必须调用 'my_custom' 自定义工具")
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
        assert!(req.ds_req.prompt.contains("格式: 无约束"));
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

    // tools injection 位置：追加到最后一个 user message

    #[test]
    fn tools_appended_to_last_user_message() {
        let body = serde_json::json!({
            "model": "deepseek-default",
            "messages": [
                { "role": "user", "content": "第一个问题" },
                { "role": "assistant", "content": "回答" },
                { "role": "user", "content": "第二个问题" }
            ],
            "tools": [
                { "type": "function", "function": { "name": "calc" } }
            ]
        });
        let req = parse_json(body).unwrap();
        let prompt = &req.ds_req.prompt;
        // 确保 defs 出现在第二个 user 消息块中，而不是第一个
        let first_user_end = prompt.find("<|im_end|>\n<|im_start|>assistant").unwrap();
        let second_user_start = prompt.find("<|im_start|>user\n第二个问题").unwrap();
        assert!(
            !prompt[..first_user_end].contains("你可以使用以下工具"),
            "工具定义不应出现在第一个 user 消息中"
        );
        assert!(
            prompt[second_user_start..].contains("你可以使用以下工具"),
            "工具定义应出现在最后一个 user 消息中"
        );
    }
}
