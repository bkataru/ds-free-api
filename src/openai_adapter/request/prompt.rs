//! Prompt assembly — map OpenAI `messages` into a ChatML-shaped string.
//!
//! When tools or behavioral constraints are present, they are appended as a standalone
//! `<|im_start|>reminder` block inserted immediately before `<|im_start|>assistant` so tool context stays next to the generation boundary.

use super::tools::ToolContext;
use crate::openai_adapter::types::{ChatCompletionRequest, ContentPart, Message, MessageContent};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Build the ChatML-shaped prompt string consumed by `ds_core`.
pub fn build(req: &ChatCompletionRequest, tool_ctx: &ToolContext) -> String {
    let mut parts: Vec<String> = req.messages.iter().map(format_message).collect();

    let mut extra_blocks: Vec<String> = Vec::new();
    if let Some(text) = tool_ctx.defs_text.as_deref() {
        extra_blocks.push(text.to_string());
    }
    if let Some(text) = tool_ctx.instruction_text.as_deref() {
        extra_blocks.push(text.to_string());
    }

    // Down-level `response_format`: fold constraints into the reminder block.
    if let Some(rf) = &req.response_format {
        match rf.ty.as_str() {
            "json_object" => {
                extra_blocks.push(
                    "Return a single valid JSON object only — no markdown fences or extra prose."
                        .to_string(),
                );
            }
            "json_schema" => {
                let schema_text = rf
                    .json_schema
                    .as_ref()
                    .map(|s| serde_json::to_string(s).unwrap_or_default())
                    .unwrap_or_default();
                if schema_text.is_empty() {
                    extra_blocks.push(
                        "Strictly follow JSON Schema output rules — do not wrap in markdown fences."
                            .to_string(),
                    );
                } else {
                    extra_blocks.push(format!(
                        "Strictly follow the JSON Schema below. Do not emit fields outside the schema and do not wrap in markdown fences.\nSchema: {}",
                        schema_text
                    ));
                }
            }
            "text" => {} // default — nothing to inject
            _ => {
                extra_blocks.push(format!("Respond using the '{}' format.", rf.ty));
            }
        }
    }

    if !extra_blocks.is_empty() {
        let extra = extra_blocks.join("\n\n");
        parts.push(format!("{IM_START}reminder\n{extra}\n{IM_END}"));
    }

    parts.push(format!("{IM_START}assistant"));
    parts.join("\n")
}

fn format_message(msg: &Message) -> String {
    let body = match msg.role.as_str() {
        "assistant" => format_assistant(msg),
        "tool" => format_tool(msg),
        "function" => format_function(msg),
        _ => format_generic(msg),
    };
    format!("{IM_START}{}\n{}\n{IM_END}", msg.role, body)
}

fn format_generic(msg: &Message) -> String {
    let mut parts = Vec::new();
    if let Some(name) = &msg.name {
        parts.push(format!("(name: {name})"));
    }
    if let Some(content) = &msg.content {
        parts.push(format_content(content));
    }
    parts.join("\n")
}

fn format_assistant(msg: &Message) -> String {
    let mut parts = Vec::new();
    if let Some(content) = &msg.content {
        parts.push(format_content(content));
    }
    if let Some(tool_calls) = &msg.tool_calls {
        let items: Vec<String> = tool_calls
            .iter()
            .filter_map(|tc| {
                tc.function.as_ref().map(|func| {
                    let args = serde_json::from_str::<serde_json::Value>(&func.arguments)
                        .unwrap_or(serde_json::Value::Null);
                    format!(
                        "{{\"name\": {}, \"arguments\": {}}}",
                        serde_json::to_string(&func.name).unwrap_or_else(|_| "\"\"".into()),
                        serde_json::to_string(&args).unwrap_or_else(|_| "null".into()),
                    )
                })
            })
            .collect();
        parts.push(format!(
            "<tool_calls>\n[{}]\n</tool_calls>",
            items.join(", ")
        ));
    }
    if let Some(fc) = &msg.function_call {
        let args = serde_json::from_str::<serde_json::Value>(&fc.arguments)
            .unwrap_or(serde_json::Value::Null);
        let item = format!(
            "{{\"name\": {}, \"arguments\": {}}}",
            serde_json::to_string(&fc.name).unwrap_or_else(|_| "\"\"".into()),
            serde_json::to_string(&args).unwrap_or_else(|_| "null".into()),
        );
        parts.push(format!("<tool_calls>\n[{item}]\n</tool_calls>"));
    }
    if let Some(refusal) = &msg.refusal {
        parts.push(format!("(refusal: {refusal})"));
    }
    parts.join("\n")
}

fn format_tool(msg: &Message) -> String {
    let mut parts = Vec::new();
    if let Some(id) = &msg.tool_call_id {
        parts.push(format!("(tool_call_id: {id})"));
    }
    if let Some(content) = &msg.content {
        parts.push(format_content(content));
    }
    parts.join("\n")
}

fn format_function(msg: &Message) -> String {
    let mut parts = Vec::new();
    if let Some(name) = &msg.name {
        parts.push(format!("(name: {name})"));
    }
    if let Some(content) = &msg.content {
        parts.push(format_content(content));
    }
    parts.join("\n")
}

fn format_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => {
            parts.iter().map(format_part).collect::<Vec<_>>().join("\n")
        }
    }
}

fn format_part(part: &ContentPart) -> String {
    match part.ty.as_str() {
        "text" => part.text.clone().unwrap_or_default(),
        "refusal" => part.refusal.clone().unwrap_or_default(),
        "image_url" => {
            let detail = part
                .image_url
                .as_ref()
                .and_then(|i| i.detail.as_deref())
                .unwrap_or("auto");
            format!("[image: detail={detail}]")
        }
        "input_audio" => {
            let fmt = part
                .input_audio
                .as_ref()
                .map(|a| a.format.as_str())
                .unwrap_or("unknown");
            format!("[audio: format={fmt}]")
        }
        "file" => {
            let filename = part
                .file
                .as_ref()
                .and_then(|f| f.filename.as_deref())
                .unwrap_or("unknown");
            format!("[file: filename={filename}]")
        }
        _ => format!("[unsupported content type: {}]", part.ty),
    }
}
