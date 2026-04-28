//! Tool parsing — validate `tools` / `tool_choice` and build injected reminder text.
//!
//! `ds_core` does not expose native function calling; this adapter downgrades definitions to prose
//! appended before the assistant block so models still emit structured `<tool_calls>` output.

use crate::openai_adapter::types::{
    AllowedTools, AllowedToolsChoice, ChatCompletionRequest, CustomTool, CustomToolFormat,
    FunctionDefinition, Tool, ToolChoice,
};

/// Tool-related context lifted out of an OpenAI request.
pub struct ToolContext {
    /// Rendered tool catalog for the reminder block.
    pub defs_text: Option<String>,
    /// Extra behavioral lines derived from `tool_choice` / `parallel_tool_calls`.
    pub instruction_text: Option<String>,
}

fn has_tools(req: &ChatCompletionRequest) -> bool {
    req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false)
}

/// Extract tooling metadata or fail validation.
///
/// When `tool_choice` is `"none"` this returns empty `ToolContext` (no injection text).
pub fn extract(req: &ChatCompletionRequest) -> Result<ToolContext, String> {
    let default_choice = if has_tools(req) {
        ToolChoice::Mode("auto".to_string())
    } else {
        ToolChoice::Mode("none".to_string())
    };
    let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);

    validate_tool_choice(tool_choice, req.tools.as_deref())?;

    if matches!(tool_choice, ToolChoice::Mode(m) if m == "none") {
        return Ok(ToolContext {
            defs_text: None,
            instruction_text: None,
        });
    }

    let mut instruction_lines = Vec::new();

    match tool_choice {
        ToolChoice::Mode(mode) => {
            if mode == "required" {
                instruction_lines
                    .push("You must call one or more tools.".to_string());
            }
        }
        ToolChoice::AllowedTools(AllowedToolsChoice { allowed_tools, .. }) => {
            build_allowed_tools_instruction(allowed_tools, &mut instruction_lines)?;
        }
        ToolChoice::Named(named) => {
            instruction_lines.push(format!(
                "You must call the '{}' tool.",
                named.function.name
            ));
        }
        ToolChoice::Custom(custom) => {
            instruction_lines.push(format!(
                "You must call the custom tool '{}'.",
                custom.custom.name
            ));
        }
    }

    if req.parallel_tool_calls == Some(false) {
        instruction_lines.push("You may only call a single tool invocation.".to_string());
    }

    instruction_lines.push(
        "When tools are required, respond using the format below with no extra prose:"
            .to_string(),
    );
    instruction_lines.push("<tool_calls>".to_string());
    instruction_lines.push(
        "[{\"name\": \"<tool_name>\", \"arguments\": {<args-json>}}]".to_string(),
    );
    instruction_lines.push("</tool_calls>".to_string());
    instruction_lines.push("Multiple tool calls can appear as multiple objects in the same array."
        .to_string());
    instruction_lines.push("Example:".to_string());
    instruction_lines.push("<tool_calls>".to_string());
    instruction_lines.push(
        "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}, {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\"}}]"
            .to_string(),
    );
    instruction_lines.push("</tool_calls>".to_string());

    let defs_text = if has_tools(req) {
        let mut lines = vec!["You can use the following tools:".to_string()];
        for (i, tool) in req.tools.as_ref().unwrap().iter().enumerate() {
            lines.push(format_tool(tool, i)?);
        }
        Some(lines.join("\n"))
    } else {
        None
    };

    let instruction_text = if instruction_lines.is_empty() {
        None
    } else {
        Some(instruction_lines.join("\n"))
    };

    Ok(ToolContext {
        defs_text,
        instruction_text,
    })
}

fn validate_tool_choice(tc: &ToolChoice, tools: Option<&[Tool]>) -> Result<(), String> {
    match tc {
        ToolChoice::Mode(mode) => {
            if !matches!(mode.as_str(), "none" | "auto" | "required") {
                return Err(format!("tool_choice has invalid mode: {}", mode));
            }
            if matches!(mode.as_str(), "auto" | "required")
                && tools.map(|t| t.is_empty()).unwrap_or(true)
            {
                return Err("tool_choice 'auto' or 'required' requires a non-empty tools list".into());
            }
            Ok(())
        }
        ToolChoice::Named(_) | ToolChoice::Custom(_) => {
            if tools.is_none() {
                return Err("tool_choice names a specific tool but `tools` is missing".into());
            }
            Ok(())
        }
        ToolChoice::AllowedTools(AllowedToolsChoice { allowed_tools, .. }) => {
            if tools.is_none() {
                return Err("tool_choice uses allowed_tools but `tools` is missing".into());
            }
            if !matches!(allowed_tools.mode.as_str(), "auto" | "required") {
                return Err(format!(
                    "allowed_tools.mode must be 'auto' or 'required'; got {}",
                    allowed_tools.mode
                ));
            }
            Ok(())
        }
    }
}

fn build_allowed_tools_instruction(
    allowed_tools: &AllowedTools,
    lines: &mut Vec<String>,
) -> Result<(), String> {
    if let Some(tool_list) = &allowed_tools.tools {
        let names: Vec<String> = tool_list
            .iter()
            .filter_map(|v| v.get("function").and_then(|f| f.get("name")))
            .filter_map(|n| n.as_str().map(|s| s.to_string()))
            .collect();
        if !names.is_empty() {
            lines.push(format!(
                "You may only choose among these allowed tools: {}.",
                names.join(", ")
            ));
        }
    }

    if allowed_tools.mode == "required" {
        lines.push("You must call one or more tools.".to_string());
    }
    Ok(())
}

fn format_tool(tool: &Tool, idx: usize) -> Result<String, String> {
    match tool.ty.as_str() {
        "function" => {
            let func = tool.function.as_ref().ok_or_else(|| {
                format!(
                    "tools[{idx}] with type 'function' must include a function definition",
                )
            })?;
            format_function(func)
        }
        "custom" => {
            let custom = tool.custom.as_ref().ok_or_else(|| {
                format!("tools[{idx}] with type 'custom' must include a custom definition")
            })?;
            Ok(format_custom(custom))
        }
        _ => Err(format!("tools[{idx}] has unsupported type: {}", tool.ty)),
    }
}

fn format_function(func: &FunctionDefinition) -> Result<String, String> {
    if func.name.trim().is_empty() {
        return Err("function tool is missing required field 'name'".into());
    }
    let params = serde_json::to_string(&func.parameters).unwrap_or_else(|_| "{}".into());
    Ok(format!(
        "- {} (function): {}\n  parameters (JSON schema): {}",
        func.name,
        func.description.as_deref().unwrap_or(""),
        params
    ))
}

fn format_custom(custom: &CustomTool) -> String {
    let format_desc: &str = match &custom.format {
        Some(CustomToolFormat::Text) => "text",
        Some(CustomToolFormat::Grammar { grammar }) => {
            return format!(
                "- {} (custom): {}\n  format: grammar(syntax: {})",
                custom.name,
                custom.description.as_deref().unwrap_or(""),
                grammar.syntax
            );
        }
        None => "unconstrained",
    };
    format!(
        "- {} (custom): {}\n  format: {}",
        custom.name,
        custom.description.as_deref().unwrap_or(""),
        format_desc
    )
}
