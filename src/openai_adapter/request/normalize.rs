//! Request validation and default normalization.
//!
//! Validates required fields and message shape, then projects optional inputs into a small internal struct.

use crate::openai_adapter::types::{ChatCompletionRequest, StopSequence};

pub struct NormalizedParams {
    pub stream: bool,
    pub include_usage: bool,
    pub include_obfuscation: bool,
    pub stop: Vec<String>,
}

/// Normalize `stream`/`stop`/`stream_options` and surface validation errors.
///
/// Rules:
/// - `model` must be non-empty
/// - `messages` must be non-empty
/// - `role=tool` messages require `tool_call_id`
/// - `role=function` messages require `name`
pub fn apply(req: &ChatCompletionRequest) -> Result<NormalizedParams, String> {
    if req.model.trim().is_empty() {
        return Err("missing required field 'model'".into());
    }

    if req.messages.is_empty() {
        return Err("missing required field 'messages'".into());
    }

    for (i, msg) in req.messages.iter().enumerate() {
        match msg.role.as_str() {
            "tool" if msg.tool_call_id.is_none() => {
                return Err(format!(
                    "messages[{i}] with role 'tool' must include 'tool_call_id'",
                ));
            }
            "function" if msg.name.is_none() => {
                return Err(format!(
                    "messages[{i}] with role 'function' must include 'name'",
                ));
            }
            _ => {}
        }
    }

    let stream = req.stream;

    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let include_obfuscation = req
        .stream_options
        .as_ref()
        .map(|o| o.include_obfuscation)
        .unwrap_or(true);

    let stop = match &req.stop {
        Some(StopSequence::Single(s)) => vec![s.clone()],
        Some(StopSequence::Multiple(v)) => v.clone(),
        None => Vec::new(),
    };

    Ok(NormalizedParams {
        stream,
        include_usage,
        include_obfuscation,
        stop,
    })
}
