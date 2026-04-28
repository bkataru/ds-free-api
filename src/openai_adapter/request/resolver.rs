//! Model id resolution — map OpenAI `model` strings to `ds_core` capability flags.
//!
//! Uses the injected registry to resolve aliases to concrete `model_type` values.

use std::collections::HashMap;

use crate::openai_adapter::types::WebSearchOptions;

/// Fully-resolved upstream model configuration.
pub struct ModelResolution {
    /// `model_type` consumed by `ds_core`.
    pub model_type: String,
    pub thinking_enabled: bool,
    pub search_enabled: bool,
}

/// Resolve `model_id` plus optional extensions into capability flags.
///
/// `thinking_enabled` is on when `reasoning_effort` is one of `minimal`/`low`/`medium`/`high`/`xhigh`.
/// When `reasoning_effort` is omitted it defaults to `"high"` (reasoning on by default).
/// `search_enabled` turns on when `web_search_options` is present (off otherwise).
pub fn resolve(
    registry: &HashMap<String, String>,
    model_id: &str,
    reasoning_effort: Option<&str>,
    web_search_options: Option<&WebSearchOptions>,
) -> Result<ModelResolution, String> {
    let key = model_id.to_lowercase();
    let model_type = registry
        .get(&key)
        .cloned()
        .ok_or_else(|| format!("unsupported model: {}", model_id))?;

    let reasoning_effort = reasoning_effort.unwrap_or("high");
    let thinking_enabled = matches!(
        reasoning_effort,
        "minimal" | "low" | "medium" | "high" | "xhigh"
    );

    let search_enabled = web_search_options.is_some();

    Ok(ModelResolution {
        model_type,
        thinking_enabled,
        search_enabled,
    })
}
