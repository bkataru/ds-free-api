//! Tool-call sniffing layer — carve `<tool_call>` XML blobs out of chunked assistant deltas.
//!
//! Phases:
//! - `Detecting`: maintain rolling UTF-8 text with slack `W` bigger than `<tool_call>` so split packets never evict prefixes.
//! - `CollectingXml`: buffer until balanced `</tool_call>` markers (or forcibly unwind on overflows).
//! - `Done`: after structured deltas emit once, choke trailing assistant spam to emulate OpenAI `tool_calls` chunks.
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use log::debug;

use crate::openai_adapter::OpenAIAdapterError;
use crate::openai_adapter::types::{
    ChatCompletionChunk, ChunkChoice, Delta, FunctionCall, ToolCall,
};

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
pub(crate) const MAX_XML_BUF_LEN: usize = 64 * 1024;

/// Sentinel `<tool_call>` opener
pub(crate) const TOOL_CALL_START: &str = "<tool_call>";
/// Sentinel `</tool_call>` terminator
pub(crate) const TOOL_CALL_END: &str = "</tool_call>";
/// OMP/GPT-style XML tool-call wrapper opener
const OMP_TOOL_CALLS_START: &str = "<tool_calls>";
/// OMP/GPT-style XML tool-call wrapper terminator
const OMP_TOOL_CALLS_END: &str = "</tool_calls>";
/// Byte-length of sentinel
const TAG_LEN: usize = OMP_TOOL_CALLS_START.len();
/// Window width equals marker length plus slack budget so partial UTF-8 never drops tags.
const W: usize = TAG_LEN + 7; // 19

fn next_call_id() -> String {
    let n = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{:016x}", n)
}

/// Clamp `max` to the preceding UTF-8 scalar boundary.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Detect whether `<tool_call>` happens inside stray triple-backtick fenced samples.
fn is_inside_code_fence(xml: &str, tag_pos: usize) -> bool {
    let before = &xml[..tag_pos];
    before.matches("```").count() % 2 == 1
}

fn find_tool_start(s: &str) -> Option<(usize, &'static str)> {
    [TOOL_CALL_START, OMP_TOOL_CALLS_START]
        .into_iter()
        .filter_map(|tag| s.find(tag).map(|pos| (pos, tag)))
        .min_by_key(|(pos, _)| *pos)
}

fn find_tool_end(s: &str) -> Option<(usize, &'static str)> {
    [TOOL_CALL_END, OMP_TOOL_CALLS_END]
        .into_iter()
        .filter_map(|tag| s.find(tag).map(|pos| (pos, tag)))
        .min_by_key(|(pos, _)| *pos)
}

fn decode_xml_entities(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn find_xml_attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=");
    let idx = tag.find(&needle)? + needle.len();
    let quote = tag[idx..].chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &tag[idx + quote.len_utf8()..];
    let end = rest.find(quote)?;
    Some(decode_xml_entities(&rest[..end]))
}

fn parameter_value_to_json(tag: &str, raw: &str) -> serde_json::Value {
    let text = decode_xml_entities(raw).trim().to_string();
    if find_xml_attr(tag, "string").as_deref() == Some("true") {
        return serde_json::Value::String(text);
    }
    serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))
}

fn parse_omp_xml_tool_calls(xml: &str) -> Option<(Vec<ToolCall>, String)> {
    let start = xml.find(OMP_TOOL_CALLS_START)?;
    if is_inside_code_fence(xml, start) {
        return None;
    }
    let after_start = start + OMP_TOOL_CALLS_START.len();
    let inner_end = xml[after_start..].find(OMP_TOOL_CALLS_END)? + after_start;
    let end = inner_end + OMP_TOOL_CALLS_END.len();
    let inner = &xml[after_start..inner_end];

    let mut calls = Vec::new();
    let mut offset = 0;
    while let Some(rel_start) = inner[offset..].find("<invoke") {
        let invoke_start = offset + rel_start;
        let tag_end = inner[invoke_start..].find('>')? + invoke_start;
        let tag = &inner[invoke_start..=tag_end];
        let name = find_xml_attr(tag, "name")?;
        let content_start = tag_end + 1;
        let close_rel = inner[content_start..].find("</invoke>")?;
        let content_end = content_start + close_rel;
        let content = &inner[content_start..content_end];
        offset = content_end + "</invoke>".len();

        let mut args = serde_json::Map::new();
        let mut param_offset = 0;
        while let Some(param_rel) = content[param_offset..].find("<parameter") {
            let param_start = param_offset + param_rel;
            let param_tag_end = content[param_start..].find('>')? + param_start;
            let param_tag = &content[param_start..=param_tag_end];
            let param_name = find_xml_attr(param_tag, "name")?;

            let (value, next_offset) = if param_tag.trim_end_matches('>').trim_end().ends_with('/')
            {
                (
                    find_xml_attr(param_tag, "value")
                        .or_else(|| find_xml_attr(param_tag, "string"))
                        .unwrap_or_default(),
                    param_tag_end + 1,
                )
            } else {
                let value_start = param_tag_end + 1;
                let close = "</parameter>";
                let value_end = content[value_start..].find(close)? + value_start;
                (
                    content[value_start..value_end].to_string(),
                    value_end + close.len(),
                )
            };

            args.insert(param_name, parameter_value_to_json(param_tag, &value));
            param_offset = next_offset;
        }

        calls.push(ToolCall {
            id: next_call_id(),
            ty: "function".to_string(),
            function: Some(FunctionCall {
                name,
                arguments: serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
            }),
            custom: None,
            index: calls.len() as u32,
        });
    }

    if calls.is_empty() {
        return None;
    }

    let remaining = xml[..start].to_string() + &xml[end..];
    Some((calls, remaining))
}

/// Repair malformed JSON escapes for permissive parses.
fn repair_invalid_backslashes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some(&next)
                    if matches!(next, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') =>
                {
                    out.push('\\');
                    out.push(next);
                    chars.next();
                }
                Some(&next) => {
                    out.push('\\');
                    out.push('\\');
                    out.push(next);
                    chars.next();
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Inject quotes around bare object keys when safe.
fn repair_unquoted_keys(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if (chars[i] == '{' || chars[i] == ',') && i + 1 < len {
            out.push(chars[i]);
            i += 1;
            while i < len && chars[i].is_whitespace() {
                out.push(chars[i]);
                i += 1;
            }
            if i < len && (chars[i].is_alphabetic() || chars[i] == '_') {
                let key_start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                if i < len && chars[i] == ':' {
                    out.push('"');
                    out.extend(&chars[key_start..i]);
                    out.push('"');
                } else {
                    out.extend(&chars[key_start..i]);
                    continue;
                }
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Two-pass sanitation (bad escapes, then quoting).
fn repair_json(s: &str) -> Option<String> {
    let step1 = repair_invalid_backslashes(s);
    if serde_json::from_str::<serde_json::Value>(&step1).is_ok() {
        return Some(step1);
    }
    let step2 = repair_unquoted_keys(&step1);
    if serde_json::from_str::<serde_json::Value>(&step2).is_ok() {
        return Some(step2);
    }
    None
}

fn balanced_json_end(s: &str) -> Option<usize> {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            ']' | '}' => {
                if stack.pop() != Some(ch) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(idx + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

fn find_json_payload(s: &str) -> Option<&str> {
    for (start, ch) in s.char_indices() {
        if ch != '[' && ch != '{' {
            continue;
        }
        if let Some(end) = balanced_json_end(&s[start..]) {
            return Some(&s[start..start + end]);
        }
    }
    None
}

fn parse_json_tool_call_items(arr: Vec<serde_json::Value>) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for item in arr {
        let name = item.get("name")?.as_str()?.to_string();
        let arguments = match item.get("arguments") {
            Some(v) => {
                if let Some(s) = v.as_str() {
                    // arguments is a JSON string like "{\"city\": \"Beijing\"}",
                    // parse into object and re-serialize to prevent double-escaping
                    serde_json::from_str::<serde_json::Value>(s)
                        .ok()
                        .and_then(|obj| serde_json::to_string(&obj).ok())
                        .unwrap_or_else(|| s.to_string())
                } else {
                    serde_json::to_string(v).unwrap_or_else(|_| "{}".into())
                }
            }
            None => "{}".into(),
        };
        calls.push(ToolCall {
            id: next_call_id(),
            ty: "function".to_string(),
            function: Some(FunctionCall { name, arguments }),
            custom: None,
            index: calls.len() as u32,
        });
    }

    if calls.is_empty() {
        return None;
    }

    Some(calls)
}

pub(crate) fn parse_json_tool_calls_text(text: &str) -> Option<Vec<ToolCall>> {
    let json_str = find_json_payload(text)?;
    let value = serde_json::from_str::<serde_json::Value>(json_str)
        .ok()
        .or_else(|| {
            let repaired = repair_json(json_str)?;
            serde_json::from_str(&repaired).ok()
        })?;
    let arr = match value {
        serde_json::Value::Array(arr) => arr,
        serde_json::Value::Object(_) => vec![value],
        _ => return None,
    };
    parse_json_tool_call_items(arr)
}

/// Parse bracketed arrays inside `<tool_call>` wrappers into canonical `Vec<ToolCall>`.
///
/// Example payload:
/// `<tool_call>[{"name":"get_weather","arguments":{"city":"Beijing"}}]</tool_call>`
pub fn parse_tool_calls(xml: &str) -> Option<(Vec<ToolCall>, String)> {
    let (start, start_tag) = find_tool_start(xml)?;
    if start_tag == OMP_TOOL_CALLS_START {
        return parse_omp_xml_tool_calls(xml);
    }
    // Markdown examples sometimes embed sentinel text — skip those.
    if is_inside_code_fence(xml, start) {
        return None;
    }
    let after_start = start + TOOL_CALL_START.len();

    // Closing tag keeps hallucinated completions from leaking onward.
    let (end, inner_end) = match xml.find(TOOL_CALL_END) {
        Some(pos) => (pos + TOOL_CALL_END.len(), pos),
        None => (xml.len(), xml.len()),
    };
    let inner = &xml[after_start..inner_end];
    let calls = parse_json_tool_calls_text(inner)?;

    let remaining = xml[..start].to_string() + &xml[end..];
    Some((calls, remaining))
}

fn make_end_chunk(model: &str, delta: Delta, finish_reason: &'static str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: "chatcmpl-end".to_string(),
        object: "chat.completion.chunk",
        created: 0,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage: None,
        service_tier: None,
        system_fingerprint: None,
    }
}

#[derive(Debug)]
enum ToolParseState {
    /// `Detecting` — rolling buffer heuristic.
    Detecting {
        /// Persist tail slack for split `<tool_call>` literals.
        buffer: String,
    },
    /// Drain XML until terminator when marker seen.
    CollectingXml(String),
    /// Parsed — squelch future assistant tokens once tooling wins.
    Done,
}

pin_project! {
    // Streaming adapter rewriting `choices[].delta.content` into structured `delta.tool_calls` surfaces.
    //
    // Applies a guarded UTF-8 window so chunked transcripts never drop split `<tool_call>` tokens, then merges
    // JSON payloads into synthesized OpenAI deltas that align with downstream `tools_present` handling.
    pub struct ToolCallStream<S> {
        #[pin]
        inner: S,
        state: ToolParseState,
        model: String,
        finish_emitted: bool,
        // Pending repair: raw XML text to emit as ToolCallRepairNeeded error on next poll
        repair_pending: Option<String>,
    }
}

impl<S> ToolCallStream<S> {
    /// Instantiate a parser bound against the upstream converter stream.
    pub fn new(inner: S, model: String) -> Self {
        Self {
            inner,
            state: ToolParseState::Detecting {
                buffer: String::new(),
            },
            model,
            finish_emitted: false,
            repair_pending: None,
        }
    }
}

impl<S> Stream for ToolCallStream<S>
where
    S: Stream<Item = Result<ChatCompletionChunk, OpenAIAdapterError>>,
{
    type Item = Result<ChatCompletionChunk, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Emit pending repair error before processing next item.
        if let Some(raw_xml) = this.repair_pending.take() {
            debug!(target: "adapter", "tool_parser emitting repair request");
            return Poll::Ready(Some(Err(OpenAIAdapterError::ToolCallRepairNeeded(raw_xml))));
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(mut chunk))) => {
                    let choice = match chunk.choices.first_mut() {
                        Some(c) => c,
                        None => return Poll::Ready(Some(Ok(chunk))),
                    };

                    if let Some(content) = choice.delta.content.take() {
                        if content.is_empty() {
                            choice.delta.content = Some(content);
                            return Poll::Ready(Some(Ok(chunk)));
                        }

                        match &mut this.state {
                            ToolParseState::Detecting { buffer } => {
                                buffer.push_str(&content);

                                // Detect `<tool_call>` substring inside slack buffer.
                                if let Some((pos, start_tag)) = find_tool_start(buffer) {
                                    debug!(
                                        target: "adapter",
                                        "tool_parser: {} detected (buffer_len={})",
                                        start_tag,
                                        buffer.len()
                                    );
                                    let before = buffer[..pos].to_string();
                                    let rest = std::mem::take(buffer)[pos..].to_string();

                                    // Determine whether `</tool_call>` completes within this chunk.
                                    if let Some((end_pos, end_tag)) = find_tool_end(&rest) {
                                        let end_abs = end_pos + end_tag.len();
                                        let collected = &rest[..end_abs];

                                        if let Some((calls, _)) = parse_tool_calls(collected) {
                                            debug!(
                                                target: "adapter",
                                                "tool_parser: extracted {} invocation(s)",
                                                calls.len()
                                            );
                                            choice.delta.content = if before.is_empty() {
                                                None
                                            } else {
                                                Some(before)
                                            };
                                            choice.delta.tool_calls = Some(calls);
                                            if choice.finish_reason == Some("stop") {
                                                choice.finish_reason = Some("tool_calls");
                                            }
                                            *this.state = ToolParseState::Done;
                                        } else {
                                            debug!(
                                                target: "adapter",
                                                "tool_parser: parse failed — streaming verbatim assistant chars"
                                            );
                                            let collected = collected.to_string();
                                            if before.is_empty() {
                                                return Poll::Ready(Some(Err(
                                                    OpenAIAdapterError::ToolCallRepairNeeded(
                                                        collected,
                                                    ),
                                                )));
                                            }
                                            choice.delta.content = Some(before);
                                            *this.repair_pending = Some(collected);
                                            return Poll::Ready(Some(Ok(chunk)));
                                        }
                                        return Poll::Ready(Some(Ok(chunk)));
                                    }

                                    // Begin XML accumulation when opener exists without terminator.
                                    if before.is_empty() {
                                        *this.state = ToolParseState::CollectingXml(rest);
                                        continue; // Suppress redundant empty deltas while collecting.
                                    }
                                    choice.delta.content = Some(before);
                                    *this.state = ToolParseState::CollectingXml(rest);
                                    return Poll::Ready(Some(Ok(chunk)));
                                } else {
                                    // Safe prefix release when marker absent.
                                    let safe =
                                        floor_char_boundary(buffer, buffer.len().saturating_sub(W));
                                    if safe > 0 {
                                        choice.delta.content = Some(buffer[..safe].to_string());
                                        buffer.drain(..safe);
                                        return Poll::Ready(Some(Ok(chunk)));
                                    }
                                    // Hold buffered text while marker may straddle chunks.
                                    continue;
                                }
                            }

                            ToolParseState::CollectingXml(buf) => {
                                buf.push_str(&content);
                                if buf.len() > MAX_XML_BUF_LEN {
                                    debug!(
                                        target: "adapter",
                                        "tool_parser: buffered XML overrun — flushing literal assistant text fallback"
                                    );
                                    let flushed = std::mem::take(buf);
                                    *this.state = ToolParseState::Detecting {
                                        buffer: String::new(),
                                    };
                                    choice.delta.content = Some(flushed);
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                if let Some((end_pos, end_tag)) = find_tool_end(buf) {
                                    let end_abs = end_pos + end_tag.len();
                                    let collected = buf[..end_abs].to_string();
                                    let _tail = buf.split_off(end_abs);

                                    if let Some((calls, _)) = parse_tool_calls(&collected) {
                                        debug!(
                                            target: "adapter",
                                            "tool_parser: extracted {} invocation(s)",
                                            calls.len()
                                        );
                                        // Drop hallucinated conversational tail after terminator.
                                        choice.delta.content = None;
                                        choice.delta.tool_calls = Some(calls);
                                        if choice.finish_reason == Some("stop") {
                                            choice.finish_reason = Some("tool_calls");
                                        }
                                        *this.state = ToolParseState::Done;
                                    } else {
                                        debug!(
                                            target: "adapter",
                                            "tool_parser: parse failed — streaming verbatim assistant chars"
                                        );
                                        return Poll::Ready(Some(Err(
                                            OpenAIAdapterError::ToolCallRepairNeeded(collected),
                                        )));
                                    }
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                // Partial XML fragment — fetch more deltas before coercion.
                                continue;
                            }

                            ToolParseState::Done => {
                                // Parsed — suppress continued assistant narrative.
                                if !*this.finish_emitted {
                                    *this.finish_emitted = true;
                                    let chunk =
                                        make_end_chunk(this.model, Delta::default(), "tool_calls");
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                return Poll::Ready(None);
                            }
                        }
                    } else {
                        // Non-text deltas (signals, refusal, reasoning, …).
                        match &mut this.state {
                            ToolParseState::Detecting { buffer } => {
                                if choice.finish_reason.is_some() {
                                    // Finish sentinel — spill buffered conversational prefix first.
                                    if !buffer.is_empty() {
                                        choice.delta.content = Some(std::mem::take(buffer));
                                    }
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                // Lightweight metadata deltas bypass rewriting.
                                return Poll::Ready(Some(Ok(chunk)));
                            }

                            ToolParseState::CollectingXml(buf) => {
                                if choice.finish_reason.is_some() {
                                    // Finish reached — coerce buffered XML.
                                    let flushed = std::mem::take(buf);
                                    if let Some((calls, _)) = parse_tool_calls(&flushed) {
                                        debug!(
                                            target: "adapter",
                                            "tool_parser: terminal flush extracted {} invocation(s)",
                                            calls.len()
                                        );
                                        choice.delta.tool_calls = Some(calls);
                                        if choice.finish_reason == Some("stop") {
                                            choice.finish_reason = Some("tool_calls");
                                        }
                                    } else {
                                        debug!(
                                            target: "adapter",
                                            "tool_parser: terminal flush unable to coerce tool JSON — flushing buffered text fallback"
                                        );
                                        *this.state = ToolParseState::Done;
                                        return Poll::Ready(Some(Err(
                                            OpenAIAdapterError::ToolCallRepairNeeded(flushed),
                                        )));
                                    }
                                    *this.state = ToolParseState::Done;
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                // Non-terminal structured deltas propagate untouched.
                                return Poll::Ready(Some(Ok(chunk)));
                            }

                            ToolParseState::Done => {
                                // Completed tool handshake — choke remainder of downstream stream.
                                if !*this.finish_emitted {
                                    *this.finish_emitted = true;
                                    let chunk =
                                        make_end_chunk(this.model, Delta::default(), "tool_calls");
                                    return Poll::Ready(Some(Ok(chunk)));
                                }
                                return Poll::Ready(None);
                            }
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    // Source ended — spill buffers before closing adapters.
                    match std::mem::replace(this.state, ToolParseState::Done) {
                        ToolParseState::Detecting { buffer } => {
                            if !buffer.is_empty() {
                                let chunk = make_end_chunk(
                                    this.model,
                                    Delta {
                                        content: Some(buffer),
                                        ..Default::default()
                                    },
                                    "stop",
                                );
                                return Poll::Ready(Some(Ok(chunk)));
                            }
                            return Poll::Ready(None);
                        }
                        ToolParseState::CollectingXml(buf) => {
                            // Source ended mid-XML — best-effort parse.
                            if let Some((calls, _)) = parse_tool_calls(&buf) {
                                debug!(
                                    target: "adapter",
                                    "tool_parser: terminal flush extracted {} invocation(s)",
                                    calls.len()
                                );
                                let chunk = make_end_chunk(
                                    this.model,
                                    Delta {
                                        tool_calls: Some(calls),
                                        ..Default::default()
                                    },
                                    "tool_calls",
                                );
                                return Poll::Ready(Some(Ok(chunk)));
                            } else {
                                debug!(
                                    target: "adapter",
                                    "tool_parser: terminal flush unable to coerce tool JSON — flushing buffered text fallback"
                                );
                                return Poll::Ready(Some(Err(
                                    OpenAIAdapterError::ToolCallRepairNeeded(buf),
                                )));
                            }
                        }
                        ToolParseState::Done => return Poll::Ready(None),
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_tool_calls() {
        let xml =
            r#"<tool_call>[{"name": "get_weather", "arguments": {"city": "Beijing"}}]</tool_call>"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(
            calls[0].function.as_ref().unwrap().arguments,
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn parse_json_with_surrounding_text() {
        // Permit commentary before/after array bodies
        let xml = r#"<tool_call>
The following is a tool call:
[{"name": "f", "arguments": {}}]
</tool_call>"#;
        let (calls, _remaining) = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "f");
    }

    #[test]
    fn parse_json_multiple_tools() {
        let xml = r#"<tool_call>[{"name": "get_weather", "arguments": {}}, {"name": "get_time", "arguments": {"tz": "bj"}}]</tool_call>"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(calls[1].index, 1);
        assert_eq!(calls[1].function.as_ref().unwrap().name, "get_time");
    }

    #[test]
    fn parse_json_with_trailing_text() {
        let xml =
            r#"<tool_call>[{"name": "get_weather", "arguments": {}}]</tool_call> trailing text"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert_eq!(remaining, " trailing text");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
    }

    #[test]
    fn parse_tool_calls_single_object() {
        let xml =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "New York"}}</tool_call>"#;
        let (calls, _) = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "get_weather");
    }

    #[test]
    fn parse_tool_calls_single_object_with_surrounding_text() {
        let xml = r#"<tool_call>here is the tool: {"name": "f", "arguments": {}}</tool_call>"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(remaining, "");
    }

    #[test]
    fn parse_omp_xml_tool_calls() {
        let xml = r#"<tool_calls>
<invoke name="bash">
<parameter name="command" string="true">git tag v0.2.2 &amp;&amp; git push bkataru v0.2.2</parameter>
<parameter name="cwd" string="true">/home/int-wizard/ds-free-api</parameter>
</invoke>
</tool_calls>"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(calls.len(), 1);
        let function = calls[0].function.as_ref().unwrap();
        assert_eq!(function.name, "bash");
        assert_eq!(
            function.arguments,
            r#"{"command":"git tag v0.2.2 && git push bkataru v0.2.2","cwd":"/home/int-wizard/ds-free-api"}"#
        );
    }

    #[test]
    fn parse_omp_xml_multiple_invokes_with_trailing_text() {
        let xml = r#"prefix <tool_calls><invoke name="read"><parameter name="path" string="true">/tmp/a</parameter></invoke><invoke name="bash"><parameter name="command" string="true">date</parameter></invoke></tool_calls> suffix"#;
        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert_eq!(remaining, "prefix  suffix");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "read");
        assert_eq!(calls[1].function.as_ref().unwrap().name, "bash");
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[1].index, 1);
    }

    #[test]
    fn parse_repair_json_with_literal_omp_markers_in_argument() {
        let text = r###"[
  {
    "name": "edit",
    "arguments": {
      "path": "/home/int-wizard/ds-free-api/CHANGELOG.md",
      "edits": [
        {
          "loc": "7jn",
          "pre": [
            "",
            "## [0.2.3] - 2026-04-28",
            "- Recognizes OMP-style `<tool_calls><invoke name=\"bash\"></invoke></tool_calls>` text."
          ]
        }
      ]
    }
  }
]"###;

        let calls = parse_json_tool_calls_text(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "edit");
        assert!(
            calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .contains("<tool_calls>")
        );
    }

    #[test]
    fn parse_tool_call_json_takes_precedence_over_literal_omp_markers() {
        let xml = r#"<tool_call>[
  {
    "name": "edit",
    "arguments": {
      "body": "Document literal <tool_calls><invoke name=\"bash\"></invoke></tool_calls> markers"
    }
  }
]</tool_call>"#;

        let (calls, remaining) = parse_tool_calls(xml).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.as_ref().unwrap().name, "edit");
    }
}
