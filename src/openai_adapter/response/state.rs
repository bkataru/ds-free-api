//! DeepSeek incremental patch state machine (`p` / `o` / `v`) that emits adapter frames.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use crate::openai_adapter::OpenAIAdapterError;

use super::sse_parser::SseEvent;

const FRAG_THINK: &str = "THINK";
const FRAG_RESPONSE: &str = "RESPONSE";

/// Single normalized delta emitted downstream of the DeepSeek SSE stream.
#[derive(Debug, Clone)]
pub enum DsFrame {
    /// `event: ready` — seeds `delta.role = assistant`.
    Role,
    /// THINK shards — reasoning channel.
    ThinkDelta(String),
    /// RESPONSE shards — outward assistant prose.
    ContentDelta(String),
    /// Mirrors `response/status` churn.
    Status(String),
    /// Numeric `accumulated_token_usage`.
    Usage(u32),
    /// Synthetic finish marker emitted on `finish` events or exhausted streams.
    Finish,
}

#[derive(Debug, Default)]
struct Fragment {
    ty: String,
    content: String,
}

/// Shared patch cursor that tracks streamed DeepSeek blobs.
#[derive(Debug, Default)]
pub struct DsState {
    current_path: Option<String>,
    fragments: Vec<Fragment>,
    status: Option<String>,
    accumulated_token_usage: Option<u32>,
}

impl DsState {
    /// Ingest another SSE stanza — returns zero-or-many frames for the converter tier.
    pub fn apply_event(&mut self, evt: &SseEvent) -> Vec<DsFrame> {
        let mut frames = Vec::new();

        match evt.event.as_deref() {
            Some("ready") => frames.push(DsFrame::Role),
            Some("finish") => frames.push(DsFrame::Finish),
            _ => {}
        }

        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&evt.data) {
            frames.extend(self.apply_patch_value(val));
        }

        frames
    }

    fn apply_patch_value(&mut self, val: serde_json::Value) -> Vec<DsFrame> {
        let mut frames = Vec::new();
        let has_p = val.get("p").is_some();
        let op = val.get("o").and_then(|v| v.as_str());

        if has_p && let Some(p) = val.get("p").and_then(|v| v.as_str()) {
            self.current_path = Some(p.to_string());
        }

        let Some(v) = val.get("v") else {
            return frames;
        };

        if has_p || op.is_some() {
            if let Some(path) = self.current_path.clone() {
                if path == "response" && op == Some("BATCH") {
                    if let Some(arr) = v.as_array() {
                        for item in arr {
                            let sub = self.apply_patch_value(item.clone());
                            frames.extend(sub);
                        }
                    }
                } else {
                    frames.extend(self.apply_path(&path, op, v));
                }
            }
        } else if self.current_path.is_some() {
            let path = self.current_path.clone().unwrap();
            frames.extend(self.apply_path(&path, Some("APPEND"), v));
        } else {
            // Bare `{"v":...}` payloads without trailing `p` act as snapshots.
            if let Some(response) = v.get("response")
                && let Some(arr) = response.get("fragments").and_then(|f| f.as_array())
            {
                self.fragments.clear();
                for frag in arr {
                    if let Some(ty) = frag.get("type").and_then(|t| t.as_str()) {
                        let content = frag
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.fragments.push(Fragment {
                            ty: ty.to_string(),
                            content: content.clone(),
                        });
                        if !content.is_empty() {
                            match ty {
                                FRAG_THINK => frames.push(DsFrame::ThinkDelta(content)),
                                FRAG_RESPONSE => frames.push(DsFrame::ContentDelta(content)),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        frames
    }

    fn apply_path(
        &mut self,
        path: &str,
        op: Option<&str>,
        val: &serde_json::Value,
    ) -> Vec<DsFrame> {
        let mut frames = Vec::new();

        match path {
            "response/status" => {
                if let Some(s) = val.as_str() {
                    self.status = Some(s.to_string());
                    frames.push(DsFrame::Status(s.to_string()));
                }
            }
            "response/accumulated_token_usage" | "accumulated_token_usage" => {
                if let Some(n) = val.as_u64() {
                    let u = n as u32;
                    self.accumulated_token_usage = Some(u);
                    frames.push(DsFrame::Usage(u));
                }
            }
            "response/search_status" | "response/search_results" => {
                // Intentionally ignored for now — annotations hook lives higher in the stack.
            }
            "response/fragments/-1/content" => {
                if let Some(s) = val.as_str()
                    && let Some(frag) = self.fragments.last_mut()
                {
                    match frag.ty.as_str() {
                        FRAG_THINK => {
                            frag.content.push_str(s);
                            frames.push(DsFrame::ThinkDelta(s.to_string()));
                        }
                        FRAG_RESPONSE => {
                            frag.content.push_str(s);
                            frames.push(DsFrame::ContentDelta(s.to_string()));
                        }
                        _ => {
                            // Internal helper fragments (`TOOL_SEARCH`, `TOOL_OPEN`, …) skip user-visible text.
                        }
                    }
                }
            }
            "response/fragments/-1/elapsed_secs" => {
                // Reasoning timings — noop for forwarding.
            }
            "response/fragments" if op == Some("APPEND") => {
                if let Some(arr) = val.as_array() {
                    for item in arr {
                        if let Some(ty) = item.get("type").and_then(|t| t.as_str()) {
                            let content = item
                                .get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            self.fragments.push(Fragment {
                                ty: ty.to_string(),
                                content: content.clone(),
                            });
                            if !content.is_empty() {
                                match ty {
                                    FRAG_THINK => frames.push(DsFrame::ThinkDelta(content)),
                                    FRAG_RESPONSE => frames.push(DsFrame::ContentDelta(content)),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        frames
    }
}

pin_project! {
    #[allow(unused_doc_comments)]
    /// Stream adapter that multiplexes SSE events through `DsState` with tiny fan-out buffering.
    pub struct StateStream<S> {
        #[pin]
        inner: S,
        state: DsState,
        pending: Vec<DsFrame>,
    }
}

impl<S> StateStream<S> {
    /// Construct a streamed patch interpreter.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            state: DsState::default(),
            pending: Vec::new(),
        }
    }
}

impl<S, E> Stream for StateStream<S>
where
    S: Stream<Item = Result<SseEvent, E>>,
    E: Into<OpenAIAdapterError>,
{
    type Item = Result<DsFrame, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if let Some(frame) = this.pending.pop() {
            return Poll::Ready(Some(Ok(frame)));
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(evt))) => {
                    let frames = this.state.apply_event(&evt);
                    if frames.is_empty() {
                        continue;
                    }
                    let mut frames = frames;
                    let first = frames.remove(0);
                    // Remaining frames enqueue in FIFO order (`Vec` stack requires reverse push-order).
                    this.pending.extend(frames.into_iter().rev());
                    return Poll::Ready(Some(Ok(first)));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e.into())));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_content_with_explicit_append() {
        let mut state = DsState::default();
        state.fragments.push(Fragment {
            ty: "RESPONSE".into(),
            content: "".into(),
        });
        let evt = SseEvent {
            event: None,
            data: r#"{"p":"response/fragments/-1/content","o":"APPEND","v":"hello"}"#.into(),
        };
        let frames = state.apply_event(&evt);
        assert!(matches!(&frames[0], DsFrame::ContentDelta(s) if s == "hello"));
    }

    #[test]
    fn append_content_with_bare_v_after_path_set() {
        let mut state = DsState::default();
        state.fragments.push(Fragment {
            ty: "RESPONSE".into(),
            content: "hello".into(),
        });
        state.current_path = Some("response/fragments/-1/content".into());
        let evt = SseEvent {
            event: None,
            data: r#"{"v":" world"}"#.into(),
        };
        let frames = state.apply_event(&evt);
        assert!(matches!(&frames[0], DsFrame::ContentDelta(s) if s == " world"));
    }

    #[test]
    fn snapshot_then_append() {
        let mut state = DsState::default();
        let evt = SseEvent {
            event: None,
            data: r#"{"v":{"response":{"fragments":[{"type":"THINK","content":"hi"}]}}}"#.into(),
        };
        let frames = state.apply_event(&evt);
        assert!(matches!(&frames[0], DsFrame::ThinkDelta(s) if s == "hi"));
    }

    #[test]
    fn ready_and_finish_events() {
        let mut state = DsState::default();
        assert!(matches!(
            state.apply_event(&SseEvent {
                event: Some("ready".into()),
                data: "{}".into(),
            })[0],
            DsFrame::Role
        ));
        assert!(matches!(
            state.apply_event(&SseEvent {
                event: Some("finish".into()),
                data: "{}".into(),
            })[0],
            DsFrame::Finish
        ));
    }

    #[test]
    fn batch_accumulated_token_usage() {
        let mut state = DsState::default();
        let evt = SseEvent {
            event: None,
            data: r#"{"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":41},{"p":"quasi_status","v":"FINISHED"}]}"#.into(),
        };
        let frames = state.apply_event(&evt);
        assert!(matches!(
            &frames[0],
            DsFrame::Usage(u) if *u == 41
        ));
    }
}
