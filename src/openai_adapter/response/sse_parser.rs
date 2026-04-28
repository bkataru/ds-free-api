//! SSE framing helpers — subdivide upstream `Bytes` payloads into discrete SSE events.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;

use log::debug;

use crate::openai_adapter::OpenAIAdapterError;

/// One parsed SSE stanza (`event` + fused `data` lines).
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

pin_project! {
    #[allow(unused_doc_comments)]
    /// Incremental adaptor that emits fully framed `SseEvent` values while honoring UTF-8 boundaries.
    pub struct SseStream<S> {
        #[pin]
        inner: S,
        text_buf: String,
        raw_buf: Vec<u8>,
    }
}

impl<S> SseStream<S> {
    /// Wrap an arbitrary byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            text_buf: String::new(),
            raw_buf: Vec::new(),
        }
    }
}

impl<S, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: std::fmt::Display,
{
    type Item = Result<SseEvent, OpenAIAdapterError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.raw_buf.extend_from_slice(&bytes);
                    decode_utf8_prefix(this.raw_buf, this.text_buf);
                    if let Some(evt) = try_pop_event(this.text_buf) {
                        return Poll::Ready(Some(Ok(evt)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    debug!(target: "adapter", "SSE upstream error: {}", e);
                    return Poll::Ready(Some(Err(OpenAIAdapterError::Internal(format!(
                        "sse stream error: {}",
                        e
                    )))));
                }
                Poll::Ready(None) => {
                    decode_utf8_prefix(this.raw_buf, this.text_buf);
                    if !this.raw_buf.is_empty() {
                        this.text_buf
                            .push_str(&String::from_utf8_lossy(this.raw_buf));
                        this.raw_buf.clear();
                    }
                    return if let Some(evt) = try_pop_event(this.text_buf) {
                        Poll::Ready(Some(Ok(evt)))
                    } else {
                        Poll::Ready(None)
                    };
                }
                Poll::Pending => {
                    decode_utf8_prefix(this.raw_buf, this.text_buf);
                    if let Some(evt) = try_pop_event(this.text_buf) {
                        return Poll::Ready(Some(Ok(evt)));
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

/// Slide complete UTF-8 prefixes from `raw_buf` onto `text_buf`, leaving dangling bytes queued.
fn decode_utf8_prefix(raw: &mut Vec<u8>, text: &mut String) {
    if raw.is_empty() {
        return;
    }
    match std::str::from_utf8(raw) {
        Ok(s) => {
            text.push_str(s);
            raw.clear();
        }
        Err(e) => {
            let up_to = e.valid_up_to();
            if up_to > 0 {
                unsafe { text.push_str(std::str::from_utf8_unchecked(&raw[..up_to])) };
                raw.drain(..up_to);
            }
        }
    }
}

/// Pop the earliest `\n\n` delimited SSE block from `buf`.
fn try_pop_event(buf: &mut String) -> Option<SseEvent> {
    let pos = buf.find("\n\n")?;
    let tail = buf.split_off(pos);
    let block = std::mem::take(buf);
    *buf = tail;
    if buf.starts_with("\n\n") {
        buf.drain(..2);
    }

    let mut event = None;
    let mut data = String::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            event = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(v.trim_start());
        }
    }
    Some(SseEvent { event, data })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn split_simple_events() {
        let input = Bytes::from("event: ready\ndata: {}\n\nevent: finish\ndata: {}\n\n");
        let stream = SseStream::new(futures::stream::iter(vec![Ok::<_, std::io::Error>(input)]));
        let events: Vec<_> = stream.map(|r| r.unwrap()).collect().await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("ready"));
        assert_eq!(events[0].data, "{}");
        assert_eq!(events[1].event.as_deref(), Some("finish"));
    }

    #[tokio::test]
    async fn split_across_chunks() {
        let parts: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from("event: ready\ndata: {}")),
            Ok(Bytes::from("\n\nevent: finish\ndata: {}\n\n")),
        ];
        let stream = SseStream::new(futures::stream::iter(parts));
        let events: Vec<_> = stream.map(|r| r.unwrap()).collect().await;
        assert_eq!(events.len(), 2);
    }
}
