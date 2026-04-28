//! SSE stream bridge — generic `Stream` into axum `Body`
//!
//! Supports streaming responses for both OpenAI and Anthropic.

use axum::{
    body::Body,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::Sleep;

/// SSE response body wrapper (generic)
pub struct SseBody<S> {
    inner: KeepaliveStream<S>,
    extra_headers: Vec<(String, String)>,
}

impl<S, E> SseBody<S>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    pub fn new(stream: S) -> Self {
        let inner = KeepaliveStream::new(stream);
        Self {
            inner,
            extra_headers: Vec::new(),
        }
    }

    /// Add custom response header
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.extra_headers
            .push((name.to_string(), value.to_string()));
        self
    }
}

impl<S, E> IntoResponse for SseBody<S>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    fn into_response(self) -> Response {
        let body = Body::from_stream(self.inner.map(|result| {
            result.map_err(|e| {
                log::error!(target: "http::response", "SSE stream error: {}", e);
                std::io::Error::other(e.to_string())
            })
        }));

        let mut builder = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive");

        for (name, value) in self.extra_headers {
            builder = builder.header(&name, &value);
        }

        builder.body(body).unwrap().into_response()
    }
}

// Injects SSE keepalive comments on idle to prevent connection timeout.
pin_project_lite::pin_project! {
    struct KeepaliveStream<S> {
        #[pin]
        inner: S,
        #[pin]
        timer: Sleep,
        interval: Duration,
        done: bool,
    }
}

impl<S> KeepaliveStream<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            timer: tokio::time::sleep(Duration::from_millis(1700)),
            interval: Duration::from_millis(1700),
            done: false,
        }
    }
}

impl<S, E> Stream for KeepaliveStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        match this.inner.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                this.timer.reset(tokio::time::Instant::now() + *this.interval);
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                *this.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => {
                match this.timer.as_mut().poll(cx) {
                    Poll::Ready(_) => {
                        this.timer.reset(tokio::time::Instant::now() + *this.interval);
                        Poll::Ready(Some(Ok(Bytes::from_static(b": keepalive\n\n"))))
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}