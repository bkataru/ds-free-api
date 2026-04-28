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

/// SSE response body wrapper (generic)
pub struct SseBody<S> {
    inner: S,
}

impl<S, E> SseBody<S>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    pub fn new(stream: S) -> Self {
        Self { inner: stream }
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

        (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::CONNECTION, "keep-alive"),
            ],
            body,
        )
            .into_response()
    }
}
