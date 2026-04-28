//! Chat orchestration — wraps `edit_message` and returns an SSE byte stream.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;

use crate::ds_core::CoreError;
use crate::ds_core::accounts::{AccountGuard, AccountPool};
use crate::ds_core::client::{DsClient, EditMessagePayload, StopStreamPayload};
use crate::ds_core::pow::PowSolver;

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub prompt: String,
    pub thinking_enabled: bool,
    pub search_enabled: bool,
    pub model_type: String,
}

pin_project! {
    pub struct GuardedStream<S> {
        #[pin]
        stream: S,
        _guard: AccountGuard,
        client: DsClient,
        token: String,
        session_id: String,
        message_id: i64,
        finished: bool,
    }

    impl<S> PinnedDrop for GuardedStream<S> {
        fn drop(this: Pin<&mut Self>) {
            let this = this.project();
            if *this.finished {
                return;
            }
            let client = this.client.clone();
            let token = this.token.clone();
            let session_id = this.session_id.clone();
            let message_id = *this.message_id;
            tokio::spawn(async move {
                let payload = StopStreamPayload {
                    chat_session_id: session_id,
                    message_id,
                };
                if let Err(e) = client.stop_stream(&token, &payload).await {
                    log::warn!(target: "ds_core::accounts", "stop_stream failed: {}", e);
                }
            });
        }
    }
}

impl<S> GuardedStream<S> {
    pub fn new(
        stream: S,
        guard: AccountGuard,
        client: DsClient,
        token: String,
        session_id: String,
        message_id: i64,
    ) -> Self {
        Self {
            stream,
            _guard: guard,
            client,
            token,
            session_id,
            message_id,
            finished: false,
        }
    }
}

impl<S, E> Stream for GuardedStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: std::fmt::Display,
{
    type Item = Result<Bytes, CoreError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match this.stream.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(CoreError::Stream(e.to_string())))),
            Poll::Ready(None) => {
                *this.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

pub struct Completions {
    client: DsClient,
    solver: PowSolver,
    pool: AccountPool,
}

impl Completions {
    pub fn new(client: DsClient, solver: PowSolver, pool: AccountPool) -> Self {
        Self {
            client,
            solver,
            pool,
        }
    }

    pub async fn v0_chat(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes, CoreError>> + Send>>, CoreError> {
        let guard = self.pool.get_account(&req.model_type).ok_or_else(|| {
            log::info!(
                target: "ds_core::accounts",
                "no available account: model_type={}", req.model_type
            );
            CoreError::Overloaded
        })?;

        let account = guard.account();
        let token_preview = if account.token().len() > 8 {
            format!("{}...", &account.token()[..8])
        } else {
            account.token().to_string()
        };
        log::debug!(
            target: "ds_core::accounts",
            "account assigned: model_type={}, token={}", req.model_type, token_preview
        );
        let token = account.token().to_string();
        let session_id = account
            .session_id(&req.model_type)
            .expect("session for this model_type exists after pool init");

        let edit_message_id = account.next_message_id(&req.model_type);
        let pow_header = self.compute_pow(&token).await?;

        let payload = EditMessagePayload {
            chat_session_id: session_id.clone(),
            message_id: edit_message_id,
            prompt: req.prompt,
            search_enabled: req.search_enabled,
            thinking_enabled: req.thinking_enabled,
            model_type: req.model_type.clone(),
        };

        let mut raw_stream = self
            .client
            .edit_message(&token, &pow_header, &payload)
            .await?;

        // Collect bytes until the first two SSE events (ready + update_session / hint)
        let mut buf = Vec::new();
        let mut text_buf = String::new();
        let (ready_block, second_block) = loop {
            let chunk = raw_stream
                .next()
                .await
                .ok_or_else(|| CoreError::Stream("empty SSE stream".into()))?
                .map_err(|e| CoreError::Stream(e.to_string()))?;
            buf.extend_from_slice(&chunk);
            text_buf.push_str(&String::from_utf8_lossy(&chunk));

            if let Some((first, second)) = split_two_events(&text_buf) {
                break (first.to_owned(), second.to_owned());
            }
        };

        let (next_edit_id, stop_id) = parse_ready_message_ids(ready_block.as_bytes());

        // Check if the second event is a rate-limit hint
        if is_rate_limit_hint(&second_block) {
            log::warn!(
                target: "ds_core::accounts",
                "edit_message rate-limited: edit_msg={}", edit_message_id
            );
            return Err(CoreError::Overloaded);
        }

        log::debug!(
            target: "ds_core::accounts",
            "SSE ready: edit_msg={} -> req={} resp={}", edit_message_id, next_edit_id, stop_id
        );

        // Update the session's next edit_message_id
        account.set_next_message_id(&req.model_type, next_edit_id);

        // Reconstruct the stream with already-consumed chunks
        let stream =
            futures::stream::once(futures::future::ready(Ok(Bytes::from(buf)))).chain(raw_stream);

        Ok(Box::pin(GuardedStream::new(
            Box::pin(stream),
            guard,
            self.client.clone(),
            token,
            session_id,
            stop_id,
        )))
    }

    async fn compute_pow(&self, token: &str) -> Result<String, CoreError> {
        let challenge_data = self.client.create_pow_challenge(token).await?;
        let result = self.solver.solve(&challenge_data)?;
        Ok(result.to_header())
    }

    pub fn account_statuses(&self) -> Vec<crate::ds_core::accounts::AccountStatus> {
        self.pool.account_statuses()
    }

    /// Graceful shutdown: delete sessions for every pooled account.
    pub async fn shutdown(&self) {
        self.pool.shutdown(&self.client).await;
    }
}

/// Extract the first two complete SSE event blocks from a string buffer
fn split_two_events(buf: &str) -> Option<(&str, &str)> {
    let parts: Vec<&str> = buf.splitn(3, "\n\n").collect();
    if parts.len() < 3 {
        return None;
    }
    Some((parts[0], parts[1]))
}

/// Check if the second SSE event is a rate-limit error
fn is_rate_limit_hint(event_block: &str) -> bool {
    let is_hint = event_block.lines().any(|l| {
        l.trim()
            .strip_prefix("event:")
            .is_some_and(|v| v.trim() == "hint")
    });
    is_hint && event_block.contains("rate_limit")
}

/// Parse request/response_message_id from the first SSE ready event
///
/// Format: `event: ready\ndata: {"request_message_id":3,"response_message_id":4,...}\n\n`
///
/// Returns `(next_edit_id, stop_id)`, falling back to `(1, 4)` on parse failure
fn parse_ready_message_ids(chunk: &[u8]) -> (i64, i64) {
    let text = std::str::from_utf8(chunk).ok();
    if let Some(text) = text {
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ")
                && let Ok(val) = serde_json::from_str::<serde_json::Value>(data)
                && let (Some(r), Some(s)) = (
                    val.get("request_message_id").and_then(|v| v.as_i64()),
                    val.get("response_message_id").and_then(|v| v.as_i64()),
                )
            {
                return (r, s);
            }
        }
    }
    (1, 4)
}
