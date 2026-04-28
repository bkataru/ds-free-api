//! Anthropic facade — exposes an Anthropic-shaped API backed by [`crate::openai_adapter`].
//!
//! This layer never calls [`crate::ds_core`] directly; it maps payloads through [`crate::openai_adapter`].
//! Flow: Anthropic JSON → OpenAI-compatible request mapping → DeepSeek core → Anthropic-shaped responses.

mod models;
pub(crate) mod request;
pub(crate) mod response;

/// Anthropic-style streaming response (SSE bytes).
pub type StreamResponse = Pin<Box<dyn Stream<Item = Result<Bytes, AnthropicCompatError>> + Send>>;

use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::Stream;
use log::debug;

use crate::openai_adapter::{ChatResult, OpenAIAdapter, OpenAIAdapterError};

/// Compatibility adapter for `/v1/messages` & `/v1/models`.
pub struct AnthropicCompat {
    openai_adapter: Arc<OpenAIAdapter>,
}

impl AnthropicCompat {
    /// Wrap an existing [`OpenAIAdapter`].
    pub fn new(openai_adapter: Arc<OpenAIAdapter>) -> Self {
        Self { openai_adapter }
    }

    /// `POST /v1/messages` (non-streaming)
    ///
    /// Map Anthropic → OpenAI for the completion call, then map the JSON back to an Anthropic `Message`.
    pub async fn messages(
        &self,
        body: &[u8],
        request_id: &str,
    ) -> Result<ChatResult<Vec<u8>>, AnthropicCompatError> {
        debug!(target: "anthropic_compat", "received /v1/messages request");
        let openai_body = request::to_openai_request(body)?;
        let result = self.openai_adapter
            .chat_completions(&openai_body, request_id)
            .await
            .map_err(AnthropicCompatError::from)?;
        let data = response::from_chat_completion_bytes(&result.data)
            .map_err(|e| AnthropicCompatError::Internal(format!("json error: {}", e)))?;
        Ok(ChatResult {
            data,
            account_id: result.account_id,
        })
    }

    /// `POST /v1/messages` (streaming)
    ///
    /// Map Anthropic → OpenAI and return an Anthropic-shaped SSE byte stream.
    pub async fn messages_stream(
        &self,
        body: &[u8],
        request_id: &str,
    ) -> Result<ChatResult<StreamResponse>, AnthropicCompatError> {
        debug!(target: "anthropic_compat", "received streaming /v1/messages request");
        let openai_body = request::to_openai_request(body)?;
        let openai_req = self
            .openai_adapter
            .parse_request(&openai_body)
            .map_err(AnthropicCompatError::from)?;
        let input_tokens = openai_req.prompt_tokens;
        let chat_resp = self
            .openai_adapter
            .try_chat(openai_req.ds_req, request_id)
            .await
            .map_err(OpenAIAdapterError::from)?;
        let repair_fn = self.openai_adapter.create_repair_fn(request_id);
        let data = crate::openai_adapter::response::stream(
            chat_resp.stream,
            openai_req.model,
            openai_req.include_usage,
            openai_req.include_obfuscation,
            openai_req.stop,
            openai_req.prompt_tokens,
            openai_req.tools_present,
            Some(repair_fn),
        );
        Ok(ChatResult {
            data: response::from_chat_completion_stream(data, input_tokens),
            account_id: chat_resp.account_id,
        })
    }

    /// `GET /v1/models`
    ///
    /// Return the catalog in Anthropic's list shape.
    pub fn list_models(&self) -> Vec<u8> {
        debug!(target: "anthropic_compat", "received /v1/models list request");
        models::list(&self.openai_adapter)
    }

    /// `GET /v1/models/{model_id}`
    ///
    /// Return a single model record in Anthropic's detail shape.
    pub fn get_model(&self, model_id: &str) -> Option<Vec<u8>> {
        debug!(target: "anthropic_compat", "lookup model: {}", model_id);
        models::get(&self.openai_adapter, model_id)
    }
}

/// Errors surfaced by the Anthropic compatibility layer.
#[derive(Debug, thiserror::Error)]
pub enum AnthropicCompatError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("service overloaded")]
    Overloaded,
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<OpenAIAdapterError> for AnthropicCompatError {
    fn from(e: OpenAIAdapterError) -> Self {
        match e {
            OpenAIAdapterError::BadRequest(msg) => Self::BadRequest(msg),
            OpenAIAdapterError::Overloaded => Self::Overloaded,
            OpenAIAdapterError::ProviderError(msg) => Self::Internal(msg),
            OpenAIAdapterError::Internal(msg) => Self::Internal(msg),
            OpenAIAdapterError::ToolCallRepairNeeded(msg) => Self::Internal(msg),
        }
    }
}

impl AnthropicCompatError {
    /// HTTP status code for this error variant.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Overloaded => 429,
            Self::Internal(_) => 500,
        }
    }
}
