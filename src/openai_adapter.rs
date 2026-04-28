//! OpenAI-compatible protocol adapter — bidirectional mapping between OpenAI JSON and `ds_core` wire formats.
//!
//! Converts OpenAI-style HTTP requests into internal `ds_core` inputs and maps `ds_core` responses back to OpenAI-compatible JSON.
//!
//! Public surface is intentionally small: `OpenAIAdapter`, `OpenAIAdapterError`.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;

use crate::ds_core::{CoreError, DeepSeekCore};

mod models;
pub(crate) mod request;
pub(crate) mod response;
mod types;

/// Streamed response type (SSE bytes).
pub type StreamResponse = Pin<Box<dyn Stream<Item = Result<Bytes, OpenAIAdapterError>> + Send>>;

/// Adapter-layer generic result wrapper: carries request result and account identifier
pub struct ChatResult<T> {
    pub data: T,
    pub account_id: String,
}

/// OpenAI-compatible adapter over `DeepSeekCore`.
pub struct OpenAIAdapter {
    ds_core: Arc<DeepSeekCore>,
    model_types: Vec<String>,
    model_registry: std::collections::HashMap<String, String>,
    max_input_tokens: Vec<u32>,
    max_output_tokens: Vec<u32>,
}

impl OpenAIAdapter {
    /// Build a new adapter instance.
    pub async fn new(config: &crate::config::Config) -> Result<Self, OpenAIAdapterError> {
        let ds_core = Arc::new(DeepSeekCore::new(config).await?);
        let model_registry = config.deepseek.model_registry();
        Ok(Self {
            ds_core,
            model_types: config.deepseek.model_types.clone(),
            model_registry,
            max_input_tokens: config.deepseek.max_input_tokens.clone(),
            max_output_tokens: config.deepseek.max_output_tokens.clone(),
        })
    }

    /// Parse the HTTP body into an `AdapterRequest` (single parse; avoids double-decoding JSON).
    pub(crate) fn parse_request(
        &self,
        body: &[u8],
    ) -> Result<request::AdapterRequest, OpenAIAdapterError> {
        request::parse(body, &self.model_registry)
    }

    /// `POST /v1/chat/completions` (non-streaming).
    ///
    /// Reuses the streaming path internally and aggregates SSE into one JSON object.
    pub async fn chat_completions(
        &self,
        body: &[u8],
        request_id: &str,
    ) -> Result<ChatResult<Vec<u8>>, OpenAIAdapterError> {
        let req = request::parse(body, &self.model_registry)?;
        let chat_resp = self.try_chat(req.ds_req, request_id).await?;
        let data =
            response::aggregate(chat_resp.stream, req.model, req.stop, req.prompt_tokens, req.tools_present).await?;
        Ok(ChatResult {
            data,
            account_id: chat_resp.account_id,
        })
    }

    /// `POST /v1/chat/completions` (streaming).
    pub async fn chat_completions_stream(
        &self,
        body: &[u8],
        request_id: &str,
    ) -> Result<ChatResult<StreamResponse>, OpenAIAdapterError> {
        let req = request::parse(body, &self.model_registry)?;
        let chat_resp = self.try_chat(req.ds_req, request_id).await?;
        let repair_fn = self.create_repair_fn(request_id);
        let data = response::stream(
            chat_resp.stream,
            req.model,
            req.include_usage,
            req.include_obfuscation,
            req.stop,
            req.prompt_tokens,
            req.tools_present,
            Some(repair_fn),
        );
        Ok(ChatResult {
            data,
            account_id: chat_resp.account_id,
        })
    }

    /// Internal helper: short delayed retries on `Overloaded` to smooth burst traffic.
    pub(crate) async fn try_chat(
        &self,
        req: crate::ds_core::ChatRequest,
        request_id: &str,
    ) -> Result<crate::ds_core::ChatResponse, CoreError> {
        const MAX_RETRIES: usize = 6;
        const BASE_DELAY_MS: u64 = 1000;

        for attempt in 0..MAX_RETRIES {
            match self.ds_core.v0_chat(req.clone(), request_id).await {
                Ok(resp) => return Ok(resp),
                Err(CoreError::Overloaded) if attempt + 1 < MAX_RETRIES => {
                    let delay = BASE_DELAY_MS * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(CoreError::Overloaded)
    }

    /// Raw DeepSeek SSE stream (no OpenAI-protocol conversion).
    ///
    /// Useful for stream analysis: comparing raw responses against OpenAI-converted
    /// output to pinpoint transformation bugs.
    pub async fn raw_chat_stream(&self, body: &[u8]) -> Result<StreamResponse, OpenAIAdapterError> {
        let req = request::parse(body, &self.model_registry)?;
        let resp = self.try_chat(req.ds_req, "raw").await?;
        Ok(Box::pin(
            resp.stream.map(|r| r.map_err(OpenAIAdapterError::from)),
        ))
    }

    /// `GET /v1/models`
    pub fn list_models(&self) -> Vec<u8> {
        models::list(
            &self.model_types,
            &self.max_input_tokens,
            &self.max_output_tokens,
        )
    }

    /// `GET /v1/models/{model_id}`
    pub fn get_model(&self, model_id: &str) -> Option<Vec<u8>> {
        models::get(
            &self.model_types,
            &self.max_input_tokens,
            &self.max_output_tokens,
            model_id,
        )
    }

    /// Upstream account pool status from `ds_core`.
    pub fn account_statuses(&self) -> Vec<crate::ds_core::AccountStatus> {
        self.ds_core.account_statuses()
    }

    /// Gracefully shut down upstream resources.
    pub async fn shutdown(&self) {
        self.ds_core.shutdown().await;
    }

    /// Builds the repair closure for tool_calls self-repair.
    /// The closure captures Arc<DeepSeekCore> and calls a repair model to fix
    /// malformed XML tool calls from the primary model.
    pub(crate) fn create_repair_fn(&self, request_id: &str) -> response::RepairFn {
        use std::sync::Arc;
        let core = self.ds_core.clone();
        let req_id = request_id.to_string();
        Arc::new(move |raw_xml: String| {
            let core = core.clone();
            let req_id = req_id.clone();
            Box::pin(async move {
                use crate::ds_core::ChatRequest;
                let prompt = format!(
                    "system: repair tool_calls\n\
                     Fix the following content into a valid JSON array of tool calls. \
                     Each element must have \"name\" (string) and \"arguments\" (object). \
                     Output ONLY the JSON array, no markdown, no explanation.\n\n\
                     Content to fix:\n{raw_xml}"
                );
                let req = ChatRequest {
                    prompt,
                    thinking_enabled: false,
                    search_enabled: false,
                    model_type: "default".to_string(),
                    files: vec![],
                };
                let resp = core.v0_chat(req, &req_id).await.map_err(OpenAIAdapterError::from)?;
                response::execute_tool_repair(resp.stream).await
            })
        })
    }
}

/// Adapter-visible error taxonomy.
#[derive(Debug, thiserror::Error)]
pub enum OpenAIAdapterError {
    /// Client request could not be parsed or validated.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// Temporary overload — no upstream `ds_core` account slot available.
    #[error("service overloaded")]
    Overloaded,

    /// Upstream/provider failure (transport, rejection, quota, …).
    #[error("provider error: {0}")]
    ProviderError(String),

    /// Internal invariant or adapter bug (serialization, stream bridging, …).
    #[error("internal error: {0}")]
    Internal(String),

    /// Malformed tool_calls XML — upstream RepairStream will attempt repair.
    #[error("tool_calls repair needed: {0}")]
    ToolCallRepairNeeded(String),
}

impl From<CoreError> for OpenAIAdapterError {
    fn from(e: CoreError) -> Self {
        match e {
            CoreError::Overloaded => Self::Overloaded,
            CoreError::ProofOfWorkFailed(err) => {
                Self::Internal(format!("proof of work failed: {}", err))
            }
            CoreError::ProviderError(msg) => Self::ProviderError(msg),
            CoreError::Stream(msg) => Self::Internal(msg),
        }
    }
}

impl From<serde_json::Error> for OpenAIAdapterError {
    fn from(e: serde_json::Error) -> Self {
        Self::Internal(format!("json serialization failed: {}", e))
    }
}

impl OpenAIAdapterError {
    /// Preferred HTTP status for this error variant.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Overloaded => 429,
            Self::ProviderError(_) => 502,
            Self::Internal(_) => 500,
            Self::ToolCallRepairNeeded(_) => 500,
        }
    }
}
