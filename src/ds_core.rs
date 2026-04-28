//! DeepSeek core — adapter layer from the OpenAI API shape to DeepSeek.
//!
//! Exposes the minimal surface area: [`DeepSeekCore`], [`CoreError`], [`ChatRequest`].

mod accounts;
mod client;
mod completions;
mod pow;

pub use accounts::AccountStatus;
pub use completions::{ChatRequest, ChatResponse};

use crate::config::Config;
use accounts::AccountPool;
use client::{ClientError, DsClient};
use pow::{PowError, PowSolver};

/// Core-layer error surface.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Overload: every account is busy or unhealthy.
    #[error("no available account")]
    Overloaded,

    /// Proof-of-work solving failed.
    #[error("proof of work failed: {0}")]
    ProofOfWorkFailed(#[from] PowError),

    /// Provider-side failure: network, business errors, expired tokens, etc.
    #[error("provider: {0}")]
    ProviderError(String),

    /// Stream processing failure: disconnects and similar.
    #[error("stream error: {0}")]
    Stream(String),
}

impl From<ClientError> for CoreError {
    fn from(e: ClientError) -> Self {
        CoreError::ProviderError(e.to_string())
    }
}

pub struct DeepSeekCore {
    completions: crate::ds_core::completions::Completions,
}

impl DeepSeekCore {
    pub async fn new(config: &Config) -> Result<Self, CoreError> {
        let client = DsClient::new(
            config.deepseek.api_base.clone(),
            config.deepseek.wasm_url.clone(),
            config.deepseek.user_agent.clone(),
            config.deepseek.client_version.clone(),
            config.deepseek.client_platform.clone(),
        );

        let wasm_bytes = client.get_wasm().await?;
        let solver = PowSolver::new(&wasm_bytes)?;

        let mut pool = AccountPool::new();
        pool.init(
            config.accounts.clone(),
            config.deepseek.model_types.clone(),
            &client,
            &solver,
        )
        .await
        .map_err(|e| match e {
            accounts::PoolError::AllAccountsFailed => {
                CoreError::ProviderError("all accounts failed to initialize".to_string())
            }
            accounts::PoolError::Client(e) => CoreError::ProviderError(e.to_string()),
            accounts::PoolError::Pow(e) => CoreError::ProofOfWorkFailed(e),
            accounts::PoolError::Validation(msg) => {
                CoreError::ProviderError(format!("configuration error: {}", msg))
            }
        })?;

        let completions = crate::ds_core::completions::Completions::new(client, solver, pool);

        Ok(Self { completions })
    }

    /// Start a chat request; returns SSE byte stream + account identifier.
    ///
    /// The leased account is released when the stream ends or is dropped.
    pub async fn v0_chat(
        &self,
        req: ChatRequest,
        request_id: &str,
    ) -> Result<ChatResponse, CoreError> {
        self.completions.v0_chat(req, request_id).await
    }

    pub fn account_statuses(&self) -> Vec<AccountStatus> {
        self.completions.account_statuses()
    }

    /// Graceful shutdown: delete sessions for every account.
    pub async fn shutdown(&self) {
        self.completions.shutdown().await;
    }
}
