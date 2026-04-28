//! HTTP server facade — thin router shell exposing `OpenAIAdapter` and `AnthropicCompat` as HTTP endpoints
//!
//! Wraps the adapter / compat layers in an axum HTTP service.

mod error;
mod handlers;
mod stream;

use axum::{
    Json, Router,
    extract::Request,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use serde::Serialize;

use crate::anthropic_compat::AnthropicCompat;
use crate::config::Config;
use crate::openai_adapter::OpenAIAdapter;

use handlers::AppState;

/// Start the HTTP server
pub async fn run(config: Config) -> anyhow::Result<()> {
    let adapter = Arc::new(OpenAIAdapter::new(&config).await?);
    let anthropic_compat = Arc::new(AnthropicCompat::new(Arc::clone(&adapter)));
    let state = AppState {
        adapter,
        anthropic_compat,
    };
    let router = build_router(state.clone(), &config.server.api_tokens);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = TcpListener::bind(&addr).await?;
    log::info!(
        target: "http::server",
        "OpenAI-compatible base_url: http://{}",
        addr
    );
    log::info!(
        target: "http::server",
        "Anthropic-compatible base_url: http://{}/anthropic",
        addr
    );

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    log::info!(
        target: "http::server",
        "HTTP server stopped; cleaning up resources"
    );
    state.adapter.shutdown().await;
    log::info!(target: "http::server", "Cleanup complete");

    Ok(())
}

/// Build the router
/// Build the router with public and protected routes
fn build_router(state: AppState, api_tokens: &[crate::config::ApiToken]) -> Router {
    let has_auth = !api_tokens.is_empty();
    let tokens: Vec<String> = api_tokens.iter().map(|t| t.token.clone()).collect();

    let public = Router::new()
        .route("/", get(root))
        .with_state(state.clone());

    let protected = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/{id}", get(handlers::get_model))
        .route("/anthropic/v1/messages", post(handlers::anthropic_messages))
        .route("/anthropic/v1/models", get(handlers::anthropic_list_models))
        .route("/anthropic/v1/models/{id}", get(handlers::anthropic_get_model))
        .with_state(state);

    let router = if has_auth {
        let tokens_clone = tokens.clone();
        public.merge(
            protected.layer(middleware::from_fn(move |req, next| {
                let tokens = tokens_clone.clone();
                async move { auth_middleware(req, next, tokens).await }
            }))
        )
    } else {
        public.merge(protected)
    };
    router
}

/// Root endpoint that lists available API paths
async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        endpoints: vec![
            "/v1/chat/completions",
            "/v1/models",
            "/anthropic/v1/messages",
            "/anthropic/v1/models",
        ],
        project: "https://github.com/bkataru/ds-free-api".into(),
    })
}

/// Response for the root endpoint
#[derive(Serialize)]
struct RootResponse {
    endpoints: Vec<&'static str>,
    project: String,
}

/// API token authentication middleware
async fn auth_middleware(req: Request, next: Next, tokens: Vec<String>) -> Response {
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let valid = match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = header.strip_prefix("Bearer ").unwrap_or("");
            tokens.iter().any(|t| t == token)
        }
        _ => false,
    };

    if !valid {
        log::debug!(target: "http::response", "401 unauthorized request");
        return error::ServerError::Unauthorized.into_response();
    }

    next.run(req).await
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    log::info!(
        target: "http::server",
        "Shutdown signal received; starting graceful shutdown"
    );
}
