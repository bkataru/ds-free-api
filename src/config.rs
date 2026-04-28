//! Configuration loading — single entry point for runtime settings
//!
//! Supports `-c <path>` CLI arguments; defaults are documented on the helpers below.
//! Commented-out keys in `config.toml` fall back to these code defaults.

use serde::Deserialize;
use std::path::Path;

/// Root application configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Account pool (required)
    pub accounts: Vec<Account>,
    /// DeepSeek-related settings
    #[serde(default)]
    pub deepseek: DeepSeekConfig,
    /// HTTP server settings (required)
    pub server: ServerConfig,
}

/// Single account entry
#[derive(Debug, Clone, Deserialize)]
pub struct Account {
    /// Email (use either this or `mobile`)
    pub email: String,
    /// Phone number (use either this or `email`)
    pub mobile: String,
    /// Country/region code paired with `mobile` (e.g. "+86")
    pub area_code: String,
    /// Password
    pub password: String,
}

/// DeepSeek client configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekConfig {
    /// API base URL
    #[serde(default = "default_api_base")]
    pub api_base: String,
    /// Full WASM asset URL (required for PoW; version may change)
    #[serde(default = "default_wasm_url")]
    pub wasm_url: String,
    /// `User-Agent` header
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    /// `X-Client-Version` header (used for expert model features, etc.)
    #[serde(default = "default_client_version")]
    pub client_version: String,
    /// `X-Client-Platform` header
    #[serde(default = "default_client_platform")]
    pub client_platform: String,
    /// Supported model types; each maps to an OpenAI `model_id` as `deepseek-<type>`
    #[serde(default = "default_model_types")]
    pub model_types: Vec<String>,
    /// Per-type input token limits (aligned with `model_types` by index)
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: Vec<u32>,
    /// Per-type output token limits (aligned with `model_types` by index)
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: Vec<u32>,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_base: default_api_base(),
            wasm_url: default_wasm_url(),
            user_agent: default_user_agent(),
            client_version: default_client_version(),
            client_platform: default_client_platform(),
            model_types: default_model_types(),
            max_input_tokens: default_max_input_tokens(),
            max_output_tokens: default_max_output_tokens(),
        }
    }
}

fn default_model_types() -> Vec<String> {
    vec!["default".to_string(), "expert".to_string()]
}

fn default_max_input_tokens() -> Vec<u32> {
    vec![1_048_576, 1_048_576]
}

fn default_max_output_tokens() -> Vec<u32> {
    vec![384_000, 384_000]
}

impl DeepSeekConfig {
    /// Build the OpenAI-facing model registry map
    ///
    /// Keys are lowercased `model_id` values (e.g. `deepseek-default`); values are internal `model_type` strings (e.g. `default`).
    pub fn model_registry(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        for ty in &self.model_types {
            map.insert(format!("deepseek-{}", ty).to_lowercase(), ty.clone());
        }
        map
    }
}

/// HTTP server settings (required)
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Bind address
    pub host: String,
    /// Listen port
    pub port: u16,
    /// API access tokens; leave empty to disable auth
    #[serde(default)]
    pub api_tokens: Vec<ApiToken>,
    // TODO: admin_password — add when admin panel endpoints ship
}

/// API access token entry
#[derive(Debug, Clone, Deserialize)]
pub struct ApiToken {
    /// Secret value (e.g. `sk-...`)
    pub token: String,
    /// Human-readable label
    #[serde(default)]
    pub description: String,
}

/// Default DeepSeek API base URL
fn default_api_base() -> String {
    "https://chat.deepseek.com/api/v0".to_string()
}

/// Default WASM asset URL (version may change; prefer setting this explicitly in config)
fn default_wasm_url() -> String {
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm".to_string()
}

/// Default `User-Agent`
fn default_user_agent() -> String {
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36".to_string()
}

/// Default `X-Client-Version`
fn default_client_version() -> String {
    "1.8.0".to_string()
}

/// Default `X-Client-Platform`
fn default_client_platform() -> String {
    "web".to_string()
}

impl Config {
    /// Load configuration from a file path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::de::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Parse CLI arguments and load configuration
    ///
    /// Supports `-c <path>` to choose a config file; defaults to `config.toml`
    pub fn load_with_args(args: impl Iterator<Item = String>) -> Result<Self, ConfigError> {
        let mut config_path = None;
        let mut iter = args.skip(1); // skip argv[0]

        while let Some(arg) = iter.next() {
            if arg == "-c" {
                if let Some(path) = iter.next() {
                    config_path = Some(path);
                } else {
                    return Err(ConfigError::Cli(
                        "-c requires a path argument".to_string(),
                    ));
                }
            }
        }

        let path = config_path.unwrap_or_else(|| "config.toml".to_string());
        Self::load(&path)
    }

    /// Validate configuration invariants
    fn validate(&self) -> Result<(), ConfigError> {
        if self.accounts.is_empty() {
            return Err(ConfigError::Validation(
                "at least one account entry is required".to_string(),
            ));
        }
        if self.deepseek.model_types.is_empty() {
            return Err(ConfigError::Validation(
                "model_types cannot be empty".to_string(),
            ));
        }
        let n = self.deepseek.model_types.len();
        if self.deepseek.max_input_tokens.len() != n {
            return Err(ConfigError::Validation(format!(
                "max_input_tokens length ({}) must match model_types length ({})",
                self.deepseek.max_input_tokens.len(),
                n
            )));
        }
        if self.deepseek.max_output_tokens.len() != n {
            return Err(ConfigError::Validation(format!(
                "max_output_tokens length ({}) must match model_types length ({})",
                self.deepseek.max_output_tokens.len(),
                n
            )));
        }
        Ok(())
    }
}

/// Errors produced while loading or validating configuration
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("configuration validation error: {0}")]
    Validation(String),
    #[error("CLI argument error: {0}")]
    Cli(String),
}
