//! 配置加载模块 —— 统一配置入口
//!
//! 支持 `-c <path>` 命令行参数，默认值见下方函数。
//! config.toml 中注释项使用代码默认值。

use serde::Deserialize;
use std::path::Path;

/// 应用配置根结构
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// 账号池（必需）
    pub accounts: Vec<Account>,
    /// DeepSeek 相关配置
    #[serde(default)]
    pub deepseek: DeepSeekConfig,
}

/// 单个账号配置
#[derive(Debug, Clone, Deserialize)]
pub struct Account {
    /// 邮箱（与 mobile 二选一）
    pub email: String,
    /// 手机号（与 email 二选一）
    pub mobile: String,
    /// 区号（与 mobile 配合使用，如 "+86"）
    pub area_code: String,
    /// 密码
    pub password: String,
}

/// DeepSeek 客户端配置
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekConfig {
    /// API 基础地址
    #[serde(default = "default_api_base")]
    pub api_base: String,
    /// WASM 文件完整 URL（PoW 计算所需，版本号可能变动）
    #[serde(default = "default_wasm_url")]
    pub wasm_url: String,
    /// User-Agent 请求头
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    /// X-Client-Version 请求头（用于 expert 模型等功能）
    #[serde(default = "default_client_version")]
    pub client_version: String,
    /// X-Client-Platform 请求头
    #[serde(default = "default_client_platform")]
    pub client_platform: String,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_base: default_api_base(),
            wasm_url: default_wasm_url(),
            user_agent: default_user_agent(),
            client_version: default_client_version(),
            client_platform: default_client_platform(),
        }
    }
}

/// 默认 API 基础地址
fn default_api_base() -> String {
    "https://chat.deepseek.com/api/v0".to_string()
}

/// 默认 WASM 文件 URL（版本号可能变动，建议配置文件中显式指定）
fn default_wasm_url() -> String {
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm".to_string()
}

/// 默认 User-Agent
fn default_user_agent() -> String {
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36".to_string()
}

/// 默认 X-Client-Version
fn default_client_version() -> String {
    "1.8.0".to_string()
}

/// 默认 X-Client-Platform
fn default_client_platform() -> String {
    "web".to_string()
}

impl Config {
    /// 从指定路径加载配置
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::de::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// 解析命令行参数并加载配置
    ///
    /// 支持 `-c <path>` 指定配置文件路径，默认使用 `../config.toml`
    pub fn load_with_args(args: impl Iterator<Item = String>) -> Result<Self, ConfigError> {
        let mut config_path = None;
        let mut iter = args.skip(1); // 跳过程序名

        while let Some(arg) = iter.next() {
            if arg == "-c" {
                if let Some(path) = iter.next() {
                    config_path = Some(path);
                } else {
                    return Err(ConfigError::Cli("-c 参数需要指定路径".to_string()));
                }
            }
        }

        let path = config_path.unwrap_or_else(|| "config.toml".to_string());
        Self::load(&path)
    }

    /// 验证配置有效性
    fn validate(&self) -> Result<(), ConfigError> {
        if self.accounts.is_empty() {
            return Err(ConfigError::Validation("至少需要一个账号配置".to_string()));
        }
        Ok(())
    }
}

/// 配置加载错误类型
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML 解析错误: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("配置验证错误: {0}")]
    Validation(String),
    #[error("命令行参数错误: {0}")]
    Cli(String),
}
