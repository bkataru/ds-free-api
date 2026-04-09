//! 账号池管理 —— 多账号负载均衡
//!
//! 1 account = 1 session = 1 concurrency。多并发需横向扩展账号数。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::config::Account as AccountConfig;
use crate::ds_core::client::{
    ClientError, CompletionPayload, DsClient, LoginPayload, UpdateTitlePayload,
};
use crate::ds_core::pow::{PowError, PowSolver};
use futures::TryStreamExt;
use log::{debug, info, warn};

/// 账号状态信息
pub struct AccountStatus {
    pub email: String,
    pub mobile: String,
}

pub struct Account {
    token: String,
    email: String,
    mobile: String,
    session_id: String,
    is_busy: AtomicBool,
}

impl Account {
    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn is_busy(&self) -> bool {
        self.is_busy.load(Ordering::Relaxed)
    }
}

/// 持有期间账号标记为 busy，Drop 时自动释放
pub struct AccountGuard {
    account: Arc<Account>,
}

impl AccountGuard {
    pub fn account(&self) -> &Account {
        &self.account
    }
}

impl Drop for AccountGuard {
    fn drop(&mut self) {
        self.account.is_busy.store(false, Ordering::Relaxed);
    }
}

pub struct AccountPool {
    creds: Vec<AccountConfig>,
    accounts: Vec<Arc<Account>>,
    index: AtomicUsize,
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// 所有账号初始化失败（没有可用账号）
    #[error("所有账号初始化失败")]
    AllAccountsFailed,

    /// 下游客户端错误（网络、API 错误等）
    #[error("客户端错误: {0}")]
    Client(#[from] ClientError),

    /// PoW 计算失败（WASM 执行错误）
    #[error("PoW 计算失败: {0}")]
    Pow(#[from] PowError),

    /// 账号配置验证失败
    #[error("账号配置错误: {0}")]
    Validation(String),
}

impl AccountPool {
    pub fn new(creds: Vec<AccountConfig>) -> Self {
        Self {
            creds,
            accounts: Vec::new(),
            index: AtomicUsize::new(0),
        }
    }

    pub async fn init(&mut self, client: &DsClient, solver: &PowSolver) -> Result<(), PoolError> {
        use futures::future::join_all;

        let creds = std::mem::take(&mut self.creds);

        // 全并发初始化所有账号
        let futures: Vec<_> = creds
            .into_iter()
            .map(|creds| {
                let client = client.clone();
                let solver = solver.clone();
                async move {
                    let mobile = creds.mobile.clone();
                    match init_account(&creds, &client, &solver).await {
                        Ok(account) => {
                            info!(target: "ds_core::accounts", "账号 {} 初始化成功", mobile);
                            Some(Arc::new(account))
                        }
                        Err(e) => {
                            warn!(target: "ds_core::accounts", "账号 {} 初始化失败: {}", mobile, e);
                            None
                        }
                    }
                }
            })
            .collect();

        let results = join_all(futures).await;
        self.accounts = results.into_iter().flatten().collect();

        if self.accounts.is_empty() {
            return Err(PoolError::AllAccountsFailed);
        }

        Ok(())
    }

    /// 轮询获取一个空闲的账号
    pub fn get_account(&self) -> Option<AccountGuard> {
        if self.accounts.is_empty() {
            return None;
        }

        let idx = self.index.fetch_add(1, Ordering::Relaxed) % self.accounts.len();

        for i in 0..self.accounts.len() {
            let account = &self.accounts[(idx + i) % self.accounts.len()];
            if !account.is_busy()
                && account
                    .is_busy
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                return Some(AccountGuard {
                    account: account.clone(),
                });
            }
        }

        None
    }

    /// 获取所有账号的详细状态
    pub fn account_statuses(&self) -> Vec<AccountStatus> {
        self.accounts
            .iter()
            .map(|a| AccountStatus {
                email: a.email.clone(),
                mobile: a.mobile.clone(),
            })
            .collect()
    }

    /// 优雅关闭：清理所有账号的 session
    pub async fn shutdown(&self, client: &DsClient) {
        use futures::future::join_all;

        let futures: Vec<_> = self
            .accounts
            .iter()
            .map(|account| {
                let session_id = account.session_id();
                let token = account.token().to_string();
                let client = client.clone();
                async move {
                    if let Err(e) = client.delete_session(&token, session_id).await {
                        warn!(
                            target: "ds_core::accounts",
                            "清理 session 失败 ({}): {}",
                            &token[..8.min(token.len())],
                            e
                        );
                    }
                }
            })
            .collect();

        join_all(futures).await;
    }
}

async fn init_account(
    creds: &AccountConfig,
    client: &DsClient,
    solver: &PowSolver,
) -> Result<Account, PoolError> {
    let mut last_error = None;

    for attempt in 1..=3 {
        match try_init_account(creds, client, solver).await {
            Ok(account) => return Ok(account),
            Err(e) => {
                last_error = Some(e);
                if attempt < 3 {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    }

    Err(last_error.unwrap())
}

async fn try_init_account(
    creds: &AccountConfig,
    client: &DsClient,
    solver: &PowSolver,
) -> Result<Account, PoolError> {
    // 验证：email 和 mobile 至少一个非空
    if creds.email.is_empty() && creds.mobile.is_empty() {
        return Err(PoolError::Validation(
            "email 和 mobile 不能同时为空".to_string(),
        ));
    }

    let login_payload = LoginPayload {
        email: if creds.email.is_empty() {
            None
        } else {
            Some(creds.email.clone())
        },
        mobile: if creds.mobile.is_empty() {
            None
        } else {
            Some(creds.mobile.clone())
        },
        password: creds.password.clone(),
        area_code: if creds.area_code.is_empty() {
            None
        } else {
            Some(creds.area_code.clone())
        },
        device_id: String::new(),
        os: "web".to_string(),
    };

    let login_data = client.login(&login_payload).await?;
    debug!(
        target: "ds_core::client",
        "登录响应: code={}, msg={}, user_id={}, email={:?}, mobile={:?}",
        login_data.code,
        login_data.msg,
        login_data.user.id,
        login_data.user.email,
        login_data.user.mobile_number
    );
    let token = login_data.user.token;

    let session_id = client.create_session(&token).await?;

    // Health check: 先发消息验证 session 有效
    health_check(&token, &session_id, client, solver).await?;

    let title_payload = UpdateTitlePayload {
        chat_session_id: session_id.clone(),
        title: "managed-by-ai-free-api".to_string(),
    };
    client.update_title(&token, &title_payload).await?;

    Ok(Account {
        token,
        email: creds.email.clone(),
        mobile: creds.mobile.clone(),
        session_id,
        is_busy: AtomicBool::new(false),
    })
}

async fn health_check(
    token: &str,
    session_id: &str,
    client: &DsClient,
    solver: &PowSolver,
) -> Result<(), PoolError> {
    debug!(target: "ds_core::accounts", "[health_check] 步骤1: 获取 PoW challenge...");
    let challenge = client.create_pow_challenge(token).await?;
    debug!(target: "ds_core::accounts", "[health_check] 步骤1完成: challenge 获取成功");

    debug!(target: "ds_core::accounts", "[health_check] 步骤2: 计算 PoW...");
    let result = solver.solve(&challenge)?;
    let pow_header = result.to_header();
    debug!(target: "ds_core::accounts", "[health_check] 步骤2完成: PoW 计算成功");

    debug!(target: "ds_core::accounts", "[health_check] 步骤3: 发送 completion 请求...");
    let payload = CompletionPayload {
        chat_session_id: session_id.to_string(),
        parent_message_id: None,
        model_type: "expert".to_string(),
        prompt: "只回复`Hello, world!`".to_string(),
        ref_file_ids: vec![],
        thinking_enabled: false,
        search_enabled: false,
        preempt: false,
    };

    debug!(target: "ds_core::accounts", "[health_check] 步骤4: 获取 completion 流...");
    let mut stream = client.completion(token, &pow_header, &payload).await?;
    debug!(target: "ds_core::accounts", "[health_check] 步骤4: 消费流确保消息写入...");
    // 消费流
    while let Some(chunk) = stream.try_next().await? {
        let text = String::from_utf8_lossy(&chunk);
        debug!(target: "ds_core::accounts", "[health_check] 收到数据: {}", text.trim());
    }
    debug!(target: "ds_core::accounts", "[health_check] 步骤4完成: 流消费完成");

    debug!(target: "ds_core::accounts", "[health_check] 全部完成!");
    Ok(())
}
