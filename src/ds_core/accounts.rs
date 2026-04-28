//! Account pool — multi-account load balancing.
//!
//! 1 account = 1 session = 1 concurrent request. Scale out by adding accounts.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::SystemTime;
use std::sync::RwLock;

use crate::config::Account as AccountConfig;
use crate::ds_core::client::{
    ClientError, CompletionPayload, DsClient, LoginPayload, UpdateTitlePayload,
};
use crate::ds_core::pow::{PowError, PowSolver};
use futures::TryStreamExt;
use log::{debug, info, warn};

/// Basic account identity for status reporting.
/// Per-session state: session id + next edit_message_id
struct SessionInfo {
    id: String,
    #[allow(dead_code)]
    next_message_id: i64,
}

pub struct AccountStatus {
    pub email: String,
    pub mobile: String,
}

pub struct Account {
    token: String,
    email: String,
    mobile: String,
    sessions: RwLock<HashMap<String, SessionInfo>>,
    is_busy: AtomicBool,
    last_released: AtomicI64,
}

impl Account {
    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn session_id(&self, model_type: &str) -> Option<String> {
        self.sessions
            .read()
            .unwrap()
            .get(model_type)
            .map(|s| s.id.clone())
    }

    #[allow(dead_code)]
    pub fn next_message_id(&self, model_type: &str) -> i64 {
        self.sessions
            .read()
            .unwrap()
            .get(model_type)
            .map(|s| s.next_message_id)
            .unwrap_or(1)
    }

    #[allow(dead_code)]
    pub fn set_next_message_id(&self, model_type: &str, id: i64) {
        if let Some(s) = self.sessions.write().unwrap().get_mut(model_type) {
            s.next_message_id = id;
        }
    }

    pub fn is_busy(&self) -> bool {
        self.is_busy.load(Ordering::Relaxed)
    }

    pub fn display_id(&self) -> &str {
        if !self.email.is_empty() {
            &self.email
        } else {
            &self.mobile
        }
    }
}

/// Marks the account busy for the guard's lifetime; released on `Drop`.
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
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        self.account.last_released.store(now_ms, Ordering::Relaxed);
    }
}

pub struct AccountPool {
    accounts: Vec<Arc<Account>>,
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// Every account failed initialization (no healthy pool members).
    #[error("all accounts failed to initialize")]
    AllAccountsFailed,

    /// Downstream client error (network, upstream API, etc.).
    #[error("client error: {0}")]
    Client(#[from] ClientError),

    /// Proof-of-work failed (WASM execution error).
    #[error("proof of work failed: {0}")]
    Pow(#[from] PowError),

    /// Account configuration validation failed.
    #[error("account configuration error: {0}")]
    Validation(String),
}

impl AccountPool {
    pub fn new() -> Self {
        Self {
            accounts: Vec::new(),
        }
    }

    pub async fn init(
        &mut self,
        creds: Vec<AccountConfig>,
        model_types: Vec<String>,
        client: &DsClient,
        solver: &PowSolver,
    ) -> Result<(), PoolError> {
        use futures::future::join_all;
        use tokio::sync::Semaphore;


        // Cap concurrent initialization to avoid overwhelming DeepSeek and local connection pool.
        let semaphore = std::sync::Arc::new(Semaphore::new(13));
        let futures: Vec<_> = creds
            .into_iter()
            .map(|creds| {
                let client = client.clone();
                let solver = solver.clone();
                let model_types = model_types.clone();
                let sem = semaphore.clone();
                async move {
                    let _permit = sem.acquire().await.expect("semaphore not closed");
                    let display_id = if creds.mobile.is_empty() {
                        creds.email.clone()
                    } else {
                        creds.mobile.clone()
                    };
                    match init_account(&creds, &client, &solver, &model_types).await {
                        Ok(account) => {
                            info!(target: "ds_core::accounts", "account {} initialized successfully", display_id);
                            Some(Arc::new(account))
                        }
                        Err(e) => {
                            warn!(target: "ds_core::accounts", "account {} failed to initialize: {}", display_id, e);
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

    /// Acquire the free account that has been idle the longest (must have a session for `model_type`).
    ///
    /// Iterates all accounts, picks the one with the oldest release timestamp to maximize reuse gaps.
    #[allow(dead_code)]
    /// Acquire the free account that has been idle the longest (must have a session for `model_type`).
    ///
    /// Iterates all accounts, picks the one with the oldest release timestamp to maximize reuse gaps.
    pub fn get_account_by_model(&self, model_type: &str) -> Option<AccountGuard> {
        if self.accounts.is_empty() {
            return None;
        }

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut best_idx: Option<usize> = None;
        let mut best_idle = i64::MIN;

        for (i, account) in self.accounts.iter().enumerate() {
            if account.session_id(model_type).is_none() || account.is_busy() {
                continue;
            }
            let idle = now_ms - account.last_released.load(Ordering::Relaxed);
            if idle > best_idle {
                best_idle = idle;
                best_idx = Some(i);
            }
        }

        let idx = best_idx?;
        let account = &self.accounts[idx];
        account
            .is_busy
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .ok()?;
        Some(AccountGuard {
            account: Arc::clone(account),
        })
    }


    /// Acquire the free account that has been idle the longest (no session filtering).
    ///
    /// Used by the temp-session flow where sessions are created per-request and
    /// model_type filtering does not apply.
    pub fn get_account(&self) -> Option<AccountGuard> {
        if self.accounts.is_empty() {
            return None;
        }

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut best_idx: Option<usize> = None;
        let mut best_idle = i64::MIN;

        for (i, account) in self.accounts.iter().enumerate() {
            if account.is_busy() {
                continue;
            }
            let idle = now_ms - account.last_released.load(Ordering::Relaxed);
            if idle > best_idle {
                best_idle = idle;
                best_idx = Some(i);
            }
        }

        let idx = best_idx?;
        let account = &self.accounts[idx];
        account
            .is_busy
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .ok()?;
        Some(AccountGuard {
            account: Arc::clone(account),
        })
    }
    /// Return a snapshot of pool members' identity fields
    pub fn account_statuses(&self) -> Vec<AccountStatus> {
        self.accounts
            .iter()
            .map(|a| AccountStatus {
                email: a.email.clone(),
                mobile: a.mobile.clone(),
            })
            .collect()
    }

    /// Graceful shutdown: delete every session for every account.
    pub async fn shutdown(&self, client: &DsClient) {
        use futures::future::join_all;

        let futures: Vec<_> = self
            .accounts
            .iter()
            .flat_map(|account| {
                let token = account.token().to_string();
                account
                    .sessions
                    .read()
                    .unwrap()
                    .values()
                    .map(|s| s.id.clone())
                    .map(move |session_id| {
                        let client = client.clone();
                        let token = token.clone();
                        async move {
                            if let Err(e) = client.delete_session(&token, &session_id).await {
                                warn!(
                                    target: "ds_core::accounts",
                                    "failed to delete session ({}): {}",
                                    &token[..8.min(token.len())],
                                    e
                                );
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        join_all(futures).await;
    }
}

async fn init_account(
    creds: &AccountConfig,
    client: &DsClient,
    solver: &PowSolver,
    model_types: &[String],
) -> Result<Account, PoolError> {
    let mut last_error = None;

    for attempt in 1..=3 {
        match try_init_account(creds, client, solver, model_types).await {
            Ok(account) => return Ok(account),
            Err(e) => {
                last_error = Some(e);
                if attempt < 3 {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    }

    Err(last_error.expect("loop must run at least once"))
}

async fn try_init_account(
    creds: &AccountConfig,
    client: &DsClient,
    solver: &PowSolver,
    model_types: &[String],
) -> Result<Account, PoolError> {
    // Require at least one of email or mobile.
    if creds.email.is_empty() && creds.mobile.is_empty() {
        return Err(PoolError::Validation(
            "email and mobile cannot both be empty".to_string(),
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
        "login response: code={}, msg={}, user_id={}, email={:?}, mobile={:?}",
        login_data.code,
        login_data.msg,
        login_data.user.id,
        login_data.user.email,
        login_data.user.mobile_number
    );
    let token = login_data.user.token;

    let mut sessions = HashMap::new();
    for model_type in model_types {
        let session_id = client.create_session(&token).await?;
        health_check(&token, &session_id, client, solver, model_type).await?;

        let title_payload = UpdateTitlePayload {
            chat_session_id: session_id.clone(),
            title: format!("auto-managed-{}-DO-NOT-DELETE", model_type),
        };
        client.update_title(&token, &title_payload).await?;

        sessions.insert(model_type.clone(), SessionInfo { id: session_id, next_message_id: 1 });
    }

    Ok(Account {
        token,
        email: creds.email.clone(),
        mobile: creds.mobile.clone(),
        sessions: RwLock::new(sessions),
        is_busy: AtomicBool::new(false),
        last_released: AtomicI64::new(0),

    })
}

async fn health_check(
    token: &str,
    session_id: &str,
    client: &DsClient,
    solver: &PowSolver,
    model_type: &str,
) -> Result<(), PoolError> {
    debug!(target: "ds_core::accounts", "health_check model_type={}", model_type);
    let challenge = client.create_pow_challenge(token, "/api/v0/chat/completion").await?;

    let result = solver.solve(&challenge)?;
    let pow_header = result.to_header();

    let payload = CompletionPayload {
        chat_session_id: session_id.to_string(),
        parent_message_id: None,
        model_type: model_type.to_string(),
        prompt: "Reply with only `Hello, world!`".to_string(),
        ref_file_ids: vec![],
        thinking_enabled: false,
        search_enabled: false,
        preempt: false,
    };

    let mut stream = client.completion(token, &pow_header, &payload).await?;
    // Drain the stream so the message is persisted
    while let Some(chunk) = stream.try_next().await? {
        let _ = chunk;
    }

    debug!(target: "ds_core::accounts", "health_check finished model_type={}", model_type);
    Ok(())
}
