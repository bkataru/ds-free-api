//! Temporary email provisioning primitives.
//!
//! This module intentionally stops at inbox provisioning and verification-code
//! polling. Account registration remains a separate integration point so startup
//! cannot silently mass-register accounts.

use crate::config::{AccountProvisioningConfig, EmailProviderKind};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAIL_TM_API_BASE: &str = "https://api.mail.tm";
const RANDOM_LOCAL_PART_LEN: usize = 10;
const RANDOM_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
const PASSWORD_ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";

pub trait EmailProvider {
    fn create_inbox<'a>(
        &'a self,
        password_length: usize,
    ) -> Pin<Box<dyn Future<Output = Result<ProvisionedInbox, EmailProvisioningError>> + Send + 'a>>;

    fn list_messages<'a>(
        &'a self,
        inbox: &'a ProvisionedInbox,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<InboxMessage>, EmailProvisioningError>> + Send + 'a>>;

    fn fetch_message<'a>(
        &'a self,
        inbox: &'a ProvisionedInbox,
        message_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<InboxMessage, EmailProvisioningError>> + Send + 'a>>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvisionedInbox {
    pub provider: EmailProviderKind,
    pub address: String,
    pub mailbox_password: String,
    pub provider_token: String,
    pub created_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InboxMessage {
    pub id: String,
    pub from: Option<String>,
    pub subject: Option<String>,
    pub intro: Option<String>,
    pub text: Option<String>,
    pub html: Option<Vec<String>>,
    pub created_at: Option<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvisioningReport {
    pub created: Vec<ProvisionedInbox>,
    pub state_path: String,
}

#[derive(Debug, thiserror::Error)]
pub enum EmailProvisioningError {
    #[error("account provisioning is disabled")]
    Disabled,
    #[error("account provisioning config error: {0}")]
    InvalidConfig(String),
    #[error("temporary email provider returned no available domains")]
    NoDomains,
    #[error("temporary email provider error: {0}")]
    Provider(String),
    #[error("timed out waiting for verification email")]
    Timeout,
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub async fn prepare_inboxes(
    config: &AccountProvisioningConfig,
) -> Result<ProvisioningReport, EmailProvisioningError> {
    if !config.enabled {
        return Err(EmailProvisioningError::Disabled);
    }
    if config.max_inboxes_per_run == 0 {
        return Err(EmailProvisioningError::InvalidConfig(
            "max_inboxes_per_run must be greater than zero".to_string(),
        ));
    }

    let provider = provider_for(config.provider.clone());
    let mut created = Vec::with_capacity(config.max_inboxes_per_run as usize);

    for _ in 0..config.max_inboxes_per_run {
        created.push(provider.create_inbox(config.password_length).await?);
    }

    append_state(&config.state_path, &created)?;

    Ok(ProvisioningReport {
        created,
        state_path: config.state_path.clone(),
    })
}

pub async fn poll_verification_code(
    config: &AccountProvisioningConfig,
    inbox: &ProvisionedInbox,
) -> Result<String, EmailProvisioningError> {
    let provider = provider_for(inbox.provider.clone());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(config.poll_timeout_secs);
    let poll_interval = Duration::from_millis(config.poll_interval_ms);

    loop {
        for preview in provider.list_messages(inbox).await? {
            let message = provider.fetch_message(inbox, &preview.id).await?;
            if let Some(code) = extract_verification_code(&message) {
                return Ok(code);
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(EmailProvisioningError::Timeout);
        }

        tokio::time::sleep(poll_interval).await;
    }
}

pub fn warn_registration_not_wired(report: &ProvisioningReport) {
    if report.created.is_empty() {
        return;
    }

    log::warn!(
        "created {} temporary inbox(es), but DeepSeek registration is not wired; credentials were written to {}",
        report.created.len(),
        report.state_path
    );
}

fn provider_for(provider: EmailProviderKind) -> Box<dyn EmailProvider + Send + Sync> {
    match provider {
        EmailProviderKind::MailTm => Box::new(MailTmProvider::new()),
    }
}

#[derive(Clone)]
struct MailTmProvider {
    client: reqwest::Client,
}

impl MailTmProvider {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn get_domain(&self) -> Result<String, EmailProvisioningError> {
        let domains = self
            .client
            .get(format!("{MAIL_TM_API_BASE}/domains"))
            .send()
            .await?
            .error_for_status()?
            .json::<MailTmCollection<MailTmDomain>>()
            .await?;

        domains
            .items
            .into_iter()
            .find(|domain| domain.is_active.unwrap_or(true))
            .map(|domain| domain.domain)
            .ok_or(EmailProvisioningError::NoDomains)
    }

    async fn create_account(
        &self,
        address: &str,
        password: &str,
    ) -> Result<(), EmailProvisioningError> {
        let payload = MailTmCredentials { address, password };

        self.client
            .post(format!("{MAIL_TM_API_BASE}/accounts"))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    async fn create_token(
        &self,
        address: &str,
        password: &str,
    ) -> Result<String, EmailProvisioningError> {
        let payload = MailTmCredentials { address, password };

        let token = self
            .client
            .post(format!("{MAIL_TM_API_BASE}/token"))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .json::<MailTmToken>()
            .await?;

        Ok(token.token)
    }
}

impl EmailProvider for MailTmProvider {
    fn create_inbox<'a>(
        &'a self,
        password_length: usize,
    ) -> Pin<Box<dyn Future<Output = Result<ProvisionedInbox, EmailProvisioningError>> + Send + 'a>>
    {
        Box::pin(async move {
            let domain = self.get_domain().await?;
            let address = format!(
                "{}@{}",
                random_string(RANDOM_ALPHABET, RANDOM_LOCAL_PART_LEN),
                domain
            );
            let mailbox_password = random_string(PASSWORD_ALPHABET, password_length);

            self.create_account(&address, &mailbox_password).await?;
            let provider_token = self.create_token(&address, &mailbox_password).await?;

            Ok(ProvisionedInbox {
                provider: EmailProviderKind::MailTm,
                address,
                mailbox_password,
                provider_token,
                created_at_unix: unix_now(),
            })
        })
    }

    fn list_messages<'a>(
        &'a self,
        inbox: &'a ProvisionedInbox,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<InboxMessage>, EmailProvisioningError>> + Send + 'a>>
    {
        Box::pin(async move {
            let response = self
                .client
                .get(format!("{MAIL_TM_API_BASE}/messages"))
                .bearer_auth(&inbox.provider_token)
                .send()
                .await?
                .error_for_status()?
                .json::<MailTmCollection<MailTmMessagePreview>>()
                .await?;

            Ok(response.items.into_iter().map(Into::into).collect())
        })
    }

    fn fetch_message<'a>(
        &'a self,
        inbox: &'a ProvisionedInbox,
        message_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<InboxMessage, EmailProvisioningError>> + Send + 'a>>
    {
        Box::pin(async move {
            let message = self
                .client
                .get(format!("{MAIL_TM_API_BASE}/messages/{message_id}"))
                .bearer_auth(&inbox.provider_token)
                .send()
                .await?
                .error_for_status()?
                .json::<MailTmMessageDetail>()
                .await?;

            Ok(message.into())
        })
    }
}

#[derive(Debug, Deserialize)]
struct MailTmCollection<T> {
    #[serde(rename = "hydra:member")]
    items: Vec<T>,
}

#[derive(Debug, Deserialize)]
struct MailTmDomain {
    domain: String,
    #[serde(rename = "isActive")]
    is_active: Option<bool>,
}

#[derive(Debug, Serialize)]
struct MailTmCredentials<'a> {
    address: &'a str,
    password: &'a str,
}

#[derive(Debug, Deserialize)]
struct MailTmToken {
    token: String,
}

#[derive(Debug, Deserialize)]
struct MailTmAddress {
    address: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MailTmMessagePreview {
    id: String,
    from: Option<MailTmAddress>,
    subject: Option<String>,
    intro: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MailTmMessageDetail {
    id: String,
    from: Option<MailTmAddress>,
    subject: Option<String>,
    intro: Option<String>,
    text: Option<String>,
    html: Option<Vec<String>>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
}

impl From<MailTmMessagePreview> for InboxMessage {
    fn from(value: MailTmMessagePreview) -> Self {
        Self {
            id: value.id,
            from: value.from.and_then(|from| from.address),
            subject: value.subject,
            intro: value.intro,
            text: None,
            html: None,
            created_at: value.created_at,
        }
    }
}

impl From<MailTmMessageDetail> for InboxMessage {
    fn from(value: MailTmMessageDetail) -> Self {
        Self {
            id: value.id,
            from: value.from.and_then(|from| from.address),
            subject: value.subject,
            intro: value.intro,
            text: value.text,
            html: value.html,
            created_at: value.created_at,
        }
    }
}

fn append_state(path: &str, created: &[ProvisionedInbox]) -> Result<(), EmailProvisioningError> {
    if created.is_empty() {
        return Ok(());
    }

    let state_path = Path::new(path);
    if let Some(parent) = state_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut existing = if state_path.exists() {
        let content = std::fs::read_to_string(state_path)?;
        if content.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str::<Vec<ProvisionedInbox>>(&content)?
        }
    } else {
        Vec::new()
    };

    existing.extend_from_slice(created);
    std::fs::write(state_path, serde_json::to_vec_pretty(&existing)?)?;

    Ok(())
}

fn extract_verification_code(message: &InboxMessage) -> Option<String> {
    let mut haystacks = Vec::new();
    if let Some(subject) = &message.subject {
        haystacks.push(subject.as_str());
    }
    if let Some(intro) = &message.intro {
        haystacks.push(intro.as_str());
    }
    if let Some(text) = &message.text {
        haystacks.push(text.as_str());
    }
    if let Some(html) = &message.html {
        haystacks.extend(html.iter().map(String::as_str));
    }

    haystacks.into_iter().find_map(extract_first_six_digit_code)
}

fn extract_first_six_digit_code(input: &str) -> Option<String> {
    let mut digits = String::new();

    for ch in input.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            if digits.len() == 6 {
                return Some(digits);
            }
        } else {
            digits.clear();
        }
    }

    None
}

fn random_string(alphabet: &[u8], len: usize) -> String {
    let mut rng = rand::rng();
    (0..len)
        .map(|_| {
            let idx = rng.random_range(0..alphabet.len());
            alphabet[idx] as char
        })
        .collect()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_code_from_subject() {
        let message = InboxMessage {
            id: "1".to_string(),
            from: None,
            subject: Some("Your code is 123456".to_string()),
            intro: None,
            text: None,
            html: None,
            created_at: None,
        };

        assert_eq!(
            extract_verification_code(&message),
            Some("123456".to_string())
        );
    }

    #[test]
    fn ignores_short_digit_runs() {
        let message = InboxMessage {
            id: "1".to_string(),
            from: None,
            subject: Some("Use 12345, not enough digits".to_string()),
            intro: None,
            text: None,
            html: None,
            created_at: None,
        };

        assert_eq!(extract_verification_code(&message), None);
    }
}
