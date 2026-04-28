//! ai-free-api binary entrypoint — starts the HTTP server

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::new().default_filter_or("info")).init();

    let config = ds_free_api::Config::load_with_args(std::env::args())?;
    if config.account_provisioning.enabled {
        let report = ds_free_api::account_provisioning::prepare_inboxes(&config.account_provisioning).await?;
        ds_free_api::account_provisioning::warn_registration_not_wired(&report);
    }
    ds_free_api::server::run(config).await
}
