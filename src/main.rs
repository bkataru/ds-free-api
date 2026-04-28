//! ai-free-api binary entrypoint — starts the HTTP server

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::new().default_filter_or("info")).init();

    let config = ds_free_api::Config::load_with_args(std::env::args())?;
    ds_free_api::server::run(config).await
}
