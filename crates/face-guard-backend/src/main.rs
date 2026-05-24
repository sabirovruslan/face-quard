use anyhow::Result;

use crate::bootstrap::server::AppServer;

pub mod bootstrap;
pub mod config;
pub mod domain;
pub mod router;
pub mod storage;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    bootstrap::tracing::init();
    tracing::info!("Tracing initialized");

    let config = bootstrap::constant::CONFIG.clone();

    let db_pool = bootstrap::db::create_pool(&config.database).await?;
    bootstrap::db::migrations(&db_pool).await?;

    let server = AppServer::new(config, db_pool).await?;

    server.run().await?;

    Ok(())
}
