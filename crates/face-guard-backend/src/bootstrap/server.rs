use std::sync::Arc;

use anyhow::Result;
use sqlx::PgPool;

use crate::{
    config::AppConfig,
    router::create_router,
    storage::{ObjectStorage, s3::S3ObjectStorage},
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub db_pool: Arc<PgPool>,
    pub s3_storage: Arc<dyn ObjectStorage>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, db_pool: Arc<PgPool>) -> Result<Self> {
        let s3_stogare = S3ObjectStorage::new(&config.storage)?;

        Ok({
            Self {
                config,
                db_pool,
                s3_storage: Arc::new(s3_stogare),
            }
        })
    }
}

pub struct AppServer {
    pub state: AppState,
    tcp: tokio::net::TcpListener,
}

impl AppServer {
    pub async fn new(mut config: AppConfig, db_pool: PgPool) -> Result<Self> {
        let tcp = tokio::net::TcpListener::bind(config.server.socket_address()?).await?;
        let address = tcp.local_addr()?;
        tracing::info!("Server initialized at {address}");

        config.server.port = address.port();

        let state = AppState::new(Arc::new(config), Arc::new(db_pool))?;

        Ok(Self { state, tcp })
    }

    pub async fn run(self) -> Result<()> {
        let router = create_router(self.state);
        axum::serve(self.tcp, router).await?;
        Ok(())
    }
}
