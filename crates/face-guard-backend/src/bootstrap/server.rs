use std::sync::Arc;

use anyhow::Result;
use sqlx::PgPool;

use crate::{config::AppConfig, router::create_router};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub db_pool: Arc<PgPool>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, db_pool: Arc<PgPool>) -> Self {
        Self { config, db_pool }
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

        let state = AppState::new(Arc::new(config), Arc::new(db_pool));

        Ok(Self { state, tcp })
    }

    pub async fn run(self) -> Result<()> {
        let router = create_router(self.state);
        axum::serve(self.tcp, router).await?;
        Ok(())
    }
}
