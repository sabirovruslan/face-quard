use anyhow::{Context, Result};
use sqlx::{PgPool, postgres::PgPoolOptions};

use crate::config::DatabaseConfig;

pub async fn create_pool(db_config: &DatabaseConfig) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(db_config.max_connections)
        .connect(&db_config.url)
        .await
        .context("failed to connect to PostgreSQL")?;

    Ok(pool)
}

pub async fn migrations(pool: &PgPool) -> Result<()> {
    sqlx::migrate!("../../migrations")
        .run(pool)
        .await
        .context("failed to run database migrations")?;

    Ok(())
}
