use std::sync::{Arc, Mutex};

use anyhow::Result;
use face_guard_ml::{FaceDetector, FaceEmbedding, FaceEmbeddingGenerator, ScrfdFaceDetector};
use sqlx::PgPool;

use crate::{
    config::AppConfig,
    router::create_router,
    storage::{ObjectStorage, s3::S3ObjectStorage},
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub db_pool: PgPool,
    pub s3_storage: Arc<dyn ObjectStorage>,
    pub face_embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>>,
    pub face_detector: Arc<Mutex<dyn FaceDetector>>,
}

impl AppState {
    pub fn new(config: Arc<AppConfig>, db_pool: PgPool) -> Result<Self> {
        let s3_stogare = S3ObjectStorage::new(&config.storage)?;
        let face_embedding = FaceEmbedding::new(&config.embedding_model)?;
        let face_detector = ScrfdFaceDetector::new(&config.detection_model)?;

        Ok({
            Self {
                config,
                db_pool,
                s3_storage: Arc::new(s3_stogare),
                face_embedding: Arc::new(Mutex::new(face_embedding)),
                face_detector: Arc::new(Mutex::new(face_detector)),
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

        let state = AppState::new(Arc::new(config), db_pool)?;

        Ok(Self { state, tcp })
    }

    pub async fn run(self) -> Result<()> {
        let router = create_router(self.state);
        axum::serve(self.tcp, router).await?;
        Ok(())
    }
}
