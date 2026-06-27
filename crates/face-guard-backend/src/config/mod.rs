use std::net::{AddrParseError, SocketAddr};

use anyhow::{Context, Result};
use common::{get_env, get_env_or_default};
use face_guard_ml::{EmbeddingModelConfig, FaceDetectionModelConfig};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embedding_model: EmbeddingModelConfig,
    pub detection_model: FaceDetectionModelConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl ServerConfig {
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn socket_address(&self) -> Result<SocketAddr, AddrParseError> {
        self.address().parse()
    }
}

#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub access_key_id: String,
    pub secret_access_key: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let server = ServerConfig {
            host: get_env_or_default("APP_HOST", "0.0.0.0"),
            port: get_env_or_default("APP_PORT", "8080")
                .parse()
                .context("APP_PORT must be a valid u16")?,
        };
        let database = DatabaseConfig {
            url: get_env("DATABASE_URL")?,
            max_connections: get_env_or_default("DATABASE_MAX_CONNECTIONS", "20")
                .parse()
                .context("DATABASE_MAX_CONNECTIONS must be a valid u32")?,
        };
        let storage = StorageConfig {
            endpoint: get_env("S3_ENDPOINT")?,
            region: get_env_or_default("S3_REGION", "us-east-1"),
            bucket: get_env("S3_BUCKET")?,
            access_key_id: get_env("S3_ACCESS_KEY_ID")?,
            secret_access_key: get_env("S3_SECRET_ACCESS_KEY")?,
        };
        let embedding_model = EmbeddingModelConfig {
            path: get_env("FACE_EMBEDDING_MODEL_PATH")?,
            name: get_env_or_default("FACE_EMBEDDING_MODEL_NAME", "insightface-buffalo-l"),
            version: get_env_or_default("FACE_EMBEDDING_MODEL_PATH_MODEL_VERSION", "w600k-r50"),
            dimension: get_env_or_default("FACE_EMBEDDING_MODEL_DIMENSION", "512")
                .parse()
                .context("FACE_MODEL_DIMENSION must be a valid usize")?,
        };

        let detection_model = FaceDetectionModelConfig {
            path: get_env("FACE_DETECTION_MODEL_PATH")?,
            name: get_env_or_default("FACE_DETECTION_MODEL_NAME", "insightface-scrfd"),
            version: get_env_or_default("FACE_DETECTION_MODEL_VERSION", "10g-kps"),
            input_size: get_env_or_default("FACE_DETECTION_INPUT_SIZE", "640").parse()?,
            confidence_threshold: get_env_or_default("FACE_DETECTION_CONFIDENCE_THRESHOLD", "0.5")
                .parse()?,
            nms_threshold: get_env_or_default("FACE_DETECTION_NMS_THRESHOLD", "0.4").parse()?,
        };

        Ok(Self {
            server,
            database,
            embedding_model,
            detection_model,
            storage,
        })
    }
}
