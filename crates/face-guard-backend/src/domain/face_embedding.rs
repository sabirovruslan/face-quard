use core::fmt;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::FaceImageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceEmbeddingId(Uuid);

impl FaceEmbeddingId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for FaceEmbeddingId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FaceEmbeddingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct FaceEmbedding {
    pub id: FaceEmbeddingId,
    pub face_image_id: FaceImageId,
    pub values: Vec<u8>,
    pub model_name: String,
    pub model_version: String,
    pub model_dimension: usize,
    pub created_at: DateTime<Utc>,
}
