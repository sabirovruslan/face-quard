use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::{
    domain::{FaceEmbeddingId, FaceImageId},
    repository::PgRepository,
};

#[derive(Debug, Clone)]
pub struct NewFaceEmbedding {
    pub id: FaceEmbeddingId,
    pub face_image_id: FaceImageId,
    pub embedding: Vec<f32>,
    pub model_name: String,
    pub model_version: String,
    pub model_dimension: usize,
}

#[async_trait]
pub trait FaceEmbeddingRepository {
    async fn insert_embedding(&self, embedding: NewFaceEmbedding) -> Result<()>;
}

#[async_trait]
impl FaceEmbeddingRepository for PgRepository {
    async fn insert_embedding(&self, embedding: NewFaceEmbedding) -> Result<()> {
        sqlx::query(
            r#"
                INSERT INTO face_embeddings (
                    id,
                    face_image_id,
                    embedding,
                    model_name,
                    model_version,
                    model_dimension
                )
                VALUES ($1, $2, $3::real[]::vector, $4, $5, $6)
            "#,
        )
        .bind(embedding.id.as_uuid())
        .bind(embedding.face_image_id.as_uuid())
        .bind(embedding.embedding)
        .bind(embedding.model_name.as_str())
        .bind(embedding.model_version.as_str())
        .bind(embedding.model_dimension as i32)
        .execute(&self.db_pool)
        .await
        .with_context(|| {
            format!(
                "failed to insert face embedding: face_embedding_id={}, face_image_id={}",
                embedding.id, embedding.face_image_id
            )
        })?;

        Ok(())
    }
}
