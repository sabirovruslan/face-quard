use anyhow::{Context, Error, Result};
use async_trait::async_trait;
use uuid::Uuid;

use crate::{
    domain::{CollectionSlug, FaceEmbeddingId, FaceImageId, FaceImageKey},
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

#[derive(Debug, Clone)]
pub struct SimilarFaceEmbedding {
    pub id: FaceEmbeddingId,
    pub face_image_id: FaceImageId,
    pub image_key: FaceImageKey,
    pub similarity: f32,
}

type SimilarFaceEmbeddingRow = (Uuid, Uuid, String, f32);

impl TryFrom<SimilarFaceEmbeddingRow> for SimilarFaceEmbedding {
    type Error = Error;

    fn try_from(value: SimilarFaceEmbeddingRow) -> std::result::Result<Self, Self::Error> {
        let (id, face_image_id, image_key, similarity) = value;

        Ok(Self {
            id: FaceEmbeddingId::from_uuid(id),
            face_image_id: FaceImageId::from_uuid(face_image_id),
            image_key: FaceImageKey::from_existing(image_key)?,
            similarity,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SearchSimilarFacesQuery {
    pub collection_slug: CollectionSlug,
    pub embedding: Vec<f32>,
    pub model_name: String,
    pub model_version: String,
    pub model_dimension: usize,
    pub max_faces: usize,
    pub similarity_threshold: f32,
}

#[async_trait]
pub trait FaceEmbeddingRepository {
    async fn insert_embedding(&self, embedding: NewFaceEmbedding) -> Result<()>;
    async fn search_similar_faces(
        &self,
        query: SearchSimilarFacesQuery,
    ) -> Result<Vec<SimilarFaceEmbedding>>;
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

    async fn search_similar_faces(
        &self,
        query: SearchSimilarFacesQuery,
    ) -> Result<Vec<SimilarFaceEmbedding>> {
        let rows = sqlx::query_as::<_, SimilarFaceEmbeddingRow>(
            r#"
                SELECT
                    fe.id AS face_embedding_id,
                    fi.id AS face_image_id,
                    fi.image_key,
                    (1.0 - (fe.embedding <=> $1::real[]::vector))::real AS similarity
                FROM face_embeddings fe
                JOIN face_images fi ON fi.id = fe.face_image_id
                WHERE
                    fi.collection_slug = $2
                    AND fe.model_name = $3
                    AND fe.model_version = $4
                    AND fe.model_dimension = $5
                    AND (1.0 - (fe.embedding <=> $1::real[]::vector)) >= $6
                ORDER BY fe.embedding <=> $1::real[]::vector
                LIMIT $7
            "#,
        )
        .bind(query.embedding)
        .bind(query.collection_slug.as_str())
        .bind(query.model_name.as_str())
        .bind(query.model_version.as_str())
        .bind(query.model_dimension as i32)
        .bind(query.similarity_threshold as f64)
        .bind(query.max_faces as i64)
        .fetch_all(&self.db_pool)
        .await
        .context("failed to search similar face embeddings")?;

        rows.into_iter()
            .map(SimilarFaceEmbedding::try_from)
            .collect()
    }
}
