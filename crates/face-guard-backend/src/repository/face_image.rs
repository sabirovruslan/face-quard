use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageKey, FaceImageStatus},
    repository::PgRepository,
};

#[derive(Debug, Clone)]
pub struct NewFaceImage {
    pub id: FaceImageId,
    pub image_key: FaceImageKey,
    pub collection_slug: CollectionSlug,
    pub status: FaceImageStatus,
}

#[async_trait]
pub trait FaceImageRepository: Send + Sync {
    async fn insert_face_image(&self, image: NewFaceImage) -> Result<()>;
    async fn mark_face_image_processed(&self, id: FaceImageId) -> Result<()>;
    async fn mark_face_image_failed(&self, id: FaceImageId) -> Result<()>;
}

#[async_trait]
impl FaceImageRepository for PgRepository {
    async fn insert_face_image(&self, image: NewFaceImage) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO face_images (
                id,
                image_key,
                collection_slug,
                status
            )
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(image.id.as_uuid())
        .bind(image.image_key.as_str())
        .bind(image.collection_slug.as_str())
        .bind(image.status.as_str())
        .execute(&self.db_pool)
        .await
        .with_context(|| {
            format!(
                "failed to insert face image: face_image_id={}, image_key={}",
                image.id,
                image.image_key.as_str()
            )
        })?;
        Ok(())
    }

    async fn mark_face_image_processed(&self, id: FaceImageId) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE face_images
            SET
                status = $2,
                updated_at = now()
            WHERE id = $1
            "#,
        )
        .bind(id.as_uuid())
        .bind(FaceImageStatus::Processed.as_str())
        .execute(&self.db_pool)
        .await
        .with_context(|| format!("failed to mark face image as processed: face_image_id={id}"))?;

        Ok(())
    }

    async fn mark_face_image_failed(&self, id: FaceImageId) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE face_images
            SET
                status = $2,
                updated_at = now()
            WHERE id = $1
            "#,
        )
        .bind(id.as_uuid())
        .bind(FaceImageStatus::Failed.as_str())
        .execute(&self.db_pool)
        .await
        .with_context(|| format!("failed to mark face image as failed: face_image_id={id}"))?;

        Ok(())
    }
}
