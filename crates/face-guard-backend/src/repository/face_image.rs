use anyhow::{Context, Error, Result, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

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

pub struct ListFaceImagesQuery {
    pub collection_slug: Option<CollectionSlug>,
    pub status: Option<FaceImageStatus>,
    pub limit: usize,
    pub cursor: Option<ListFaceImagesCursor>,
}

#[derive(Debug)]
pub struct ListFaceImagesCursor {
    pub created_at: DateTime<Utc>,
    pub id: FaceImageId,
}

impl From<&FaceImageItem> for ListFaceImagesCursor {
    fn from(value: &FaceImageItem) -> Self {
        Self {
            created_at: value.created_at,
            id: value.id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaceImageItem {
    pub id: FaceImageId,
    pub image_key: FaceImageKey,
    pub collection_slug: CollectionSlug,
    pub status: FaceImageStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TryFrom<FaceImageItemRow> for FaceImageItem {
    type Error = Error;

    fn try_from(value: FaceImageItemRow) -> std::result::Result<Self, Self::Error> {
        let (face_image_id, image_key, collection_slug, status, created_at, updated_at) = value;
        Ok(Self {
            id: FaceImageId::from_uuid(face_image_id),
            image_key: FaceImageKey::from_existing(image_key)?,
            collection_slug: CollectionSlug::try_from(collection_slug)?,
            status: FaceImageStatus::try_from(status.as_str())?,
            created_at,
            updated_at,
        })
    }
}

type FaceImageItemRow = (Uuid, String, String, String, DateTime<Utc>, DateTime<Utc>);

#[derive(Debug)]
pub struct ListFaceImagesPage {
    pub items: Vec<FaceImageItem>,
    pub next_cursor: Option<ListFaceImagesCursor>,
    pub has_more: bool,
}

#[async_trait]
pub trait FaceImageReadRepository: Send + Sync {
    async fn search(&self, query: ListFaceImagesQuery) -> Result<ListFaceImagesPage>;
}

#[async_trait]
impl FaceImageReadRepository for PgRepository {
    async fn search(&self, query: ListFaceImagesQuery) -> Result<ListFaceImagesPage> {
        if query.limit == 0 {
            bail!("limit must be greater than 0");
        }

        let fetch_limit = query.limit + 1;

        let collection_slug = query.collection_slug.as_ref().map(|value| value.as_str());
        let status = query.status.as_ref().map(|value| value.as_str());
        let cursor_created_at = query.cursor.as_ref().map(|value| value.created_at);
        let cursor_id = query.cursor.as_ref().map(|value| value.id.as_uuid());

        let rows = sqlx::query_as::<_, FaceImageItemRow>(
            r#"
              SELECT
                  id,
                  image_key,
                  collection_slug,
                  status,
                  created_at,
                  updated_at
              FROM face_images
              WHERE
                  ($1::text IS NULL OR collection_slug = $1)
                  AND ($2::text IS NULL OR status = $2)
                  AND (
                    ($3::timestamptz IS NULL AND $4::uuid IS NULL)
                    OR (created_at, id) < ($3, $4)
                  )
              ORDER BY created_at DESC, id DESC
              LIMIT $5
              "#,
        )
        .bind(collection_slug)
        .bind(status)
        .bind(cursor_created_at)
        .bind(cursor_id)
        .bind(fetch_limit as i64)
        .fetch_all(&self.db_pool)
        .await?;

        let has_more = rows.len() > query.limit;

        let searched_face_images = rows
            .into_iter()
            .take(query.limit)
            .map(FaceImageItem::try_from)
            .collect::<Result<Vec<FaceImageItem>>>()?;

        let next_cursor = if has_more {
            searched_face_images.last().map(ListFaceImagesCursor::from)
        } else {
            None
        };

        Ok(ListFaceImagesPage {
            items: searched_face_images,
            next_cursor,
            has_more,
        })
    }
}
