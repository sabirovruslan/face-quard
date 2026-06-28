use anyhow::{Context, Result};

use crate::{
    domain::{CollectionSlug, FaceImageStatus},
    repository::face_image::{
        FaceImageReadRepository, ListFaceImagesCursor, ListFaceImagesOutput, ListFaceImagesQuery,
    },
};

#[derive(Debug)]
pub struct ListFaceImagesInput {
    pub collection_slug: Option<CollectionSlug>,
    pub status: Option<FaceImageStatus>,
    pub limit: usize,
    pub cursor: Option<ListFaceImagesCursor>,
}

pub struct ListFaceImagesUseCase<R>
where
    R: FaceImageReadRepository,
{
    repository: R,
}

impl<R> ListFaceImagesUseCase<R>
where
    R: FaceImageReadRepository,
{
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, input: ListFaceImagesInput) -> Result<ListFaceImagesOutput> {
        let items = self
            .repository
            .search(ListFaceImagesQuery {
                collection_slug: input.collection_slug,
                status: input.status,
                limit: input.limit,
                cursor: input.cursor.map(|value| ListFaceImagesCursor {
                    created_at: value.created_at,
                    id: value.id,
                }),
            })
            .await
            .context("failed to get list of face images")?;

        Ok(items)
    }
}
