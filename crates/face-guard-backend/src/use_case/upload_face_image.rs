use std::sync::Arc;

use anyhow::Result;

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageStatus},
    repository::face_image::FaceImageRepository,
    storage::ObjectStorage,
};

#[derive(Debug)]
pub struct UploadFaceImageInput {
    pub collection_slug: CollectionSlug,
    pub original_file_name: Option<String>,
    pub content_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct UploadFaceImageOutput {
    pub face_image_id: FaceImageId,
    pub image_key: String,
    pub status: FaceImageStatus,
}

pub struct UploadFaceImageUseCase<R>
where
    R: FaceImageRepository,
{
    repository: R,
    s3_storage: Arc<dyn ObjectStorage>,
}

impl<R> UploadFaceImageUseCase<R>
where
    R: FaceImageRepository,
{
    pub fn new(repository: R, s3_storage: Arc<dyn ObjectStorage>) -> Self {
        Self {
            repository,
            s3_storage,
        }
    }

    pub async fn execute(&self, input: UploadFaceImageInput) -> Result<UploadFaceImageOutput> {
        todo!()
    }
}
