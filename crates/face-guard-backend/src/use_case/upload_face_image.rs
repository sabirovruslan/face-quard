use std::sync::Arc;

use anyhow::{Context, Result};

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageStatus},
    repository::face_image::FaceImageRepository,
    storage::ObjectStorage,
    use_case::{
        image_util::{extract_image_format, image_content_type, image_extension},
        image_validation::{validate_image_bytes, validate_upload_input},
    },
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
        validate_upload_input(&input.bytes).context("invalid upload input")?;
        validate_image_bytes(&input.bytes).context("invalid image")?;
        let format = extract_image_format(&input.bytes)?;
        let extention = image_extension(format);
        let detected_content_type = image_content_type(format);

        let face_image_id = FaceImageId::new();

        todo!()
    }
}
