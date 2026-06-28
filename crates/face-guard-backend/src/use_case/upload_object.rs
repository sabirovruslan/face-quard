use std::sync::Arc;

use anyhow::{Context, Result};

use crate::{
    domain::FaceImageKey,
    storage::ObjectStorage,
    util::image_content_type,
    validation::{validate_image_bytes, validate_image_format, validate_upload_input},
};

#[derive(Debug)]
pub struct UploadObjectInput {
    pub image_key: FaceImageKey,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct UploadOgjectOutput {
    pub image_key: FaceImageKey,
}

pub struct UploadObjectUseCase {
    s3_storage: Arc<dyn ObjectStorage>,
}

impl UploadObjectUseCase {
    pub fn new(s3_storage: Arc<dyn ObjectStorage>) -> Self {
        Self { s3_storage }
    }

    pub async fn execute(&self, input: UploadObjectInput) -> Result<UploadOgjectOutput> {
        validate_upload_input(&input.bytes)?;

        let image_format =
            image::guess_format(&input.bytes).context("failed to detect image format")?;

        validate_image_format(image_format)?;
        validate_image_bytes(&input.bytes)?;

        let content_type = image_content_type(image_format);

        self.s3_storage
            .put_object(input.image_key.as_str(), content_type, input.bytes)
            .await
            .context("failed to upload object to storage")?;

        Ok(UploadOgjectOutput {
            image_key: input.image_key,
        })
    }
}
