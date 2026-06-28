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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Cursor, sync::Mutex};

    use async_trait::async_trait;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    #[derive(Debug, Clone)]
    struct StoredObject {
        key: String,
        content_type: String,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Default)]
    struct FakeObjectStorage {
        objects: Mutex<Vec<StoredObject>>,
        put_object_error: Option<String>,
    }

    impl FakeObjectStorage {
        fn failing(error: impl Into<String>) -> Self {
            Self {
                objects: Mutex::new(Vec::new()),
                put_object_error: Some(error.into()),
            }
        }

        fn objects(&self) -> Vec<StoredObject> {
            self.objects.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ObjectStorage for FakeObjectStorage {
        async fn put_object(&self, key: &str, content_type: &str, bytes: Vec<u8>) -> Result<()> {
            if let Some(error) = &self.put_object_error {
                anyhow::bail!(error.clone());
            }

            self.objects.lock().unwrap().push(StoredObject {
                key: key.to_string(),
                content_type: content_type.to_string(),
                bytes,
            });

            Ok(())
        }

        async fn get_object(&self, _key: &str) -> Result<Vec<u8>> {
            unimplemented!("upload object use case does not read objects")
        }
    }

    fn valid_png_bytes() -> Vec<u8> {
        let image = ImageBuffer::from_pixel(64, 64, Rgba([10, 20, 30, 255]));
        let mut bytes = Cursor::new(Vec::new());

        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();

        bytes.into_inner()
    }

    fn input(bytes: Vec<u8>) -> UploadObjectInput {
        UploadObjectInput {
            image_key: FaceImageKey::from_existing("faces/person.png").unwrap(),
            bytes,
        }
    }

    #[tokio::test]
    async fn execute_uploads_valid_image_to_storage() {
        let storage = Arc::new(FakeObjectStorage::default());
        let use_case = UploadObjectUseCase::new(storage.clone());
        let bytes = valid_png_bytes();

        let output = use_case.execute(input(bytes.clone())).await.unwrap();

        assert_eq!(output.image_key.as_str(), "faces/person.png");

        let objects = storage.objects();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].key, "faces/person.png");
        assert_eq!(objects[0].content_type, "image/png");
        assert_eq!(objects[0].bytes, bytes);
    }

    #[tokio::test]
    async fn execute_rejects_empty_bytes_before_storage_write() {
        let storage = Arc::new(FakeObjectStorage::default());
        let use_case = UploadObjectUseCase::new(storage.clone());

        let error = use_case.execute(input(Vec::new())).await.unwrap_err();

        assert_eq!(error.to_string(), "image file cannot be empty");
        assert!(storage.objects().is_empty());
    }

    #[tokio::test]
    async fn execute_rejects_invalid_image_before_storage_write() {
        let storage = Arc::new(FakeObjectStorage::default());
        let use_case = UploadObjectUseCase::new(storage.clone());

        let error = use_case
            .execute(input(b"not an image".to_vec()))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "failed to detect image format");
        assert!(storage.objects().is_empty());
    }

    #[tokio::test]
    async fn execute_returns_storage_error() {
        let storage = Arc::new(FakeObjectStorage::failing("storage is unavailable"));
        let use_case = UploadObjectUseCase::new(storage);

        let error = use_case
            .execute(input(valid_png_bytes()))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "failed to upload object to storage");
    }
}
