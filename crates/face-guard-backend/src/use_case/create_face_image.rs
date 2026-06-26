use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use face_guard_ml::FaceEmbeddingGenerator;

use crate::{
    domain::{CollectionSlug, FaceEmbeddingId, FaceImageId, FaceImageKey, FaceImageStatus},
    repository::{
        face_embedding::{FaceEmbeddingRepository, NewFaceEmbedding},
        face_image::{FaceImageRepository, NewFaceImage},
    },
    storage::ObjectStorage,
    validation::{validate_image_bytes, validate_upload_input},
};

#[derive(Debug)]
pub struct CreateFaceImageInput {
    pub collection_slug: CollectionSlug,
    pub image_key: FaceImageKey,
}

#[derive(Debug)]
pub struct CreateFaceImageOutput {
    pub face_image_id: FaceImageId,
    pub image_key: FaceImageKey,
    pub status: FaceImageStatus,
}

pub struct CreateFaceImageUseCase<R>
where
    R: FaceImageRepository + FaceEmbeddingRepository,
{
    repository: R,
    s3_storage: Arc<dyn ObjectStorage>,
    face_embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>>,
}

impl<R> CreateFaceImageUseCase<R>
where
    R: FaceImageRepository + FaceEmbeddingRepository,
{
    pub fn new(
        repository: R,
        s3_storage: Arc<dyn ObjectStorage>,
        face_embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>>,
    ) -> Self {
        Self {
            repository,
            s3_storage,
            face_embedding,
        }
    }

    pub async fn execute(&self, input: CreateFaceImageInput) -> Result<CreateFaceImageOutput> {
        let bytes = self
            .s3_storage
            .get_object(input.image_key.as_str())
            .await
            .context("failed to download face image from object storage")?;

        validate_upload_input(&bytes).context("invalid upload input")?;
        validate_image_bytes(&bytes).context("invalid image")?;

        let face_image_id = FaceImageId::new();

        let new_face_image = NewFaceImage {
            id: face_image_id,
            image_key: input.image_key.clone(),
            collection_slug: input.collection_slug.clone(),
            status: FaceImageStatus::Processing,
        };
        self.repository
            .insert_face_image(new_face_image)
            .await
            .context("failed to insert face image")?;

        let embedding_model = self.face_embedding.clone();
        let image_bytes = bytes;
        let generated_embedding = match tokio::task::spawn_blocking(move || {
            let mut embedding_model = embedding_model
                .lock()
                .map_err(|_| anyhow!("face embedding model mutex poisoned"))?;

            embedding_model.generate_embedding(&image_bytes)
        })
        .await
        {
            Ok(Ok(embedding)) => embedding,

            Ok(Err(err)) => {
                self.repository
                    .mark_face_image_failed(face_image_id)
                    .await
                    .context("failed to mark face image as failed")?;

                return Err(err).context("failed to generate face embedding");
            }

            Err(err) => {
                self.repository
                    .mark_face_image_failed(face_image_id)
                    .await
                    .context("failed to mark face image as failed")?;

                return Err(err).context("failed to join face embedding task");
            }
        };

        let face_embedding_id = FaceEmbeddingId::new();
        self.repository
            .insert_embedding(NewFaceEmbedding{
                id: face_embedding_id,
                face_image_id,
                embedding: generated_embedding.vector.into_values(),
                model_name: generated_embedding.model.name,
                model_version: generated_embedding.model.version,
                model_dimension: generated_embedding.model.dimension
            })
            .await
            .with_context(|| {
                format!(
                    "failed to insert face embedding and mark image as processed: face_image_id={face_image_id}, face_embedding_id={face_embedding_id}"
                )
            })?;

        self.repository
            .mark_face_image_processed(face_image_id)
            .await
            .context("failed to mark face image as processed")?;

        Ok(CreateFaceImageOutput {
            face_image_id,
            image_key: input.image_key,
            status: FaceImageStatus::Processed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Cursor, sync::Mutex};

    use async_trait::async_trait;
    use face_guard_ml::{EmbeddingModel, EmbeddingVector, GeneratedFaceEmbedding};
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    #[derive(Debug, Clone)]
    struct UploadedObject {
        key: String,
        content_type: String,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Default)]
    struct FakeStorageState {
        uploaded_objects: Vec<UploadedObject>,
        put_object_error: Option<String>,
    }

    #[derive(Debug, Default)]
    struct FakeObjectStorage {
        state: Mutex<FakeStorageState>,
    }

    impl FakeObjectStorage {
        fn failing(error: impl Into<String>) -> Self {
            Self {
                state: Mutex::new(FakeStorageState {
                    uploaded_objects: Vec::new(),
                    put_object_error: Some(error.into()),
                }),
            }
        }

        fn uploaded_objects(&self) -> Vec<UploadedObject> {
            self.state.lock().unwrap().uploaded_objects.clone()
        }
    }

    #[async_trait]
    impl ObjectStorage for FakeObjectStorage {
        async fn put_object(&self, key: &str, content_type: &str, bytes: Vec<u8>) -> Result<()> {
            let mut state = self.state.lock().unwrap();

            if let Some(error) = &state.put_object_error {
                anyhow::bail!(error.clone());
            }

            state.uploaded_objects.push(UploadedObject {
                key: key.to_string(),
                content_type: content_type.to_string(),
                bytes,
            });

            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct FakeRepositoryState {
        face_images: Vec<NewFaceImage>,
        processed_face_image_ids: Vec<FaceImageId>,
        failed_face_image_ids: Vec<FaceImageId>,
        embeddings: Vec<NewFaceEmbedding>,
    }

    #[derive(Debug, Clone, Default)]
    struct FakeRepository {
        state: Arc<Mutex<FakeRepositoryState>>,
    }

    impl FakeRepository {
        fn state(&self) -> std::sync::MutexGuard<'_, FakeRepositoryState> {
            self.state.lock().unwrap()
        }
    }

    #[async_trait]
    impl FaceImageRepository for FakeRepository {
        async fn insert_face_image(&self, image: NewFaceImage) -> Result<()> {
            self.state.lock().unwrap().face_images.push(image);
            Ok(())
        }

        async fn mark_face_image_processed(&self, id: FaceImageId) -> Result<()> {
            self.state.lock().unwrap().processed_face_image_ids.push(id);
            Ok(())
        }

        async fn mark_face_image_failed(&self, id: FaceImageId) -> Result<()> {
            self.state.lock().unwrap().failed_face_image_ids.push(id);
            Ok(())
        }
    }

    #[async_trait]
    impl FaceEmbeddingRepository for FakeRepository {
        async fn insert_embedding(&self, embedding: NewFaceEmbedding) -> Result<()> {
            self.state.lock().unwrap().embeddings.push(embedding);
            Ok(())
        }
    }

    #[derive(Debug)]
    enum FakeEmbeddingMode {
        Success,
        Error(String),
    }

    #[derive(Debug)]
    struct FakeEmbeddingGenerator {
        mode: FakeEmbeddingMode,
        calls: usize,
    }

    impl FakeEmbeddingGenerator {
        fn success() -> Self {
            Self {
                mode: FakeEmbeddingMode::Success,
                calls: 0,
            }
        }

        fn failing(error: impl Into<String>) -> Self {
            Self {
                mode: FakeEmbeddingMode::Error(error.into()),
                calls: 0,
            }
        }
    }

    impl FaceEmbeddingGenerator for FakeEmbeddingGenerator {
        fn generate_embedding(&mut self, _image_bytes: &[u8]) -> Result<GeneratedFaceEmbedding> {
            self.calls += 1;

            match &self.mode {
                FakeEmbeddingMode::Success => Ok(GeneratedFaceEmbedding {
                    vector: EmbeddingVector::new(vec![0.1; 512], 512).unwrap(),
                    model: EmbeddingModel {
                        name: "test-model".to_string(),
                        version: "test-version".to_string(),
                        dimension: 512,
                    },
                }),
                FakeEmbeddingMode::Error(error) => anyhow::bail!(error.clone()),
            }
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

    fn upload_input(bytes: Vec<u8>) -> CreateFaceImageInput {
        CreateFaceImageInput {
            collection_slug: CollectionSlug::new("test_collection").unwrap(),
            original_file_name: Some("face.png".to_string()),
            content_type: "image/png".to_string(),
            bytes,
        }
    }

    fn build_use_case(
        repository: FakeRepository,
        storage: Arc<FakeObjectStorage>,
        embedding: FakeEmbeddingGenerator,
    ) -> CreateFaceImageUseCase<FakeRepository> {
        let storage: Arc<dyn ObjectStorage> = storage;
        let embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>> = Arc::new(Mutex::new(embedding));

        CreateFaceImageUseCase::new(repository, storage, embedding)
    }

    #[tokio::test]
    async fn execute_uploads_image_generates_embedding_and_marks_image_processed() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::default());
        let input_bytes = valid_png_bytes();
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
        );

        let output = use_case
            .execute(upload_input(input_bytes.clone()))
            .await
            .unwrap();

        assert_eq!(output.status, FaceImageStatus::Processed);

        let uploaded_objects = storage.uploaded_objects();
        assert_eq!(uploaded_objects.len(), 1);
        assert_eq!(uploaded_objects[0].key, output.image_key.as_str());
        assert_eq!(uploaded_objects[0].content_type, "image/png");
        assert_eq!(uploaded_objects[0].bytes, input_bytes);

        let state = repository.state();
        assert_eq!(state.face_images.len(), 1);
        assert_eq!(state.face_images[0].id, output.face_image_id);
        assert_eq!(
            state.face_images[0].image_key.as_str(),
            output.image_key.as_str()
        );
        assert_eq!(state.face_images[0].status, FaceImageStatus::Processing);

        assert_eq!(state.embeddings.len(), 1);
        assert_eq!(state.embeddings[0].face_image_id, output.face_image_id);
        assert_eq!(state.embeddings[0].embedding.len(), 512);
        assert_eq!(state.embeddings[0].model_name, "test-model");
        assert_eq!(state.embeddings[0].model_version, "test-version");
        assert_eq!(state.embeddings[0].model_dimension, 512);

        assert_eq!(state.processed_face_image_ids, vec![output.face_image_id]);
        assert!(state.failed_face_image_ids.is_empty());
    }

    #[tokio::test]
    async fn execute_rejects_invalid_image_before_side_effects() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::default());
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
        );

        let error = use_case
            .execute(upload_input(b"not an image".to_vec()))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "invalid image");
        assert!(storage.uploaded_objects().is_empty());

        let state = repository.state();
        assert!(state.face_images.is_empty());
        assert!(state.embeddings.is_empty());
        assert!(state.processed_face_image_ids.is_empty());
        assert!(state.failed_face_image_ids.is_empty());
    }

    #[tokio::test]
    async fn execute_returns_storage_error_before_repository_writes() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::failing("storage is unavailable"));
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
        );

        let error = use_case
            .execute(upload_input(valid_png_bytes()))
            .await
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "failed to upload face image to object storage"
        );

        let state = repository.state();
        assert!(state.face_images.is_empty());
        assert!(state.embeddings.is_empty());
        assert!(state.processed_face_image_ids.is_empty());
        assert!(state.failed_face_image_ids.is_empty());
    }

    #[tokio::test]
    async fn execute_marks_image_failed_when_embedding_generation_fails() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::default());
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::failing("embedding service failed"),
        );

        let error = use_case
            .execute(upload_input(valid_png_bytes()))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "failed to generate face embedding");
        assert_eq!(storage.uploaded_objects().len(), 1);

        let state = repository.state();
        assert_eq!(state.face_images.len(), 1);
        assert_eq!(state.failed_face_image_ids, vec![state.face_images[0].id]);
        assert!(state.processed_face_image_ids.is_empty());
        assert!(state.embeddings.is_empty());
    }
}
