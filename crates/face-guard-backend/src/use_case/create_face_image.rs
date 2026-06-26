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

    const TEST_IMAGE_KEY: &str = "faces/test.png";

    #[derive(Debug, Clone)]
    struct StoredObject {
        key: String,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Default)]
    struct FakeStorageState {
        objects: Vec<StoredObject>,
        get_object_error: Option<String>,
    }

    #[derive(Debug, Default)]
    struct FakeObjectStorage {
        state: Mutex<FakeStorageState>,
    }

    impl FakeObjectStorage {
        fn failing(error: impl Into<String>) -> Self {
            Self {
                state: Mutex::new(FakeStorageState {
                    objects: Vec::new(),
                    get_object_error: Some(error.into()),
                }),
            }
        }

        fn with_object(key: impl Into<String>, bytes: Vec<u8>) -> Self {
            Self {
                state: Mutex::new(FakeStorageState {
                    objects: vec![StoredObject {
                        key: key.into(),
                        bytes,
                    }],
                    get_object_error: None,
                }),
            }
        }
    }

    #[async_trait]
    impl ObjectStorage for FakeObjectStorage {
        async fn put_object(&self, _key: &str, _content_type: &str, _bytes: Vec<u8>) -> Result<()> {
            Ok(())
        }

        async fn get_object(&self, key: &str) -> Result<Vec<u8>> {
            let state = self.state.lock().unwrap();

            if let Some(error) = &state.get_object_error {
                anyhow::bail!(error.clone());
            }

            state
                .objects
                .iter()
                .find(|object| object.key == key)
                .map(|object| object.bytes.clone())
                .ok_or_else(|| anyhow::anyhow!("object not found: {key}"))
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

        async fn search_similar_faces(
            &self,
            _query: crate::repository::face_embedding::SearchSimilarFacesQuery,
        ) -> Result<Vec<crate::repository::face_embedding::SimilarFaceEmbedding>> {
            Ok(Vec::new())
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

    fn create_input() -> CreateFaceImageInput {
        CreateFaceImageInput {
            collection_slug: CollectionSlug::new("test_collection").unwrap(),
            image_key: FaceImageKey::from_existing(TEST_IMAGE_KEY).unwrap(),
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
    async fn execute_downloads_image_generates_embedding_and_marks_image_processed() {
        let repository = FakeRepository::default();
        let input_bytes = valid_png_bytes();
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            input_bytes.clone(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
        );

        let output = use_case.execute(create_input()).await.unwrap();

        assert_eq!(output.status, FaceImageStatus::Processed);
        assert_eq!(output.image_key.as_str(), TEST_IMAGE_KEY);

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
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            b"not an image".to_vec(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
        );

        let error = use_case.execute(create_input()).await.unwrap_err();

        assert_eq!(error.to_string(), "invalid image");

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

        let error = use_case.execute(create_input()).await.unwrap_err();

        assert_eq!(
            error.to_string(),
            "failed to download face image from object storage"
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
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            valid_png_bytes(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::failing("embedding service failed"),
        );

        let error = use_case.execute(create_input()).await.unwrap_err();

        assert_eq!(error.to_string(), "failed to generate face embedding");

        let state = repository.state();
        assert_eq!(state.face_images.len(), 1);
        assert_eq!(state.failed_face_image_ids, vec![state.face_images[0].id]);
        assert!(state.processed_face_image_ids.is_empty());
        assert!(state.embeddings.is_empty());
    }
}
