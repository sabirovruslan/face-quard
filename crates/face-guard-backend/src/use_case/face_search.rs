use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use face_guard_ml::{FaceDetector, FaceEmbeddingGenerator};

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageKey},
    http::error::AppHttpError,
    repository::face_embedding::{
        FaceEmbeddingRepository, SearchSimilarFacesQuery, SimilarFaceEmbedding,
    },
    storage::ObjectStorage,
    validation::{validate_image_bytes, validate_upload_input},
};

#[derive(Debug)]
pub struct SearchSimilarFaceInput {
    pub collection_slug: CollectionSlug,
    pub image_key: FaceImageKey,
    pub max_faces: usize,
    pub similarity_threshold: f32,
}

#[derive(Debug)]
pub struct SearchSimilarFaceOutput {
    pub collection_slug: CollectionSlug,
    pub matches: Vec<SearchSimilarFaceMatch>,
}

#[derive(Debug)]
pub struct SearchSimilarFaceMatch {
    pub face_image_id: FaceImageId,
    pub image_key: FaceImageKey,
    pub similarity: f32,
}

impl From<SimilarFaceEmbedding> for SearchSimilarFaceMatch {
    fn from(value: SimilarFaceEmbedding) -> Self {
        Self {
            face_image_id: value.face_image_id,
            image_key: value.image_key,
            similarity: value.similarity,
        }
    }
}

pub struct SearchSimilarFaceUseCase<R>
where
    R: FaceEmbeddingRepository,
{
    repository: R,
    s3_storage: Arc<dyn ObjectStorage>,
    face_embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>>,
    face_detector: Arc<Mutex<dyn FaceDetector>>,
}

impl<R> SearchSimilarFaceUseCase<R>
where
    R: FaceEmbeddingRepository,
{
    pub fn new(
        repository: R,
        s3_storage: Arc<dyn ObjectStorage>,
        face_embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>>,
        face_detector: Arc<Mutex<dyn FaceDetector>>,
    ) -> Self {
        Self {
            repository,
            s3_storage,
            face_embedding,
            face_detector,
        }
    }

    pub async fn execute(
        &self,
        input: SearchSimilarFaceInput,
    ) -> Result<SearchSimilarFaceOutput, AppHttpError> {
        let bytes = self
            .s3_storage
            .get_object(input.image_key.as_str())
            .await
            .context("failed to download search image from object storage")?;

        validate_upload_input(&bytes)?;
        validate_image_bytes(&bytes)?;

        let face_crop = {
            let face_detector = self.face_detector.clone();
            let image_bytes = bytes;

            tokio::task::spawn_blocking(move || {
                let mut face_detector = face_detector
                    .lock()
                    .map_err(|_| anyhow!("face detector mutex poisoned"))?;

                face_detector.detect_primary_face(&image_bytes)
            })
            .await
            .context("failed to join face detection task")?
            .context("failed to detect face")?
        };

        let generated_embedding = {
            let embedding_model = self.face_embedding.clone();
            let face_bytes = face_crop.into_bytes();

            tokio::task::spawn_blocking(move || {
                let mut embedding_model = embedding_model
                    .lock()
                    .map_err(|_| anyhow!("face embedding model mutex poisoned"))?;

                embedding_model.generate_embedding(&face_bytes)
            })
            .await
            .context("failed to join face embedding task")?
            .context("failed to generate face embedding")?
        };

        let matches = self
            .repository
            .search_similar_faces(SearchSimilarFacesQuery {
                collection_slug: input.collection_slug.clone(),
                embedding: generated_embedding.vector.into_values(),
                model_name: generated_embedding.model.name,
                model_version: generated_embedding.model.version,
                model_dimension: generated_embedding.model.dimension,
                max_faces: input.max_faces,
                similarity_threshold: input.similarity_threshold,
            })
            .await
            .context("failed to search similar faces")?;

        Ok(SearchSimilarFaceOutput {
            collection_slug: input.collection_slug,
            matches: matches
                .into_iter()
                .map(SearchSimilarFaceMatch::from)
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Cursor, sync::Mutex};

    use async_trait::async_trait;
    use face_guard_ml::{
        EmbeddingModel, EmbeddingVector, FaceCrop, FaceDetector, GeneratedFaceEmbedding,
    };
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    use crate::{domain::FaceEmbeddingId, repository::face_embedding::NewFaceEmbedding};

    const TEST_IMAGE_KEY: &str = "faces/search.png";

    #[derive(Debug, Clone)]
    struct StoredObject {
        key: String,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Default)]
    struct FakeStorageState {
        objects: Vec<StoredObject>,
        requested_keys: Vec<String>,
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
                    requested_keys: Vec::new(),
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
                    requested_keys: Vec::new(),
                    get_object_error: None,
                }),
            }
        }

        fn requested_keys(&self) -> Vec<String> {
            self.state.lock().unwrap().requested_keys.clone()
        }
    }

    #[async_trait]
    impl ObjectStorage for FakeObjectStorage {
        async fn put_object(&self, _key: &str, _content_type: &str, _bytes: Vec<u8>) -> Result<()> {
            Ok(())
        }

        async fn get_object(&self, key: &str) -> Result<Vec<u8>> {
            let mut state = self.state.lock().unwrap();
            state.requested_keys.push(key.to_string());

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

        async fn presigned_get_url(
            &self,
            _key: &str,
            _expires_in: std::time::Duration,
        ) -> Result<String> {
            unimplemented!("face search use case does not generate presigned URLs")
        }
    }

    #[derive(Debug, Default)]
    struct FakeRepositoryState {
        query: Option<SearchSimilarFacesQuery>,
        matches: Vec<SimilarFaceEmbedding>,
        search_error: Option<String>,
    }

    #[derive(Debug, Clone, Default)]
    struct FakeRepository {
        state: Arc<Mutex<FakeRepositoryState>>,
    }

    impl FakeRepository {
        fn with_matches(matches: Vec<SimilarFaceEmbedding>) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeRepositoryState {
                    query: None,
                    matches,
                    search_error: None,
                })),
            }
        }

        fn failing(error: impl Into<String>) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeRepositoryState {
                    query: None,
                    matches: Vec::new(),
                    search_error: Some(error.into()),
                })),
            }
        }

        fn query(&self) -> Option<SearchSimilarFacesQuery> {
            self.state.lock().unwrap().query.clone()
        }
    }

    #[async_trait]
    impl FaceEmbeddingRepository for FakeRepository {
        async fn insert_embedding(&self, _embedding: NewFaceEmbedding) -> Result<()> {
            Ok(())
        }

        async fn search_similar_faces(
            &self,
            query: SearchSimilarFacesQuery,
        ) -> Result<Vec<SimilarFaceEmbedding>> {
            let mut state = self.state.lock().unwrap();
            state.query = Some(query);

            if let Some(error) = &state.search_error {
                anyhow::bail!(error.clone());
            }

            Ok(state.matches.clone())
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
                    vector: EmbeddingVector::new(vec![0.25; 512], 512).unwrap(),
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

    #[derive(Debug)]
    enum FakeDetectorMode {
        Success(Vec<u8>),
        Error(String),
    }

    #[derive(Debug)]
    struct FakeFaceDetector {
        mode: FakeDetectorMode,
    }

    impl FakeFaceDetector {
        fn success(face_bytes: Vec<u8>) -> Self {
            Self {
                mode: FakeDetectorMode::Success(face_bytes),
            }
        }

        fn failing(error: impl Into<String>) -> Self {
            Self {
                mode: FakeDetectorMode::Error(error.into()),
            }
        }
    }

    impl FaceDetector for FakeFaceDetector {
        fn detect_primary_face(&mut self, _image_bytes: &[u8]) -> Result<FaceCrop> {
            match &self.mode {
                FakeDetectorMode::Success(face_bytes) => Ok(FaceCrop::new(face_bytes.clone())),
                FakeDetectorMode::Error(error) => anyhow::bail!(error.clone()),
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

    fn search_input() -> SearchSimilarFaceInput {
        SearchSimilarFaceInput {
            collection_slug: CollectionSlug::new("test_collection").unwrap(),
            image_key: FaceImageKey::from_existing(TEST_IMAGE_KEY).unwrap(),
            max_faces: 3,
            similarity_threshold: 0.8,
        }
    }

    fn similar_face(image_key: &str, similarity: f32) -> SimilarFaceEmbedding {
        SimilarFaceEmbedding {
            id: FaceEmbeddingId::new(),
            face_image_id: FaceImageId::new(),
            image_key: FaceImageKey::from_existing(image_key).unwrap(),
            similarity,
        }
    }

    fn build_use_case(
        repository: FakeRepository,
        storage: Arc<FakeObjectStorage>,
        embedding: FakeEmbeddingGenerator,
        detector: FakeFaceDetector,
    ) -> SearchSimilarFaceUseCase<FakeRepository> {
        let storage: Arc<dyn ObjectStorage> = storage;
        let embedding: Arc<Mutex<dyn FaceEmbeddingGenerator>> = Arc::new(Mutex::new(embedding));
        let detector: Arc<Mutex<dyn FaceDetector>> = Arc::new(Mutex::new(detector));

        SearchSimilarFaceUseCase::new(repository, storage, embedding, detector)
    }

    #[tokio::test]
    async fn execute_downloads_image_searches_similar_faces_and_maps_matches() {
        let matches = vec![
            similar_face("faces/match-1.png", 0.93),
            similar_face("faces/match-2.png", 0.87),
        ];
        let repository = FakeRepository::with_matches(matches.clone());
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            valid_png_bytes(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage.clone(),
            FakeEmbeddingGenerator::success(),
            FakeFaceDetector::success(valid_png_bytes()),
        );

        let output = match use_case.execute(search_input()).await {
            Ok(output) => output,
            Err(_) => panic!("search similar face should succeed"),
        };

        assert_eq!(output.collection_slug.as_str(), "test_collection");
        assert_eq!(output.matches.len(), 2);
        assert_eq!(output.matches[0].face_image_id, matches[0].face_image_id);
        assert_eq!(output.matches[0].image_key.as_str(), "faces/match-1.png");
        assert_eq!(output.matches[0].similarity, 0.93);
        assert_eq!(storage.requested_keys(), vec![TEST_IMAGE_KEY.to_string()]);

        let query = repository.query().unwrap();
        assert_eq!(query.collection_slug.as_str(), "test_collection");
        assert_eq!(query.embedding, vec![0.25; 512]);
        assert_eq!(query.model_name, "test-model");
        assert_eq!(query.model_version, "test-version");
        assert_eq!(query.model_dimension, 512);
        assert_eq!(query.max_faces, 3);
        assert_eq!(query.similarity_threshold, 0.8);
    }

    #[tokio::test]
    async fn execute_rejects_invalid_image_before_search() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            b"not an image".to_vec(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage,
            FakeEmbeddingGenerator::success(),
            FakeFaceDetector::success(valid_png_bytes()),
        );

        let result = use_case.execute(search_input()).await;

        assert!(result.is_err());
        assert!(repository.query().is_none());
    }

    #[tokio::test]
    async fn execute_returns_storage_error_before_search() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::failing("storage is unavailable"));
        let use_case = build_use_case(
            repository.clone(),
            storage,
            FakeEmbeddingGenerator::success(),
            FakeFaceDetector::success(valid_png_bytes()),
        );

        let result = use_case.execute(search_input()).await;

        assert!(result.is_err());
        assert!(repository.query().is_none());
    }

    #[tokio::test]
    async fn execute_returns_embedding_error_before_search() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            valid_png_bytes(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage,
            FakeEmbeddingGenerator::failing("embedding service failed"),
            FakeFaceDetector::success(valid_png_bytes()),
        );

        let result = use_case.execute(search_input()).await;

        assert!(result.is_err());
        assert!(repository.query().is_none());
    }

    #[tokio::test]
    async fn execute_returns_face_detection_error_before_search() {
        let repository = FakeRepository::default();
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            valid_png_bytes(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage,
            FakeEmbeddingGenerator::success(),
            FakeFaceDetector::failing("no face detected"),
        );

        let result = use_case.execute(search_input()).await;

        assert!(result.is_err());
        assert!(repository.query().is_none());
    }

    #[tokio::test]
    async fn execute_returns_repository_search_error() {
        let repository = FakeRepository::failing("database is unavailable");
        let storage = Arc::new(FakeObjectStorage::with_object(
            TEST_IMAGE_KEY,
            valid_png_bytes(),
        ));
        let use_case = build_use_case(
            repository.clone(),
            storage,
            FakeEmbeddingGenerator::success(),
            FakeFaceDetector::success(valid_png_bytes()),
        );

        let result = use_case.execute(search_input()).await;

        assert!(result.is_err());
        assert!(repository.query().is_some());
    }
}
