use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use face_guard_ml::FaceEmbeddingGenerator;

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageKey},
    http::error::AppHttpError,
    repository::face_embedding::{FaceEmbeddingRepository, SimilarFaceEmbedding},
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
}

impl<R> SearchSimilarFaceUseCase<R>
where
    R: FaceEmbeddingRepository,
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

        let embedding_model = self.face_embedding.clone();
        let image_bytes = bytes;

        let generated_embedding = tokio::task::spawn_blocking(move || {
            let mut embedding_model = embedding_model
                .lock()
                .map_err(|_| anyhow!("face embedding model mutex poisoned"))?;

            embedding_model.generate_embedding(&image_bytes)
        })
        .await
        .context("failed to join face embedding task")?
        .context("failed to generate face embedding")?;

        let matches = self
            .repository
            .search_similar_faces(
                &input.collection_slug,
                generated_embedding.vector.into_values(),
                &generated_embedding.model.name,
                &generated_embedding.model.version,
                generated_embedding.model.dimension,
                input.max_faces,
                input.similarity_threshold,
            )
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
