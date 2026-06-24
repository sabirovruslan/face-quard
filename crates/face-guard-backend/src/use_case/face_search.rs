use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use face_guard_ml::FaceEmbeddingGenerator;

use crate::{
    domain::{CollectionSlug, FaceImageId, FaceImageKey},
    http::error::AppHttpError,
    repository::face_embedding::FaceEmbeddingRepository,
    storage::ObjectStorage,
    use_case::image_validation::{validate_image_bytes, validate_upload_input},
};

#[derive(Debug)]
pub struct SearchSimilarFaceInput {
    pub collection_slug: CollectionSlug,
    pub bytes: Vec<u8>,
    pub max_faces: usize,
    pub similarity_threshold: f32,
}

#[derive(Debug)]
pub struct SearchSimilarFaceOutput {
    pub collection_slug: CollectionSlug,
    pub matches: Vec<SearchSimilarFaceOutput>,
}

#[derive(Debug)]
pub struct SearchSimilarFaceMatch {
    pub face_image_id: FaceImageId,
    pub image_key: FaceImageKey,
    pub similarity: f32,
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
        validate_upload_input(&input.bytes)?;
        validate_image_bytes(&input.bytes)?;

        let embedding_model = self.face_embedding.clone();
        let image_bytes = input.bytes;

        let generated_embedding = tokio::task::spawn_blocking(move || {
            let mut embedding_model = embedding_model
                .lock()
                .map_err(|_| anyhow!("face embedding model mutex poisoned"))?;

            embedding_model.generate_embedding(&image_bytes)
        })
        .await
        .context("failed to join face embedding task")?
        .context("failed to generate face embedding")?;

        todo!()
    }
}
