use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use face_guard_ml::FaceEmbedding;

use crate::{
    domain::{CollectionSlug, FaceEmbeddingId, FaceImageId, FaceImageKey, FaceImageStatus},
    repository::{
        face_embedding::{FaceEmbeddingRepository, NewFaceEmbedding},
        face_image::{FaceImageRepository, NewFaceImage},
    },
    storage::ObjectStorage,
    use_case::{
        image_util::{extract_image_format, image_extension},
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
    pub image_key: FaceImageKey,
    pub status: FaceImageStatus,
}

pub struct UploadFaceImageUseCase<R>
where
    R: FaceImageRepository + FaceEmbeddingRepository,
{
    repository: R,
    s3_storage: Arc<dyn ObjectStorage>,
    face_embedding: Arc<Mutex<FaceEmbedding>>,
}

impl<R> UploadFaceImageUseCase<R>
where
    R: FaceImageRepository + FaceEmbeddingRepository,
{
    pub fn new(
        repository: R,
        s3_storage: Arc<dyn ObjectStorage>,
        face_embedding: Arc<Mutex<FaceEmbedding>>,
    ) -> Self {
        Self {
            repository,
            s3_storage,
            face_embedding,
        }
    }

    pub async fn execute(&self, input: UploadFaceImageInput) -> Result<UploadFaceImageOutput> {
        validate_upload_input(&input.bytes).context("invalid upload input")?;
        validate_image_bytes(&input.bytes).context("invalid image")?;
        let format = extract_image_format(&input.bytes)?;
        let extention = image_extension(format);

        let face_image_id = FaceImageId::new();
        let face_image_key = FaceImageKey::new(extention);

        self.s3_storage
            .put_object(
                face_image_key.as_str(),
                &input.content_type,
                input.bytes.clone(),
            )
            .await
            .context("failed to upload face image to object storage")?;

        let new_face_image = NewFaceImage {
            id: face_image_id,
            image_key: face_image_key.clone(),
            collection_slug: input.collection_slug,
            status: FaceImageStatus::Processing,
        };
        self.repository
            .insert_face_image(new_face_image)
            .await
            .context("failed to insert face image")?;

        let embedding_model = self.face_embedding.clone();
        let image_bytes = input.bytes;
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
                values: generated_embedding.vector.into_values(),
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

        Ok(UploadFaceImageOutput {
            face_image_id,
            image_key: face_image_key,
            status: FaceImageStatus::Processed,
        })
    }
}
