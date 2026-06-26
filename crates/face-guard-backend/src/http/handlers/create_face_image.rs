use anyhow::Result;
use axum::{Json, extract::State};

use crate::{
    bootstrap::server::AppState,
    http::{
        dto::{request::CreateFaceImageRequest, response::UploadFaceImageResponse},
        error::AppHttpError,
    },
    repository::PgRepository,
    use_case::create_face_image::{CreateFaceImageInput, CreateFaceImageUseCase},
};

pub async fn create_face_image(
    State(state): State<AppState>,
    Json(request): Json<CreateFaceImageRequest>,
) -> Result<Json<UploadFaceImageResponse>, AppHttpError> {
    let input = CreateFaceImageInput {
        collection_slug: request.collection_slug()?,
        image_key: request.image_key()?,
    };

    let use_case = CreateFaceImageUseCase::new(
        PgRepository::new(state.db_pool.clone()),
        state.s3_storage.clone(),
        state.face_embedding.clone(),
    );

    let output = use_case.execute(input).await?;
    let response = UploadFaceImageResponse::try_from(output)?;

    Ok(Json(response))
}
