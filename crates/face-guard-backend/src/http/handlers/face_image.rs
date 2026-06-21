use anyhow::Result;
use axum::{
    Json,
    extract::{Multipart, State},
};

use crate::{
    bootstrap::server::AppState,
    domain::CollectionSlug,
    http::{dto::face_images::response::UploadFaceImageResponse, error::AppHttpError},
    repository::PgRepository,
    use_case::upload_face_image::{UploadFaceImageInput, UploadFaceImageUseCase},
};

pub async fn upload_face_image(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadFaceImageResponse>, AppHttpError> {
    let mut collection_slug: Option<String> = None;
    let mut file_name: Option<String> = None;
    let mut content_type: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "collection_slug" => {
                collection_slug = Some(field.text().await?);
            }
            "file" => {
                file_name = field.file_name().map(ToString::to_string);
                content_type = field.content_type().map(ToString::to_string);
                file_bytes = Some(field.bytes().await?.to_vec());
            }
            _ => {}
        }
    }

    let collection_slug = CollectionSlug::new(collection_slug.unwrap_or_default())?;
    let bytes = file_bytes.ok_or_else(|| anyhow::anyhow!("file field is required"))?;
    let content_type = content_type.unwrap_or_else(|| "application/octet-stream".to_string());

    let input = UploadFaceImageInput {
        collection_slug,
        original_file_name: file_name,
        content_type,
        bytes,
    };

    let use_case = UploadFaceImageUseCase::new(
        PgRepository::new(state.db_pool.clone()),
        state.s3_storage.clone(),
        state.face_embedding.clone(),
    );

    let output = use_case.execute(input).await?;
    let response = UploadFaceImageResponse::try_from(output)?;

    Ok(Json(response))
}
