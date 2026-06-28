use anyhow::Result;
use axum::{
    Json,
    extract::{Multipart, State},
};

use crate::{
    bootstrap::server::AppState,
    domain::FaceImageKey,
    http::{dto::response::UploadObjectResponse, error::AppHttpError},
    use_case::upload_object::{UploadObjectInput, UploadObjectUseCase},
};

pub async fn upload_object(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadObjectResponse>, AppHttpError> {
    let mut image_key: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "image_key" => {
                image_key = Some(field.text().await?);
            }
            "file" => {
                file_bytes = Some(field.bytes().await?.to_vec());
            }
            _ => {}
        }
    }

    let image_key = image_key
        .ok_or_else(|| anyhow::anyhow!("image_key field is required"))
        .and_then(FaceImageKey::from_existing)?;

    let input = UploadObjectInput {
        image_key,
        bytes: file_bytes.ok_or_else(|| anyhow::anyhow!("file field is required"))?,
    };

    let use_case = UploadObjectUseCase::new(state.s3_storage);

    let output = use_case.execute(input).await?;

    Ok(Json(UploadObjectResponse::from(output)))
}
