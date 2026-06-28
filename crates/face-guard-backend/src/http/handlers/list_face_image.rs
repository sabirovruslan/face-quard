use axum::{Json, extract::State};

use crate::{
    bootstrap::server::AppState,
    domain::FaceImageId,
    http::{
        dto::{request::ListFaceImagesRequest, response::ListFaceImagesResponse},
        error::AppHttpError,
    },
    repository::{PgRepository, face_image::ListFaceImagesCursor},
    use_case::list_face_image::{ListFaceImagesInput, ListFaceImagesUseCase},
};

pub async fn list_face_images(
    State(state): State<AppState>,
    Json(request): Json<ListFaceImagesRequest>,
) -> Result<Json<ListFaceImagesResponse>, AppHttpError> {
    request.validate()?;

    let input = ListFaceImagesInput {
        collection_slug: request.collection_slug()?,
        status: request.status()?,
        limit: request.limit(),
        cursor: request.cursor.map(|value| ListFaceImagesCursor {
            created_at: value.created_at,
            id: FaceImageId::from_uuid(value.id),
        }),
    };

    let use_case = ListFaceImagesUseCase::new(PgRepository::new(state.db_pool));
    let output = use_case.execute(input).await?;
    let response = ListFaceImagesResponse::from(output);

    Ok(Json(response))
}
