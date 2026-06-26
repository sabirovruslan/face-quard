use anyhow::Result;
use axum::{Json, extract::State};

use crate::{
    bootstrap::server::AppState,
    http::{
        dto::{request::SearchFaceRequest, response::SearchFaceResponse},
        error::AppHttpError,
    },
    repository::PgRepository,
    use_case::face_search::{SearchSimilarFaceInput, SearchSimilarFaceUseCase},
};

pub async fn search_similar_face(
    State(state): State<AppState>,
    Json(request): Json<SearchFaceRequest>,
) -> Result<Json<SearchFaceResponse>, AppHttpError> {
    request.validate()?;

    let input = SearchSimilarFaceInput {
        collection_slug: request.collection_slug()?,
        image_key: request.image_key()?,
        max_faces: request.max_faces(),
        similarity_threshold: request.similarity_threshold_ratio(),
    };

    let use_case = SearchSimilarFaceUseCase::new(
        PgRepository::new(state.db_pool),
        state.s3_storage.clone(),
        state.face_embedding.clone(),
    );

    let output = use_case.execute(input).await?;
    let response = SearchFaceResponse::from(output);

    Ok(Json(response))
}
