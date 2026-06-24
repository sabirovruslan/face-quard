use anyhow::Result;
use axum::{
    Json,
    extract::{Multipart, State},
};

use crate::{
    bootstrap::server::AppState,
    domain::CollectionSlug,
    http::{
        dto::face_search::{request::SearchFaceRequest, response::SearchFaceResponse},
        error::AppHttpError,
    },
    repository::PgRepository,
    use_case::face_search::{SearchSimilarFaceInput, SearchSimilarFaceUseCase},
};

pub async fn search_similar_face(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<SearchFaceResponse>, AppHttpError> {
    let mut collection_slug: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut max_faces: Option<usize> = None;
    let mut similarity_threshold: Option<f32> = None;

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "file" => {
                file_bytes = Some(field.bytes().await?.to_vec());
            }
            "max_faces" => {
                let value = field.text().await?;
                max_faces = Some(value.parse()?);
            }
            "similarity_threshold" => {
                let value = field.text().await?;
                similarity_threshold = Some(value.parse()?);
            }
            "collection_slug" => {
                collection_slug = Some(field.text().await?);
            }
            _ => {}
        }
    }

    let collection_slug = CollectionSlug::new(collection_slug.unwrap_or_default())?;
    let bytes = file_bytes.ok_or_else(|| anyhow::anyhow!("file field is required"))?;
    let request = SearchFaceRequest {
        max_faces: max_faces.unwrap_or(10),
        similarity_threshold: similarity_threshold.unwrap_or(80.0),
    };

    let input = SearchSimilarFaceInput {
        collection_slug,
        bytes,
        max_faces: request.max_faces,
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
