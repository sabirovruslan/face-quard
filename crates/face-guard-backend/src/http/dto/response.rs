use serde::Serialize;

use crate::use_case::create_face_image::CreateFaceImageOutput;
use crate::use_case::face_search::{SearchSimilarFaceMatch, SearchSimilarFaceOutput};
use crate::use_case::upload_object::UploadOgjectOutput;

#[derive(Debug, Serialize)]
pub struct SearchFaceMatchResponse {
    pub id: String,
    pub image_key: String,
    pub similarity: f32,
}

#[derive(Debug, Serialize)]
pub struct SearchFaceResponse {
    pub collection_slug: String,
    pub matches: Vec<SearchFaceMatchResponse>,
}

impl From<SearchSimilarFaceMatch> for SearchFaceMatchResponse {
    fn from(value: SearchSimilarFaceMatch) -> Self {
        Self {
            id: value.face_image_id.to_string(),
            image_key: value.image_key.as_str().to_string(),
            similarity: value.similarity,
        }
    }
}

impl From<SearchSimilarFaceOutput> for SearchFaceResponse {
    fn from(value: SearchSimilarFaceOutput) -> Self {
        Self {
            collection_slug: value.collection_slug.to_string(),
            matches: value
                .matches
                .into_iter()
                .map(SearchFaceMatchResponse::from)
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CreateFaceImageResponse {
    pub id: String,
    pub status: String,
}

impl TryFrom<CreateFaceImageOutput> for CreateFaceImageResponse {
    type Error = anyhow::Error;

    fn try_from(value: CreateFaceImageOutput) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.face_image_id.to_string(),
            status: value.status.as_str().to_string(),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct UploadObjectResponse {
    pub image_key: String,
}

impl From<UploadOgjectOutput> for UploadObjectResponse {
    fn from(value: UploadOgjectOutput) -> Self {
        Self {
            image_key: value.image_key.as_str().to_string(),
        }
    }
}
