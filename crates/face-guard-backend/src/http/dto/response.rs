use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::repository::face_image::ListFaceImagesCursor;
use crate::use_case::create_face_image::CreateFaceImageOutput;
use crate::use_case::face_search::{SearchSimilarFaceMatch, SearchSimilarFaceOutput};
use crate::use_case::list_face_image::{ListFaceImagesOutput, ListedFaceImageItem};
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

#[derive(Debug, Serialize)]
pub struct ListFaceImagesCursorResponse {
    pub created_at: DateTime<Utc>,
    pub id: String,
}

impl From<ListFaceImagesCursor> for ListFaceImagesCursorResponse {
    fn from(value: ListFaceImagesCursor) -> Self {
        Self {
            created_at: value.created_at,
            id: value.id.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FaceImageResponse {
    pub id: String,
    pub image_key: String,
    pub downlod_url: String,
    pub collection_slug: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<ListedFaceImageItem> for FaceImageResponse {
    fn from(value: ListedFaceImageItem) -> Self {
        Self {
            id: value.item.id.to_string(),
            image_key: value.item.image_key.as_str().to_string(),
            downlod_url: value.downlod_url.to_string(),
            collection_slug: value.item.collection_slug.to_string(),
            status: value.item.status.as_str().to_string(),
            created_at: value.item.created_at,
            updated_at: value.item.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ListFaceImagesResponse {
    pub items: Vec<FaceImageResponse>,
    pub next_cursor: Option<ListFaceImagesCursorResponse>,
    pub has_more: bool,
}

impl From<ListFaceImagesOutput> for ListFaceImagesResponse {
    fn from(value: ListFaceImagesOutput) -> Self {
        Self {
            items: value
                .items
                .into_iter()
                .map(FaceImageResponse::from)
                .collect(),
            next_cursor: value.next_cursor.map(ListFaceImagesCursorResponse::from),
            has_more: value.has_more,
        }
    }
}
