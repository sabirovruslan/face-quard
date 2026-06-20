use anyhow::Result;
use serde::Serialize;

use crate::use_case::upload_face_image::UploadFaceImageOutput;

#[derive(Debug, Serialize)]
pub struct UploadFaceImageResponse {
    pub id: String,
    pub status: String,
}

impl TryFrom<UploadFaceImageOutput> for UploadFaceImageResponse {
    type Error = anyhow::Error;

    fn try_from(value: UploadFaceImageOutput) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.face_image_id.to_string(),
            status: value.status.as_str().to_string(),
        })
    }
}
