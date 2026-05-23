use anyhow::Ok;
use chrono::{DateTime, Utc};
use std::fmt;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceImageId(Uuid);

impl FaceImageId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for FaceImageId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FaceImageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceImageStatus {
    Uploaded,
    Processed,
    Failed,
}

impl FaceImageStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uploaded => "uploaded",
            Self::Processed => "processed",
            Self::Failed => "failed",
        }
    }
}

impl TryFrom<&str> for FaceImageStatus {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "uploaded" => Ok(Self::Uploaded),
            "processed" => Ok(Self::Processed),
            "failed" => Ok(Self::Failed),
            _ => anyhow::bail!("unknown face image status: {value}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaceImage {
    pub id: FaceImageId,
    pub image_key: String,
    pub collection_slug: String,
    pub status: FaceImageStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
