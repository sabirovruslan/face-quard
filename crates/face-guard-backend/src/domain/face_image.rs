use anyhow::{Ok, Result, bail};
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FaceImageKey(String);

impl FaceImageKey {
    pub fn new(extension: &str) -> Self {
        let image_key = format!("{}.{}", Uuid::new_v4(), extension);
        Self(image_key)
    }

    pub fn from_existing(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let value = value.trim().to_string();

        if value.is_empty() {
            bail!("image key cannot be empty");
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceImageStatus {
    Uploaded,
    Processing,
    Processed,
    Failed,
}

impl FaceImageStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uploaded => "uploaded",
            Self::Processing => "processing",
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
            "processing" => Ok(Self::Processing),
            "processed" => Ok(Self::Processed),
            "failed" => Ok(Self::Failed),
            _ => anyhow::bail!("unknown face image status: {value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectionSlug(String);

impl CollectionSlug {
    const MAX_LEN: usize = 128;

    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let value = value.trim().to_string();

        Self::validate(&value)?;

        Ok(Self(value))
    }

    pub fn default_collection() -> Self {
        Self("default".to_string())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn into_string(self) -> String {
        self.0
    }

    fn validate(value: &str) -> Result<()> {
        if value.is_empty() {
            anyhow::bail!("collection slug cannot be empty");
        };

        if value.len() > Self::MAX_LEN {
            anyhow::bail!(
                "collection slug cannot be longer than {} characters",
                Self::MAX_LEN
            );
        };

        let is_valid = value
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '-' || c == '_');

        if !is_valid {
            anyhow::bail!(
                "collection slug can contain only ascii alphabetic letters, digits, '-' and '_'"
            );
        }

        Ok(())
    }
}

impl Default for CollectionSlug {
    fn default() -> Self {
        Self::default_collection()
    }
}

impl fmt::Display for CollectionSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<String> for CollectionSlug {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for CollectionSlug {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Debug, Clone)]
pub struct FaceImage {
    pub id: FaceImageId,
    pub image_key: String,
    pub collection_slug: CollectionSlug,
    pub status: FaceImageStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
