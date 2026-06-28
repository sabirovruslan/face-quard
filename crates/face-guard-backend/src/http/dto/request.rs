use anyhow::{Result, bail};
use serde::Deserialize;

use crate::domain::{CollectionSlug, FaceImageKey};

#[derive(Debug, Deserialize)]
pub struct SearchSimilarFaceRequest {
    pub image_key: String,
    pub max_faces: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub collection_slug: Option<String>,
}

impl SearchSimilarFaceRequest {
    pub const MAX_ALLOWED_FACES: usize = 100;

    pub fn image_key(&self) -> Result<FaceImageKey> {
        FaceImageKey::from_existing(&self.image_key)
    }

    pub fn max_faces(&self) -> usize {
        self.max_faces.unwrap_or(10)
    }

    pub fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold.unwrap_or(80.0)
    }

    pub fn similarity_threshold_ratio(&self) -> f32 {
        self.similarity_threshold() / 100.0
    }

    pub fn collection_slug(&self) -> Result<CollectionSlug> {
        match &self.collection_slug {
            Some(value) => CollectionSlug::new(value),
            None => Ok(CollectionSlug::default_collection()),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let max_faces = self.max_faces();
        if max_faces == 0 {
            bail!("max_faces must be greater than 0");
        }

        if max_faces > Self::MAX_ALLOWED_FACES {
            bail!(
                "max_faces cannot be greater than {}",
                Self::MAX_ALLOWED_FACES
            );
        }

        if !(0.0..100.0).contains(&self.similarity_threshold()) {
            bail!("similarity_threshold must be between 0 and 100");
        }

        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateFaceImageRequest {
    pub image_key: String,
    pub collection_slug: Option<String>,
}

impl CreateFaceImageRequest {
    pub fn image_key(&self) -> Result<FaceImageKey> {
        FaceImageKey::from_existing(&self.image_key)
    }

    pub fn collection_slug(&self) -> Result<CollectionSlug> {
        match &self.collection_slug {
            Some(value) => CollectionSlug::new(value),
            None => Ok(CollectionSlug::default_collection()),
        }
    }
}
