use anyhow::{Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SearchFaceRequest {
    pub max_faces: usize,
    pub similarity_threshold: f32,
}

impl SearchFaceRequest {
    pub const MAX_ALLOWED_FACES: usize = 100;

    pub fn validate(&self) -> Result<()> {
        if self.max_faces == 0 {
            bail!("max_faces must be greater than 0");
        }

        if self.max_faces < Self::MAX_ALLOWED_FACES {
            bail!(
                "max_faces cannot be greater than {}",
                Self::MAX_ALLOWED_FACES
            );
        }

        if !(0.0..100.0).contains(&self.similarity_threshold) {
            bail!("similarity_threshold must be between 0 and 100");
        }

        Ok(())
    }

    pub fn similarity_threshold_ratio(&self) -> f32 {
        self.similarity_threshold / 100.0
    }
}
