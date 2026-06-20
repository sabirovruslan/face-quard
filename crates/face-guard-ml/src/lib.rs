use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct ModelsConfig {
    pub face_embedding_model_path: String,
    pub face_model_name: String,
    pub face_model_version: String,
    pub face_model_dimension: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    pub name: String,
    pub version: String,
    pub dimension: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingVector {
    values: Vec<f32>,
}

impl EmbeddingVector {
    pub fn new(values: Vec<f32>, expected_dimension: usize) -> Result<Self> {
        if values.is_empty() {
            bail!("embedding vector cannot be empty");
        }

        if values.len() != expected_dimension {
            bail!(
                "invalid embedding dimension: expected {}, got {}",
                expected_dimension,
                values.len()
            );
        }

        if values.iter().any(|value| !value.is_finite()) {
            bail!("embedding vector contains NaN or infinity");
        }

        Ok(Self { values })
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedFaceEmbedding {
    pub vector: EmbeddingVector,
    pub model: EmbeddingModel,
}

#[derive(Debug, Clone)]
pub struct FaceEmbeddingService {
    model: EmbeddingModel,
}

impl FaceEmbeddingService {
    pub fn new(model_config: &ModelsConfig) -> Result<Self> {
        let model = EmbeddingModel {
            name: model_config.face_model_name.clone(),
            version: model_config.face_model_version.clone(),
            dimension: model_config.face_model_dimension,
        };

        if model.dimension == 0 {
            bail!("face model dimension must be greater than 0");
        }

        if model_config.face_embedding_model_path.trim().is_empty() {
            bail!("face embedding model path cannot be empty");
        }

        Ok(Self { model })
    }

    pub fn generate_embedding(&self) -> Result<GeneratedFaceEmbedding> {
        _ = self.model.name.len();
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_values_with_expected_dimension() {
        let values = vec![0.1, 0.2, 0.3];

        let vector = EmbeddingVector::new(values.clone(), 3).unwrap();

        assert_eq!(vector.dimension(), 3);
        assert_eq!(vector.values(), values.as_slice());
        assert_eq!(vector.into_values(), values);
    }

    #[test]
    fn new_rejects_empty_values() {
        let error = EmbeddingVector::new(Vec::new(), 0).unwrap_err();

        assert_eq!(error.to_string(), "embedding vector cannot be empty");
    }

    #[test]
    fn new_rejects_invalid_dimension() {
        let error = EmbeddingVector::new(vec![0.1, 0.2], 3).unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid embedding dimension: expected 3, got 2"
        );
    }

    #[test]
    fn new_rejects_non_finite_values() {
        for invalid_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = EmbeddingVector::new(vec![0.1, invalid_value, 0.3], 3).unwrap_err();

            assert_eq!(
                error.to_string(),
                "embedding vector contains NaN or infinity"
            );
        }
    }
}
