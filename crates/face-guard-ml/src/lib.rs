use crate::image::{normalize_l2, preprocess_face_image};
use anyhow::{Context, Result, bail};
use ort::{inputs, session::Session, value::TensorRef};

pub mod image;

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

#[derive(Debug)]
pub struct FaceEmbeddingService {
    model: EmbeddingModel,
    session: Session,
    input_name: String,
    output_name: String,
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

        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .with_intra_threads(2)
            .map_err(|error| -> ort::Error { error.into() })
            .context("failed to configure ONNX Runtime intra threads")?
            .commit_from_file(&model_config.face_embedding_model_path)
            .with_context(|| {
                format!(
                    "failed to load face embedding ONNX model from {}",
                    model_config.face_embedding_model_path
                )
            })?;

        let input_name = session
            .inputs()
            .first()
            .context("ONNX model has no inputs")?
            .name()
            .to_string();

        let output_name = session
            .outputs()
            .first()
            .context("ONNX model has no outputs")?
            .name()
            .to_string();

        Ok(Self {
            model,
            session,
            input_name,
            output_name,
        })
    }

    pub fn generate_embedding(&mut self, image_bytes: &[u8]) -> Result<GeneratedFaceEmbedding> {
        if image_bytes.is_empty() {
            bail!("image bytes cannot be empty");
        }

        let input_tensor =
            preprocess_face_image(image_bytes).context("failed to preprocess face image")?;
        let input_tensor = TensorRef::from_array_view(&input_tensor)
            .context("failed to create ONNX input tensor")?;

        let mut values = {
            let outputs = self
                .session
                .run(inputs![self.input_name.as_str() => input_tensor])
                .context("failed to run face embedding ONNX model")?;

            let output = outputs.get(self.output_name.as_str()).with_context(|| {
                format!(
                    "failed to get ONNX output tensor by name '{}'",
                    self.output_name
                )
            })?;

            let (_, output_values) = output
                .try_extract_tensor::<f32>()
                .context("failed to extract embedding tensor as f32")?;

            output_values.to_vec()
        };

        normalize_l2(&mut values).context("failed to normalize embedding vector")?;

        self.validate_embedding_values(&values)?;

        let embedding_vector = EmbeddingVector::new(values, self.model.dimension)
            .context("failed to create embedding_vector")?;

        Ok(GeneratedFaceEmbedding {
            vector: embedding_vector,
            model: self.model.clone(),
        })
    }

    fn validate_embedding_values(&self, values: &[f32]) -> Result<()> {
        if values.len() != self.model.dimension {
            bail!(
                "invalid embedding dimension: expected {}, got {}",
                self.model.dimension,
                values.len()
            );
        }

        if values.iter().any(|value| !value.is_finite()) {
            bail!("embedding contains NaN or infinity");
        }

        Ok(())
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
