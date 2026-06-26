use crate::image::{normalize_l2, preprocess_face_image};
use anyhow::{Context, Result, bail};
use ort::{inputs, session::Session, value::TensorRef};

#[derive(Debug, Clone)]
pub struct EmbeddingModelsConfig {
    pub path: String,
    pub name: String,
    pub version: String,
    pub dimension: usize,
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

pub trait FaceEmbeddingGenerator: Send {
    fn generate_embedding(&mut self, image_bytes: &[u8]) -> Result<GeneratedFaceEmbedding>;
}

#[derive(Debug)]
pub struct FaceEmbedding {
    model: EmbeddingModel,
    session: Session,
    input_name: String,
    output_name: String,
}

impl FaceEmbedding {
    pub fn new(model_config: &EmbeddingModelsConfig) -> Result<Self> {
        let model = EmbeddingModel {
            name: model_config.name.clone(),
            version: model_config.version.clone(),
            dimension: model_config.dimension,
        };

        if model.dimension == 0 {
            bail!("face model dimension must be greater than 0");
        }

        if model_config.path.trim().is_empty() {
            bail!("face embedding model path cannot be empty");
        }

        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .with_intra_threads(2)
            .map_err(|error| -> ort::Error { error.into() })
            .context("failed to configure ONNX Runtime intra threads")?
            .commit_from_file(&model_config.path)
            .with_context(|| {
                format!(
                    "failed to load face embedding ONNX model from {}",
                    model_config.path
                )
            })?;

        Self::from_session(model, session)
    }

    fn from_session(model: EmbeddingModel, session: Session) -> Result<Self> {
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

impl FaceEmbeddingGenerator for FaceEmbedding {
    fn generate_embedding(&mut self, image_bytes: &[u8]) -> Result<GeneratedFaceEmbedding> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use ::image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
    use ort::{
        editor::{Graph, Model, Node, ONNX_DOMAIN, Opset},
        session::builder::SessionBuilder,
        value::{Outlet, Shape, SymbolicDimensions, TensorElementType, ValueType},
    };

    const FACE_EMBEDDING_DIMENSION: usize = 3 * 112 * 112;
    const EPSILON: f32 = 1e-4;

    fn model_config(path: impl Into<String>) -> EmbeddingModelsConfig {
        EmbeddingModelsConfig {
            path: path.into(),
            name: "test-face-model".to_string(),
            version: "test-version".to_string(),
            dimension: FACE_EMBEDDING_DIMENSION,
        }
    }

    fn identity_session() -> Result<Session> {
        let tensor_type = ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new([1, 3, 112, 112]),
            dimension_symbols: SymbolicDimensions::empty(4),
        };

        let mut graph = Graph::new()?;
        graph.set_inputs([Outlet::new("image", tensor_type.clone())])?;
        graph.set_outputs([Outlet::new("embedding", tensor_type)])?;
        graph.add_node(Node::new(
            "Identity",
            ONNX_DOMAIN,
            "identity",
            ["image"],
            ["embedding"],
            [],
        )?)?;

        let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22)?])?;
        model.add_graph(graph)?;

        let builder = SessionBuilder::new()?;
        Ok(model.into_session(&builder)?)
    }

    fn identity_service(expected_dimension: usize) -> Result<FaceEmbedding> {
        FaceEmbedding::from_session(
            EmbeddingModel {
                name: "test-face-model".to_string(),
                version: "test-version".to_string(),
                dimension: expected_dimension,
            },
            identity_session()?,
        )
    }

    fn encode_png(pixel: Rgba<u8>) -> Vec<u8> {
        let image = ImageBuffer::from_pixel(1, 1, pixel);
        let mut bytes = Cursor::new(Vec::new());

        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();

        bytes.into_inner()
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {actual} to be close to {expected}"
        );
    }

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

    #[test]
    fn face_embedding_service_new_rejects_zero_model_dimension() {
        let mut config = model_config("/tmp/model.onnx");
        config.dimension = 0;

        let error = FaceEmbedding::new(&config).unwrap_err();

        assert_eq!(
            error.to_string(),
            "face model dimension must be greater than 0"
        );
    }

    #[test]
    fn face_embedding_service_new_rejects_empty_model_path() {
        let error = FaceEmbedding::new(&model_config("   ")).unwrap_err();

        assert_eq!(
            error.to_string(),
            "face embedding model path cannot be empty"
        );
    }

    #[test]
    fn face_embedding_service_new_wraps_model_load_errors() {
        let missing_path = std::env::temp_dir().join(format!(
            "face-guard-missing-model-{}.onnx",
            std::process::id()
        ));
        let config = model_config(missing_path.display().to_string());

        let error = FaceEmbedding::new(&config).unwrap_err();

        assert!(
            error
                .to_string()
                .starts_with("failed to load face embedding ONNX model from ")
        );
    }

    #[test]
    fn face_embedding_service_from_session_stores_model_and_io_names() {
        let service = identity_service(FACE_EMBEDDING_DIMENSION).unwrap();

        assert_eq!(service.model.name, "test-face-model");
        assert_eq!(service.model.version, "test-version");
        assert_eq!(service.model.dimension, FACE_EMBEDDING_DIMENSION);
        assert_eq!(service.input_name, "image");
        assert_eq!(service.output_name, "embedding");
    }

    #[test]
    fn generate_embedding_rejects_empty_image_bytes() {
        let mut service = identity_service(FACE_EMBEDDING_DIMENSION).unwrap();

        let error = service.generate_embedding(&[]).unwrap_err();

        assert_eq!(error.to_string(), "image bytes cannot be empty");
    }

    #[test]
    fn generate_embedding_returns_normalized_embedding() {
        let mut service = identity_service(FACE_EMBEDDING_DIMENSION).unwrap();
        let image_bytes = encode_png(Rgba([255, 0, 127, 255]));

        let generated = service.generate_embedding(&image_bytes).unwrap();

        assert_eq!(generated.model.name, "test-face-model");
        assert_eq!(generated.model.version, "test-version");
        assert_eq!(generated.vector.dimension(), FACE_EMBEDDING_DIMENSION);
        assert!(
            generated
                .vector
                .values()
                .iter()
                .all(|value| value.is_finite())
        );

        let norm = generated
            .vector
            .values()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert_close(norm, 1.0);
    }

    #[test]
    fn generate_embedding_rejects_unexpected_output_dimension() {
        let mut service = identity_service(1).unwrap();
        let image_bytes = encode_png(Rgba([255, 0, 127, 255]));

        let error = service.generate_embedding(&image_bytes).unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid embedding dimension: expected 1, got 37632"
        );
    }
}
