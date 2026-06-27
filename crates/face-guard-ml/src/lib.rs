pub mod detector;
pub mod embedding;
mod image;

pub use embedding::{
    EmbeddingModel, EmbeddingModelConfig, EmbeddingVector, FaceEmbedding, FaceEmbeddingGenerator,
    GeneratedFaceEmbedding,
};

pub use detector::{FaceCrop, FaceDetectionModelConfig, FaceDetector, ScrfdFaceDetector};
