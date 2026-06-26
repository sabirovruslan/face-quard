pub mod embedding;
mod image;

pub use embedding::{
    EmbeddingModel, EmbeddingModelsConfig, EmbeddingVector, FaceEmbedding, FaceEmbeddingGenerator,
    GeneratedFaceEmbedding,
};
