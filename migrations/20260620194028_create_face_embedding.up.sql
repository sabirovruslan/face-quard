-- Add up migration script here
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY,

    face_image_id UUID NOT NULL REFERENCES face_images(id) ON DELETE CASCADE,

    embedding vector(512) NOT NULL,

    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_dimension INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT face_embeddings_model_dimension_check CHECK (
        model_dimension = 512
    )
);

CREATE INDEX face_embeddings_face_image_id_idx
ON face_embeddings (face_image_id);

CREATE INDEX face_embeddings_model_idx
ON face_embeddings (model_name, model_version, model_dimension);

CREATE INDEX face_embeddings_embedding_hnsw_idx
ON face_embeddings
USING hnsw (embedding vector_cosine_ops);
