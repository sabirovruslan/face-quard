-- Add down migration script here
DROP INDEX IF EXISTS face_embeddings_face_image_id_idx;
DROP INDEX IF EXISTS face_embeddings_model_idx;
DROP INDEX IF EXISTS face_embeddings_embedding_hnsw_idx;

DROP TABLE IF EXISTS face_embeddings;

DROP EXTENSION IF EXISTS vector;
