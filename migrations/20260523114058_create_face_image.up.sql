-- Add up migration script here
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE face_images (
    id UUID PRIMARY KEY,

    image_key TEXT NOT NULL,

    collection_slug TEXT NOT NULL,
    status TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()

);


CREATE INDEX face_images_collection_slug_idx
ON face_images (collection_slug);

CREATE INDEX face_images_status_idx
ON face_images (status);

CREATE INDEX face_images_created_at_idx
ON face_images (created_at);
