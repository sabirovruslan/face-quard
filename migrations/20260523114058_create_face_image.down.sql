DROP INDEX IF EXISTS face_images_created_at_idx;
DROP INDEX IF EXISTS face_images_status_idx;
DROP INDEX IF EXISTS face_images_collection_slug_idx;

DROP TABLE IF EXISTS face_images;

DROP EXTENSION IF EXISTS vector;
