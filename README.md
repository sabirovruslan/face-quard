# Face Guard

Face Guard is a Rust pet project for testing a hypothesis: a simple face search pipeline can be built with ONNX models without running a separate ML service.

The project covers this flow:

1. Upload an image to S3 compatible storage.
2. Create a face image from an `image_key`.
3. Detect the primary face with an ONNX face detector.
4. Crop the detected face.
5. Generate a face embedding with an ONNX embedding model.
6. Store the embedding in PostgreSQL with pgvector.
7. Search for similar faces with cosine similarity.

## Stack

- Rust
- Axum for the backend API
- Leptos for the web UI
- ONNX Runtime through `ort`
- PostgreSQL with pgvector
- MinIO as S3 compatible storage

## Project Structure

```text
crates/
  face-guard-backend  - Axum API, use cases, repositories, storage
  face-guard-ml       - ONNX face detector, embedding generator, image preprocessing
  face-guard-web      - Leptos UI for manual API checks

```

## Models

The project uses two ONNX models:

```text
models/det_10g.onnx
models/w600k_r50.onnx
```

`det_10g.onnx` is used for face detection.

This is an InsightFace SCRFD detector model, configured as:

```text
FACE_DETECTION_MODEL_NAME=insightface-scrfd
FACE_DETECTION_MODEL_VERSION=10g-kps
FACE_DETECTION_INPUT_SIZE=640
```

`w600k_r50.onnx` is used for face embedding generation.

This is an InsightFace recognition model from the Buffalo-L model pack, configured as:

```text
FACE_EMBEDDING_MODEL_NAME=insightface-buffalo-l
FACE_EMBEDDING_MODEL_PATH_MODEL_VERSION=w600k-r50
FACE_EMBEDDING_MODEL_DIMENSION=512
```

The detector finds the face area, then the cropped face is passed to the embedding model. The resulting 512 dimensional vector is stored in PostgreSQL and indexed with pgvector.

## API

Main endpoints:

```text
GET  /health
POST /api/v1/objects/upload
POST /api/v1/faces/create
POST /api/v1/faces/search_similar
POST /api/v1/faces/list
```

`/api/v1/objects/upload` uploads a file to the bucket by `image_key`.

`/api/v1/faces/create` downloads an image from the bucket, detects a face, generates an embedding and stores the result.

`/api/v1/faces/search_similar` downloads an image from the bucket, generates an embedding for the detected face and searches for similar saved faces.

`/api/v1/faces/list` returns saved face images with pagination and presigned URLs for image previews.

## Local Run

Copy the env file:

```bash
cp .env.example .env
```

Start the services:

```bash
docker compose up --build
```

After startup:

```text
Backend API: http://localhost:8080
Web UI:      http://localhost:8081
MinIO API:  http://localhost:9000
MinIO UI:   http://localhost:9001
```

## Environment

Important variables:

```text
APP_PORT=8080
WEB_PORT=8081

DATABASE_URL=postgres://postgres:postgres@postgres:5432/face_guard

S3_ENDPOINT=http://minio:9000
S3_PUBLIC_ENDPOINT=http://localhost:9000
S3_BUCKET=face-images

FACE_DETECTION_MODEL_PATH=/app/models/det_10g.onnx
FACE_DETECTION_MODEL_NAME=insightface-scrfd
FACE_DETECTION_MODEL_VERSION=10g-kps
FACE_DETECTION_INPUT_SIZE=640

FACE_EMBEDDING_MODEL_PATH=/app/models/w600k_r50.onnx
FACE_EMBEDDING_MODEL_NAME=insightface-buffalo-l
FACE_EMBEDDING_MODEL_PATH_MODEL_VERSION=w600k-r50
FACE_EMBEDDING_MODEL_DIMENSION=512
```

## Checks

```bash
cargo check --workspace
cargo test --workspace
```

To check the web crate for the wasm target:

```bash
cargo check -p face-guard-web --target wasm32-unknown-unknown
```

## Status

This is not a production system. It is a hypothesis check and a playground for ONNX inference in Rust.

