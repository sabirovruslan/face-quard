use axum::{
    Router,
    routing::{get, post},
};

use crate::{bootstrap::server::AppState, http::handlers::face_image::upload_face_image};

pub fn create_router(state: AppState) -> Router {
    let router = Router::new();

    let router = router.route("/", get(|| async { "Main page" }));
    let router = router.route("/health", get(|| async { "OK" }));
    let router = router.route("/api/v1/face-images", post(upload_face_image));

    router.with_state(state)
}
