use axum::{
    Router,
    routing::{get, post},
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    bootstrap::server::AppState,
    http::handlers::{
        create_face_image::create_face_image, face_search::search_similar_face,
        list_face_image::list_face_images, object::upload_object,
    },
};

pub fn create_router(state: AppState) -> Router {
    let router = Router::new();

    let router = router.route("/", get(|| async { "Main page" }));
    let router = router.route("/health", get(|| async { "OK" }));
    let router = router.route("/api/v1/objects/upload", post(upload_object));
    let router = router.route("/api/v1/faces/create", post(create_face_image));
    let router = router.route("/api/v1/faces/list", post(list_face_images));
    let router = router.route("/api/v1/faces/search_similar", post(search_similar_face));

    router
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}
