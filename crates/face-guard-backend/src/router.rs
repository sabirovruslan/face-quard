use axum::{Router, routing::get};

use crate::bootstrap::server::AppState;

pub fn create_router(state: AppState) -> Router {
    let router = Router::new();

    let router = router.route("/", get(|| async { "Main page" }));
    let router = router.route("/health", get(|| async { "OK" }));

    router.with_state(state)
}
