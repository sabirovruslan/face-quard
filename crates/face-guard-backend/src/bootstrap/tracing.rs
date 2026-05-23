use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "face_guard_backend=debug,tower_http=debug".into());

    let formating_layer = BunyanFormattingLayer::new("face-quard-backend".into(), std::io::stdout);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(JsonStorageLayer)
        .with(formating_layer)
        .init();
}
