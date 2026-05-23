use std::sync::LazyLock;

pub static CONFIG: LazyLock<crate::config::AppConfig> =
    LazyLock::new(|| crate::config::AppConfig::from_env().unwrap());
