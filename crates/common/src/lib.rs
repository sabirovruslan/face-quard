use anyhow::{Context, Result};
use std::env;

pub fn get_env(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("{name} is required"))
}

pub fn get_env_or_default(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}
