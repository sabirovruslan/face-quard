pub mod s3;

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait ObjectStorage: Send + Sync {
    async fn put_object(&self, key: &str, content_type: &str, bytes: Vec<u8>) -> Result<()>;
    async fn get_object(&self, key: &str) -> Result<Vec<u8>>;
    async fn presigned_get_url(&self, key: &str, expires_in: Duration) -> Result<String>;
}
