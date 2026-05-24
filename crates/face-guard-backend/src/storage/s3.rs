use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use object_store::{
    Attribute, Attributes, ObjectStore, PutMode, PutOptions, aws::AmazonS3Builder, path::Path,
};

use crate::{config::StorageConfig, storage::ObjectStorage};

#[derive(Clone)]
pub struct S3ObjectStorage {
    store: Arc<dyn ObjectStore>,
}

impl S3ObjectStorage {
    pub fn new(config: &StorageConfig) -> Result<Self> {
        let store = AmazonS3Builder::new()
            .with_bucket_name(&config.bucket)
            .with_region(&config.region)
            .with_access_key_id(&config.access_key_id)
            .with_secret_access_key(&config.secret_access_key)
            .with_endpoint(&config.endpoint)
            .with_allow_http(true)
            .build()
            .context("ailed to build S3 object storage client")?;

        Ok(Self {
            store: Arc::new(store),
        })
    }
}

#[async_trait]
impl ObjectStorage for S3ObjectStorage {
    async fn put_object(&self, key: &str, content_type: &str, bytes: Vec<u8>) -> Result<()> {
        if bytes.is_empty() {
            anyhow::bail!("object bytes cannot be empty");
        }

        let mut attributes = Attributes::new();
        attributes.insert(Attribute::ContentType, content_type.to_string().into());

        let options = PutOptions {
            mode: PutMode::Create,
            attributes,
            ..Default::default()
        };

        self.store
            .put_opts(&Path::from(key), bytes.into(), options)
            .await
            .with_context(|| format!("failed to put object to S3 with key '{}'", key))?;

        Ok(())
    }
}
