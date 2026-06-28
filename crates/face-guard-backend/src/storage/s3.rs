use std::{sync::Arc, time::Duration};

use anyhow::{Context, Ok, Result};
use async_trait::async_trait;
use object_store::{
    Attribute, Attributes, ObjectStore, ObjectStoreExt, PutMode, PutOptions,
    aws::{AmazonS3, AmazonS3Builder},
    path::Path,
    signer::Signer,
};
use reqwest::Method;

use crate::{config::StorageConfig, storage::ObjectStorage};

#[derive(Clone)]
pub struct S3ObjectStorage {
    store: Arc<AmazonS3>,
    public_signer: Arc<AmazonS3>,
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
            .context("failed to build S3 object storage client")?;

        let public_signer = AmazonS3Builder::new()
            .with_bucket_name(&config.bucket)
            .with_region(&config.region)
            .with_access_key_id(&config.access_key_id)
            .with_secret_access_key(&config.secret_access_key)
            .with_endpoint(&config.public_endpoint)
            .with_allow_http(true)
            .build()
            .context("failed to build public S3 signer")?;

        Ok(Self {
            store: Arc::new(store),
            public_signer: Arc::new(public_signer),
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

    async fn get_object(&self, key: &str) -> Result<Vec<u8>> {
        let bytes = self
            .store
            .get(&Path::from(key))
            .await
            .with_context(|| format!("failed to get object from S3 with key '{}'", key))?
            .bytes()
            .await
            .with_context(|| format!("failed to read object bytes from S3 with key '{}'", key))?;

        Ok(bytes.to_vec())
    }

    async fn presigned_get_url(&self, key: &str, expires_in: Duration) -> Result<String> {
        let url = self
            .public_signer
            .signed_url(Method::GET, &Path::from(key), expires_in)
            .await
            .with_context(|| format!("failed to generate presigned GET URL for key '{key}'"))?;

        Ok(url.to_string())
    }
}
