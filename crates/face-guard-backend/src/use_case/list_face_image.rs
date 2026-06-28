use anyhow::{Context, Result};

use crate::{
    domain::{CollectionSlug, FaceImageStatus},
    repository::face_image::{
        FaceImageReadRepository, ListFaceImagesCursor, ListFaceImagesOutput, ListFaceImagesQuery,
    },
};

#[derive(Debug)]
pub struct ListFaceImagesInput {
    pub collection_slug: Option<CollectionSlug>,
    pub status: Option<FaceImageStatus>,
    pub limit: usize,
    pub cursor: Option<ListFaceImagesCursor>,
}

pub struct ListFaceImagesUseCase<R>
where
    R: FaceImageReadRepository,
{
    repository: R,
}

impl<R> ListFaceImagesUseCase<R>
where
    R: FaceImageReadRepository,
{
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, input: ListFaceImagesInput) -> Result<ListFaceImagesOutput> {
        let items = self
            .repository
            .search(ListFaceImagesQuery {
                collection_slug: input.collection_slug,
                status: input.status,
                limit: input.limit,
                cursor: input.cursor.map(|value| ListFaceImagesCursor {
                    created_at: value.created_at,
                    id: value.id,
                }),
            })
            .await
            .context("failed to get list of face images")?;

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use anyhow::bail;
    use async_trait::async_trait;
    use chrono::{TimeZone, Utc};
    use uuid::Uuid;

    use crate::{
        domain::{FaceImageId, FaceImageKey},
        repository::face_image::FaceImageItem,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CapturedQuery {
        collection_slug: Option<String>,
        status: Option<FaceImageStatus>,
        limit: usize,
        cursor_created_at: Option<chrono::DateTime<Utc>>,
        cursor_id: Option<Uuid>,
    }

    #[derive(Debug, Default)]
    struct FakeRepositoryState {
        last_query: Option<CapturedQuery>,
        output: Option<ListFaceImagesOutput>,
        error: Option<String>,
    }

    #[derive(Debug, Clone, Default)]
    struct FakeRepository {
        state: Arc<Mutex<FakeRepositoryState>>,
    }

    impl FakeRepository {
        fn with_output(output: ListFaceImagesOutput) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeRepositoryState {
                    last_query: None,
                    output: Some(output),
                    error: None,
                })),
            }
        }

        fn failing(error: impl Into<String>) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeRepositoryState {
                    last_query: None,
                    output: None,
                    error: Some(error.into()),
                })),
            }
        }

        fn last_query(&self) -> Option<CapturedQuery> {
            self.state.lock().unwrap().last_query.clone()
        }
    }

    #[async_trait]
    impl FaceImageReadRepository for FakeRepository {
        async fn search(&self, query: ListFaceImagesQuery) -> Result<ListFaceImagesOutput> {
            let mut state = self.state.lock().unwrap();

            state.last_query = Some(CapturedQuery {
                collection_slug: query
                    .collection_slug
                    .as_ref()
                    .map(|value| value.as_str().to_string()),
                status: query.status,
                limit: query.limit,
                cursor_created_at: query.cursor.as_ref().map(|value| value.created_at),
                cursor_id: query.cursor.as_ref().map(|value| value.id.as_uuid()),
            });

            if let Some(error) = &state.error {
                bail!(error.clone());
            }

            state
                .output
                .take()
                .ok_or_else(|| anyhow::anyhow!("fake repository output was not configured"))
        }
    }

    #[tokio::test]
    async fn execute_forwards_input_to_repository_query() {
        let cursor_id = FaceImageId::from_uuid(Uuid::new_v4());
        let cursor_created_at = Utc.with_ymd_and_hms(2026, 6, 28, 10, 0, 0).unwrap();
        let repository = FakeRepository::with_output(empty_output());
        let use_case = ListFaceImagesUseCase::new(repository.clone());

        let output = use_case
            .execute(ListFaceImagesInput {
                collection_slug: Some(CollectionSlug::new("test_collection").unwrap()),
                status: Some(FaceImageStatus::Processed),
                limit: 25,
                cursor: Some(ListFaceImagesCursor {
                    created_at: cursor_created_at,
                    id: cursor_id,
                }),
            })
            .await
            .unwrap();

        assert!(output.items.is_empty());
        assert_eq!(
            repository.last_query().unwrap(),
            CapturedQuery {
                collection_slug: Some("test_collection".to_string()),
                status: Some(FaceImageStatus::Processed),
                limit: 25,
                cursor_created_at: Some(cursor_created_at),
                cursor_id: Some(cursor_id.as_uuid()),
            }
        );
    }

    #[tokio::test]
    async fn execute_returns_repository_output() {
        let item = face_image_item("faces/person-1.jpg");
        let next_cursor = ListFaceImagesCursor::from(&item);
        let repository = FakeRepository::with_output(ListFaceImagesOutput {
            items: vec![item.clone()],
            next_cursor: Some(next_cursor),
            has_more: true,
        });
        let use_case = ListFaceImagesUseCase::new(repository);

        let output = use_case
            .execute(ListFaceImagesInput {
                collection_slug: None,
                status: None,
                limit: 10,
                cursor: None,
            })
            .await
            .unwrap();

        assert_eq!(output.items.len(), 1);
        assert_eq!(output.items[0].id, item.id);
        assert_eq!(output.items[0].image_key, item.image_key);
        assert!(output.next_cursor.is_some());
        assert!(output.has_more);
    }

    #[tokio::test]
    async fn execute_wraps_repository_error_with_context() {
        let repository = FakeRepository::failing("database is unavailable");
        let use_case = ListFaceImagesUseCase::new(repository);

        let error = use_case
            .execute(ListFaceImagesInput {
                collection_slug: None,
                status: None,
                limit: 10,
                cursor: None,
            })
            .await
            .unwrap_err();

        let error = format!("{error:#}");
        assert!(error.contains("failed to get list of face images"));
        assert!(error.contains("database is unavailable"));
    }

    fn empty_output() -> ListFaceImagesOutput {
        ListFaceImagesOutput {
            items: Vec::new(),
            next_cursor: None,
            has_more: false,
        }
    }

    fn face_image_item(image_key: &str) -> FaceImageItem {
        let created_at = Utc.with_ymd_and_hms(2026, 6, 28, 10, 0, 0).unwrap();

        FaceImageItem {
            id: FaceImageId::from_uuid(Uuid::new_v4()),
            image_key: FaceImageKey::from_existing(image_key).unwrap(),
            collection_slug: CollectionSlug::new("test_collection").unwrap(),
            status: FaceImageStatus::Processed,
            created_at,
            updated_at: created_at,
        }
    }
}
