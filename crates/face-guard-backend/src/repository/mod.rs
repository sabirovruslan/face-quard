pub mod face_embedding;
pub mod face_image;

use sqlx::PgPool;

#[derive(Debug, Clone)]
pub struct PgRepository {
    db_pool: PgPool,
}

impl PgRepository {
    pub fn new(db_pool: PgPool) -> Self {
        Self { db_pool }
    }
}
