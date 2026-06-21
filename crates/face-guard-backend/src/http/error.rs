use axum::{extract::multipart::MultipartError, http::StatusCode, response::IntoResponse};

pub struct AppHttpError(anyhow::Error);

impl IntoResponse for AppHttpError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

impl From<anyhow::Error> for AppHttpError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl From<MultipartError> for AppHttpError {
    fn from(value: MultipartError) -> Self {
        Self(value.into())
    }
}
