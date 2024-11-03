use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ByteTrackError {
    #[error("Error: {0}")]
    LapjvError(String),
}
