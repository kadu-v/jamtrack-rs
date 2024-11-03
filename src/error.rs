use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ByteTrackError {
    #[error("Error: {0}")]
    LapjvError(String),
    #[error("Error: {0}")]
    ExecLapjvError(String),
    #[error("Error: {0}")]
    ByteTrackerError(String),
}
