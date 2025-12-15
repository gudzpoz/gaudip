use thiserror::Error;

use crate::ipc::PacketInfo;

#[derive(Error, Debug)]
pub enum IpcError {
    #[error("failed to stream packet")]
    IoError(#[from] std::io::Error),
    #[error("failed to parse packet")]
    ParseError(PacketInfo),
    #[error("invalid request")]
    InvalidRequest,
    #[error("invalid response")]
    InvalidResponse,
    #[error("failed internal communication")]
    BrokenPipeError,
}

pub type IpcResult<T> = Result<T, IpcError>;
