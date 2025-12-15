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
    #[error("unsupported operation")]
    Unsupported,
    #[error("remote error")]
    RemoteError(Vec<u8>),
}

pub type IpcResult<T> = Result<T, IpcError>;
