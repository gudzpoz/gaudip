use crate::schema::*;
use crate::error::IpcError;
use flatbuffers::FlatBufferBuilder;

macro_rules! handle {
    ($self:ident, $handler:ident, $request:ty, $response:ty, $body:expr, $out:expr) => {{
        let request = flatbuffers::root::<$request>($body).map_err(|_| IpcError::InvalidRequest)?;
        $self.$handler(&request, $out).await
    }};
}

include!(concat!(env!("OUT_DIR"), "/flatbuffers/listener.rs"));