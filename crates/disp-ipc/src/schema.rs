#[allow(warnings)]
#[rustfmt::skip]
mod gen_flatbuffers {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use gen_flatbuffers::party::iroiro::juicemacs::ipc::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_request() {
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let msg = fbb.create_string("ping");
        let ping = PingRequest::create(&mut fbb, &PingRequestArgs {
            ping: Some(msg),
        });
        fbb.finish(ping, None);
        let buf = fbb.finished_data();

        let ping = flatbuffers::root::<PingRequest<'_>>(buf).unwrap();
        assert_eq!("ping", ping.ping());
    }

    #[test]
    fn test_ping_response() {
        let mut fbb = flatbuffers::FlatBufferBuilder::new();
        let msg = fbb.create_string("pong");
        let ping = PingResponse::create(&mut fbb, &PingResponseArgs {
            pong: Some(msg),
        });
        fbb.finish(ping, None);
        let buf = fbb.finished_data();

        let ping = flatbuffers::root::<PingResponse<'_>>(buf).unwrap();
        assert_eq!("pong", ping.pong());
    }
}
