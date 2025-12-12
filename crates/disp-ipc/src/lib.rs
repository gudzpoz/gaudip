pub mod ipc;
pub mod schema;

#[cfg(test)]
mod tests {
    use std::{io::{Read, Write}, sync::{Arc, Condvar, Mutex}, thread};

    use interprocess::{
        bound_util::{RefRead, RefWrite},
        local_socket::{
            prelude::*,
            GenericFilePath, GenericNamespaced, ListenerOptions, Name, NameType,
            Stream, ToFsName, ToNsName,
        },
    };

    use super::*;

    fn get_name() -> Name<'static> {
        if GenericFilePath::is_supported() {
            "/tmp/test.sock".to_fs_name::<GenericFilePath>().unwrap()
        } else {
            "test.sock".to_ns_name::<GenericNamespaced>().unwrap()
        }
    }

    const PING_PONG_COUNT: u32 = 50_000;

    #[test]
    fn test_ipc_ping_pong() {
        let init_recv = Arc::new((Mutex::new(false), Condvar::new()));
        let init_send = Arc::clone(&init_recv);
        let server = thread::spawn(move || {
            let mut listener = ListenerOptions::new()
                .name(get_name())
                .create_sync()
                .unwrap();
            let (started, cvar) = &*init_send;
            let mut started = started.lock().unwrap();
            *started = true;
            cvar.notify_all();
            drop(started);
            let stream = listener.next().unwrap().unwrap();
            let mut count = 0;
            let mut buf = [0u8; 1024];
            let mut fbb = flatbuffers::FlatBufferBuilder::new();
            loop {
                let mut reader = stream.as_read();
                let mut length_bytes = [0u8; 4];
                let Ok(()) = reader.read_exact(&mut length_bytes) else { break };
                let length = u32::from_le_bytes(length_bytes);
                let b = &mut buf[0..length as usize];
                reader.read_exact(b).unwrap();

                fbb.reset();
                let ping = flatbuffers::root::<schema::PingRequest>(b).unwrap();
                let s = fbb.create_string(ping.ping().unwrap());
                let pong = schema::PingResponse::create(&mut fbb, &schema::PingResponseArgs {
                    pong: Some(s),
                });
                fbb.finish(pong, None);
                let buf = fbb.finished_data();

                let length = buf.len() as u32;
                let length_bytes = length.to_le_bytes();
                let mut writer = stream.as_write();
                writer.write_all(&length_bytes).unwrap();
                writer.write_all(buf).unwrap();
                writer.flush().unwrap();
                count += 1;
            }
            assert_eq!(count, PING_PONG_COUNT);
        });

        let (started, cvar) = &*init_recv;
        let mut started = started.lock().unwrap();
        while !*started {
            started = cvar.wait(started).unwrap();
        }
        let client = thread::spawn(|| {
            let stream = Stream::connect(get_name()).unwrap();
            let mut read_buf = [0u8; 1024];
            let mut fbb = flatbuffers::FlatBufferBuilder::new();
            let mut latencies = 0u128;
            for i in 0..PING_PONG_COUNT {
                let start_time = std::time::Instant::now();

                fbb.reset();
                let ping_str = format!("{}", i);
                let s = fbb.create_string(&ping_str);
                let ping = schema::PingRequest::create(&mut fbb, &schema::PingRequestArgs {
                    ping: Some(s),
                });
                fbb.finish(ping, None);
                let buf = fbb.finished_data();

                let length = buf.len() as u32;
                let length_bytes = length.to_le_bytes();
                let mut writer = stream.as_write();
                writer.write_all(&length_bytes).unwrap();
                writer.write_all(buf).unwrap();
                writer.flush().unwrap();

                let mut reader = stream.as_read();
                let mut length_bytes = [0u8; 4];
                reader.read_exact(&mut length_bytes).unwrap();
                let length = u32::from_le_bytes(length_bytes);
                let b = &mut read_buf[0..length as usize];
                reader.read_exact(b).unwrap();

                let ping = flatbuffers::root::<schema::PingResponse>(b).unwrap();
                assert_eq!(ping.pong(), Some(ping_str.as_str()));

                let duration = std::time::Instant::now() - start_time;
                latencies += duration.as_nanos();
            }
            println!("latency: {} us", latencies as f64 / 1000.0 / (PING_PONG_COUNT as f64));
        });

        server.join().unwrap();
        client.join().unwrap();
    }
}
