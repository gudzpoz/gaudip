use std::io;

use interprocess::{
    local_socket::{
    GenericFilePath, GenericNamespaced, Name, tokio::{RecvHalf, prelude::*}
}};
use tokio::io::{AsyncReadExt, BufReader};

struct Packet {
    packet_id: u32,
    endpoint_id: u32,
    content: Vec<u8>,
}
async fn read_packet(
    reader: &mut BufReader<RecvHalf>, mut content_buf: Vec<u8>,
) -> io::Result<Packet> {
    let mut buf = [0u8; 4 * 4];
    reader.read_exact(&mut buf).await?;
    let packet_id = u32::from_ne_bytes(buf[0..4].try_into().unwrap());
    let endpoint_id = u32::from_ne_bytes(buf[4..8].try_into().unwrap());
    let byte_length = u32::from_ne_bytes(buf[8..12].try_into().unwrap());
    let _reserved = u32::from_ne_bytes(buf[12..16].try_into().unwrap());

    reader.take(byte_length as u64).read_to_end(&mut content_buf).await?;

    Ok(Packet { packet_id, endpoint_id, content: content_buf })
}

fn get_name() -> Option<Name<'static>> {
    if GenericFilePath::is_supported() {
        for provider in [
            dirs::runtime_dir,
            dirs::config_dir,
            dirs::data_dir,
            dirs::cache_dir,
        ] {
            if let Some(path) = provider() {
                let path = path.join("juicemacs").join("server");
                if !path.exists() {
                    continue;
                }
                if let Ok(name) = path.to_fs_name::<GenericFilePath>() {
                    return Some(name);
                }
            }
        }
    }
    if GenericNamespaced::is_supported() {
        return "juicemacs.socket".to_ns_name::<GenericNamespaced>().ok();
    }
    None
}