use std::{io, sync::{atomic::{AtomicU32, Ordering}, Arc}};

use dashmap::DashMap;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use interprocess::local_socket::{
    prelude::*, tokio::{RecvHalf, SendHalf, Stream}, traits::tokio::Stream as _, GenericFilePath, GenericNamespaced,
    Name,
};
use object_pool::Pool;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot}, task::JoinHandle,
};

use crate::{error::{IpcError, IpcResult}, listener, schema::*};

type RecvBufferPool = Arc<Pool<Vec<u8>>>;
type SendBufferPool = Arc<Pool<FlatBufferBuilder<'static>>>;

struct RequestQueueItem {
    info: RequestQueueItemInfo,
    body: FlatBufferBuilder<'static>,
}
enum RequestQueueItemInfo {
    Request {
        endpoint_id: u32,
        callback: oneshot::Sender<IpcResult<Vec<u8>>>,
    },
    Response {
        packet_id: u32,
    },
    ErrorResponse {
        packet_id: u32,
    },
}

struct PacketSender {
    packet_id: IdGenerator,
    pool: SendBufferPool,
    sender: SendHalf,
    request_queue: mpsc::Receiver<RequestQueueItem>,
    callback_queue: Arc<DashMap<u32, oneshot::Sender<IpcResult<Vec<u8>>>>>,

    stopped: oneshot::Receiver<()>,
}
impl PacketSender {
    pub async fn run(mut self) -> IpcResult<()> {
        let mut run_loop = async || -> IpcResult<()> {
            while let Some(RequestQueueItem { info, body }) = self.request_queue.recv().await {
                let info = match info {
                    RequestQueueItemInfo::Request { endpoint_id, callback } => {
                        let packet_id = self.packet_id.next();
                        assert!(self.callback_queue.insert(packet_id, callback).is_none());
                        PacketInfo { packet_id, endpoint_id }
                    },
                    RequestQueueItemInfo::Response { packet_id } => {
                        PacketInfo { packet_id, endpoint_id: 0 }
                    },
                    RequestQueueItemInfo::ErrorResponse { packet_id } => {
                        PacketInfo { packet_id, endpoint_id: u32::MAX }
                    },
                };
                write_packet(&mut self.sender, &info, &body).await?;
                self.pool.attach(body);
            }
            Ok(())
        };
        tokio::select! {
            _ = self.stopped => Ok(()),
            result = run_loop() => result,
        }
    }
}

#[derive(Clone)]
struct PacketSenderSender {
    send_pool: SendBufferPool,
    recv_pool: RecvBufferPool,
    sender: mpsc::Sender<RequestQueueItem>,
}
impl PacketSenderSender {
    fn get_buffer(&self) -> FlatBufferBuilder<'static> {
        let (_, mut buffer) = self.send_pool.pull(FlatBufferBuilder::new).detach();
        buffer.reset();
        buffer
    }

    fn return_buffer(&self, buffer: Vec<u8>) {
        self.recv_pool.attach(buffer);
    }

    async fn send(&self, endpoint_id: u32, body: FlatBufferBuilder<'static>) -> IpcResult<Vec<u8>> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(RequestQueueItem {
            body,
            info: RequestQueueItemInfo::Request { endpoint_id, callback: tx },
        }).await.map_err(|_| IpcError::BrokenPipeError)?;
        rx.await.map_err(|_| IpcError::BrokenPipeError)?
    }
}

struct PacketReceiver<T: listener::IpcListener> {
    recv_pool: RecvBufferPool,
    send_pool: SendBufferPool,
    receiver: BufReader<RecvHalf>,
    sender: mpsc::Sender<RequestQueueItem>,
    callback_queue: Arc<DashMap<u32, oneshot::Sender<IpcResult<Vec<u8>>>>>,
    listener: T,

    stopped: oneshot::Receiver<()>,
}
impl<T: listener::IpcListener> PacketReceiver<T> {
    pub async fn run(mut self) -> Result<(), IpcError> {
        let mut run_loop = async || -> Result<(), IpcError> {
            loop {
                let (_, mut buffer) = self.recv_pool.pull(Vec::new).detach();
                let packet_info = read_packet(&mut self.receiver, &mut buffer).await?;
                if packet_info.endpoint_id == 0 || packet_info.endpoint_id == u32::MAX {
                    if let Some((_, callback)) = self.callback_queue.remove(&packet_info.packet_id) {
                        callback.send(if packet_info.endpoint_id == 0 {
                            Ok(buffer)
                        } else {
                            Err(IpcError::RemoteError(buffer))
                        }).map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))?;
                        // TODO: log
                    } else {
                        self.recv_pool.attach(buffer);
                        // TODO: log
                    }
                } else {
                    let (_, mut output) = self.send_pool.pull(FlatBufferBuilder::new).detach();
                    output.reset();
                    let result = listener::handle(
                        &self.listener, packet_info.endpoint_id, &buffer, &mut output,
                    ).await;
                    let info = match result {
                        Ok(()) => RequestQueueItemInfo::Response { packet_id: packet_info.packet_id },
                        Err(_) => {
                            output.reset();
                            let root = crate::schema::IpcError::create(&mut output, &IpcErrorArgs {
                                type_: IpcErrorType::GenericError, message: None,
                            });
                            output.finish(root, None);
                            RequestQueueItemInfo::ErrorResponse { packet_id: packet_info.packet_id }
                        }
                    };
                    self.sender.send(RequestQueueItem { body: output, info }).await
                        .map_err(|_| IpcError::BrokenPipeError)?;
                }
            }
        };

        tokio::select! {
            _ = self.stopped => Ok(()),
            result = run_loop() => result,
        }
    }
}

pub struct IpcChannels {
    sender_sender: PacketSenderSender,

    sender_task: JoinHandle<IpcResult<()>>,
    sender_stop: oneshot::Sender<()>,
    receiver_task: JoinHandle<IpcResult<()>>,
    receiver_stop: oneshot::Sender<()>,
}
impl IpcChannels {
    pub fn create_ipc_channels<T: listener::IpcListener + Sync + Send + 'static>(
        stream: Stream, listener: T, client: bool,
    ) -> IpcChannels {
        let (recv_half, send_half) = stream.split();

        let (sender_stopper, sender_stop) = oneshot::channel();
        let (receiver_stopper, receiver_stop) = oneshot::channel();

        let (tx, rx) = mpsc::channel(4);
        let sender = PacketSender {
            packet_id: IdGenerator::new(client),
            pool: SendBufferPool::new(Pool::new(32, FlatBufferBuilder::new)),
            sender: send_half,
            request_queue: rx,
            callback_queue: Arc::new(DashMap::new()),
            stopped: sender_stop,
        };

        let receiver = PacketReceiver {
            recv_pool: RecvBufferPool::new(Pool::new(32, Vec::new)),
            send_pool: sender.pool.clone(),
            sender: tx.clone(),
            receiver: BufReader::new(recv_half),
            callback_queue: sender.callback_queue.clone(),
            stopped: receiver_stop,
            listener,
        };

        let sender_sender = PacketSenderSender {
            send_pool: sender.pool.clone(),
            recv_pool: receiver.recv_pool.clone(),
            sender: tx,
        };

        let sender_task = tokio::spawn(async { sender.run().await });
        let receiver_task = tokio::spawn(async { receiver.run().await });

        IpcChannels {
            sender_sender,
            sender_task, receiver_task,
            sender_stop: sender_stopper,
            receiver_stop: receiver_stopper,
        }
    }

    pub async fn close(self) -> Result<(), IpcError> {
        let _ = self.sender_stop.send(());
        self.sender_task.await.map_err(|_| IpcError::BrokenPipeError)??;
        let _ = self.receiver_stop.send(());
        self.receiver_task.await.map_err(|_| IpcError::BrokenPipeError)??;
        Ok(())
    }
}
include!(concat!(env!("OUT_DIR"), "/flatbuffers/requests.rs"));

#[derive(Copy, Clone, Debug)]
pub struct PacketInfo {
    pub packet_id: u32,
    pub endpoint_id: u32,
}

/// Read a packet from the given reader
pub async fn read_packet(
    reader: &mut BufReader<RecvHalf>, content_buf: &mut Vec<u8>,
) -> io::Result<PacketInfo> {
    content_buf.clear();

    let mut header = [0u8; 16];
    reader.read_exact(&mut header).await?;

    let packet_id = u32::from_ne_bytes(header[0..4].try_into().unwrap());
    let endpoint_id = u32::from_ne_bytes(header[4..8].try_into().unwrap());
    let byte_length = u32::from_ne_bytes(header[8..12].try_into().unwrap());
    let reserved = u32::from_ne_bytes(header[12..16].try_into().unwrap());
    assert_eq!(reserved, 0);

    content_buf.clear();
    reader.take(byte_length as u64).read_to_end(content_buf).await?;

    Ok(PacketInfo { packet_id, endpoint_id })
}

/// Write a packet to the given writer
pub async fn write_packet(
    writer: &mut SendHalf, info: &PacketInfo, content: &FlatBufferBuilder<'static>,
) -> io::Result<()> {
    let mut header = [0u8; 16];
    let body = content.finished_data();

    header[0..4].copy_from_slice(&info.packet_id.to_ne_bytes());
    header[4..8].copy_from_slice(&info.endpoint_id.to_ne_bytes());
    header[8..12].copy_from_slice(&(body.len() as u32).to_ne_bytes());
    header[12..16].copy_from_slice(&0u32.to_ne_bytes());

    writer.write_all(&header).await?;
    writer.write_all(body).await?;
    writer.flush().await?;

    Ok(())
}

pub fn get_server_name() -> Option<Name<'static>> {
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

pub struct IdGenerator(AtomicU32);
impl IdGenerator {
    pub fn new(client: bool) -> Self {
        Self(AtomicU32::new(if client { 1 } else { 2 }))
    }

    pub fn next(&self) -> u32 {
        self.0.fetch_add(2, Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listener::*;
    use interprocess::local_socket::{traits::tokio::Listener, ListenerOptions};

    #[test]
    fn test_id_generator() {
        let id_gen = IdGenerator::new(true);
        assert_eq!(id_gen.next(), 1);
        assert_eq!(id_gen.next(), 3);
        let id_gen = IdGenerator::new(false);
        assert_eq!(id_gen.next(), 2);
        assert_eq!(id_gen.next(), 4);
    }

    struct Server();
    impl IpcListener for Server {
        async fn handle_ping(
            &self, request: &'_ PingRequest<'_>, response: &mut FlatBufferBuilder<'static>
        ) -> Result<(), IpcError> {
            let pong = Some(response.create_string(request.ping()));
            let root = PingResponse::create(response, &PingResponseArgs { pong });
            response.finish(root, None);
            Ok(())
        }
    }

    fn get_test_server_name() -> Name<'static> {
        let i = rand::random::<u32>();
        format!("juicemacs-{i}.socket").to_ns_name::<GenericNamespaced>().unwrap()
    }

    async fn try_test<F1, F2>(test_server: F1, test_client: F2) -> Result<(), IpcError>
    where F1: Send + Sync + 'static + AsyncFnOnce(&'_ mut IpcChannels),
          F2: Send + Sync + 'static + AsyncFnOnce(&'_ mut IpcChannels)
    {
        let name = get_test_server_name();
        let name_client = name.clone();

        let (stop, stopped) = oneshot::channel::<()>();

        let server = async move {
            let listener = ListenerOptions::new()
                .name(name)
                .create_tokio()
                .unwrap();
            let conn = listener.accept().await.unwrap();
            let mut server = IpcChannels::create_ipc_channels(conn, Server(), false);
            test_server(&mut server).await;
            stopped.await.unwrap();
            server.close().await.expect_err("disconnected");
        };

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let client = async move {
            let conn = Stream::connect(name_client).await.unwrap();
            let mut client = IpcChannels::create_ipc_channels(conn, Server(), true);
            test_client(&mut client).await;
            client.close().await.unwrap();
            stop.send(()).unwrap();
        };

        futures::join!(client, server);
        Ok(())
    }

    #[tokio::test]
    async fn test_client() {
        try_test(async |server| {
            server.ping(|buf| {
                let ping = buf.create_string("ping from server");
                Ok(PingRequest::create(buf, &PingRequestArgs { ping: Some(ping) }))
            }, |response| {
                assert_eq!(response.pong(), "ping from server");
                Ok(())
            }).await.unwrap();
        }, async |client| {
            client.ping(|buf| {
                let ping = buf.create_string("ping from client");
                Ok(PingRequest::create(buf, &PingRequestArgs { ping: Some(ping) }))
            }, |response| {
                assert_eq!(response.pong(), "ping from client");
                Ok(())
            }).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }).await.unwrap();
    }

    #[tokio::test]
    async fn test_latency() {
        try_test(
            async |_server| {},
            async |client| {
                for msg_size in [16, 128, 1024, 8 * 1024] {
                    let s = "ping".repeat(msg_size / 4);
                    let mut latency_sum = 0;
                    let mut latency_max = 0;
                    let loop_count = 1000;
                    for _ in 0..loop_count {
                        let start = std::time::Instant::now();
                        let time = client.ping(|buf| {
                            let ping = buf.create_string(&s);
                            Ok(PingRequest::create(buf, &PingRequestArgs { ping: Some(ping) }))
                        }, |response| {
                            let nanos = start.elapsed().as_nanos();
                            assert_eq!(response.pong(), &s);
                            Ok(nanos)
                        }).await.unwrap();
                        latency_sum += time;
                        latency_max = latency_max.max(time);
                    }

                    println!(
                        "latency({} bytes): avg: {} us, max: {} us",
                        msg_size,
                        latency_sum as f64 / loop_count as f64 / 1000.0,
                        latency_max as f64 / 1000.0,
                    );
                }
            },
        ).await.unwrap();
    }

    #[tokio::test]
    async fn test_unsupported() {
        try_test(
            async |_server| {},
            async |client| {
                client.register(|buf| {
                    let type_ = Some(buf.create_string("t"));
                    let display = Some(buf.create_string(""));
                    Ok(RegistrationRequest::create(buf, &RegistrationRequestArgs { type_, display }))
                }, |_success| -> Result<(), IpcError> {
                    panic!("unreachable");
                }).await.expect_err("unsupported");
            },
        ).await.unwrap();
    }
}
