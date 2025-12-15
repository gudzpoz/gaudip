use core::panic;
use std::{env, fs, path::{Path, PathBuf}, process::Command};

const RPC_SERVICE_SCHEMA: &str = "schema/rpc_service.fbs";
const SCHEMA_FILES: &[&str] = &[
    "schema/registration.fbs",
    "schema/buffer.fbs",
    RPC_SERVICE_SCHEMA,
];

fn main() {
    flatc_compile();
    rpc_service_compile();
}

fn rpc_service_compile() {
    let content = fs::read_to_string(RPC_SERVICE_SCHEMA).unwrap();
    #[derive(PartialEq, Eq)]
    enum Type {
        Client2Server,
        Server2Client,
        TwoWay,
    }
    struct Endpoint {
        name: String,
        request: String,
        response: String,
        id: u32,
        ty: Type,
    }
    let mut endpoints = Vec::new();
    for line in content.lines() {
        if !line.starts_with("  ") {
            continue;
        }
        // `  Ping(PingRequest):PingResponse (id: 1, dir: "2way");`
        let info = line.split(&['(', ')', ':', ','])
            .map(|s| s.trim()).collect::<Vec<_>>();
        // ["Ping", "PingRequest", "", "PingResponse", "id", "1", "dir", "2way", ";"]
        assert_eq!(info.len(), 9);
        assert_eq!(info[2], "");
        assert_eq!(info[4], "id");
        assert_eq!(info[6], "dir");
        assert_eq!(info[8], ";");
        let (endpoint, request, response, id, ty_str) = (
            info[0], info[1], info[3], info[5], info[7],
        );
        let id: usize = id.parse().unwrap();
        assert_ne!(id, 0);
        let ty = match ty_str.trim_matches('"') {
            "c2s" => Type::Client2Server,
            "s2c" => Type::Server2Client,
            "2way" => Type::TwoWay,
            _ => panic!("unknown type {}", ty_str),
        };

        assert_eq!(
            line,
            format!("  {endpoint}({request}):{response} (id: {id}, dir: {ty_str});")
        );

        endpoints.push(Endpoint {
            name: endpoint.to_string(),
            request: request.to_string(),
            response: response.to_string(),
            id: id as u32,
            ty,
        });
    }

    let out_file = out_dir().join("listener.rs");
    fs::write(
        out_file,
        format!(
            r#"
/// Server (remote) to client (this crate) requests
pub trait IpcListener {{
{}
}}
pub async fn handle<T: IpcListener>(
    listener: &T, endpoint_id: u32, body: &[u8],
    response: &mut FlatBufferBuilder<'static>,
) -> Result<(), IpcError> {{
    match endpoint_id {{
{}
        _ => Err(IpcError::InvalidRequest),
    }}
}}
"#,
            endpoints.iter()
                .map(|e| format!(
                    r#"    fn handle{}(
        &self, request: &'_ {}<'_>, response: &mut FlatBufferBuilder<'static>
    ) -> impl Future<Output = Result<(), IpcError>> + Send{}
"#,
                    camel_to_snake(&e.name),
                    e.request,
                    if e.ty != Type::Client2Server { ";" } else {
                        " {
        async { Err(IpcError::Unsupported) }
    }"
                    }
                ))
                .collect::<String>(),
            endpoints.iter()
                .map(|e| format!(
                    "        {} => handle!(listener, handle{}, {}, {}, body, response),\n",
                    e.id,
                    camel_to_snake(&e.name),
                    e.request,
                    e.response,
                ))
                .collect::<String>(),
        ),
    ).expect("failed to write listener.rs");

    let out_file = out_dir().join("requests.rs");
    fs::write(
        out_file,
        format!(
            r#"
impl IpcChannels {{
{}
}}
"#,
            endpoints.iter()
                .map(|e| format!(
                    r#"    #[inline]
    /// Send a [{}]
    ///
    /// - `request: F1`: request builder
    /// - `handler: F2`: [{}] handler
    pub async fn {}<T, F1, F2>(
        &self, mut request: F1, mut handler: F2,
    ) -> Result<T, IpcError>
    where F1: FnMut(&'_ mut FlatBufferBuilder<'static>) -> Result<WIPOffset<{}<'static>>, IpcError>,
          F2: FnMut({}) -> Result<T, IpcError>
    {{
        let mut builder = self.sender_sender.get_buffer();
        let root = request(&mut builder)?;
        builder.finish(root, None);
        let body = self.sender_sender.send({}, builder).await?;
        let response = flatbuffers::root::<{}>(&body).map_err(|_| IpcError::InvalidResponse)?;
        let result = handler(response);
        self.sender_sender.return_buffer(body);
        result
    }}"#,
                    e.request,
                    e.response,
                    &camel_to_snake(&e.name)[1..],
                    e.request,
                    e.response,
                    e.id,
                    e.response,
                ))
                .collect::<String>(),
        ),
    ).expect("failed to write requests.rs");
}

fn flatc_compile() {
    for file in SCHEMA_FILES.iter() {
        println!("cargo:rerun-if-changed={}", file);
    }
    let expected_version = extract_host_flatbuffers_version().expect("depends on flatbuffers");
    let flatbuffers_version = extract_flatbuffers_version().expect("depends on flatbuffers");
    if expected_version != flatbuffers_version {
        panic!("flatbuffers version mismatch: expected {}, got {}", expected_version, flatbuffers_version);
    }

    let out_dir = out_dir();
    std::fs::create_dir_all(&out_dir).unwrap();
    let aggregate_file = out_dir.join("all.fbs");
    std::fs::write(
        &aggregate_file,
        SCHEMA_FILES.iter()
            .map(|f| format!("include \"{f}\";\n"))
            .collect::<String>(),
    ).unwrap();

    let mut args = vec![
        "--rust",
        "--rust-module-root-file",
        "-I", ".",
        "-o",
        out_dir.to_str().unwrap(),
    ];
    args.extend(SCHEMA_FILES);
    args.push(aggregate_file.to_str().unwrap());
    let status = Command::new("flatc")
        .args(args)
        .status()
        .expect("flatc failed");
    assert!(status.success());
}

fn out_dir() -> PathBuf {
    Path::new(&env::var("OUT_DIR").unwrap())
        .join("flatbuffers")
}

fn workspace_cargo_toml() -> Option<PathBuf> {
    let sub = env::var("CARGO_MANIFEST_DIR").ok()?;
    Path::new(&sub).parent()?.parent()?.join("Cargo.toml").to_path_buf().into()
}

fn extract_flatbuffers_version() -> Option<usize> {
    let toml = workspace_cargo_toml()?;
    let contents = std::fs::read_to_string(toml).ok()?;
    let mut lines = contents.lines();
    for line in &mut lines {
        if line.starts_with("[workspace]") {
            break;
        }
    }
    for line in lines {
        if line.starts_with("flatbuffers = ") {
            let value = line.split('=').nth(1)?.trim().trim_matches('"');
            return value.split(".").next()?.parse().ok();
        }
    }
    None
}

fn extract_host_flatbuffers_version() -> Option<usize> {
    let out = Command::new("flatc")
        .arg("--version")
        .output()
        .ok()?
        .stdout;
    let line = str::from_utf8(&out).ok()?;
    line.split(' ').next_back()?
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn camel_to_snake(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_uppercase() { format!("_{}", c.to_ascii_lowercase()) } else { c.to_string() })
        .collect()
}
