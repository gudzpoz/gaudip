use std::{env, path::{Path, PathBuf}, process::Command};

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
    let content = std::fs::read_to_string(RPC_SERVICE_SCHEMA).unwrap();
    struct Endpoint {
        name: String,
        request: String,
        response: String,
        id: u32,
    }
    let mut endpoints = Vec::new();
    for line in content.lines() {
        if !line.starts_with("  ") {
            continue;
        }
        // "  Ping(PingRequest):PingResponse (id: 1);"
        let info = line.split(&['(', ')', ':']).map(|s| s.trim()).collect::<Vec<_>>();
        // ["Ping", "PingRequest", "", "PingResponse", "id", "1", ";"]
        assert_eq!(info.len(), 7);
        assert_eq!(info[2], "");
        assert_eq!(info[4], "id");
        assert_eq!(info[6], ";");
        let (endpoint, request, response, id) = (info[0], info[1], info[3], info[5]);
        let id: usize = id.parse().unwrap();
        endpoints.push(Endpoint {
            name: endpoint.to_string(),
            request: request.to_string(),
            response: response.to_string(),
            id: id as u32,
        });
    }

    let out_file = out_dir().join("rpc_service.rs");
    // TODO: code-gen after we come up with an IPC inteface
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
