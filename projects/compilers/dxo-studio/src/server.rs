//! Loopback HTTP API over the inspect run store.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::store::{
    artifacts_from_events, checkpoints_from_events, confusion_matrix_path, image_samples_path, list_runs, metrics_from_events,
    mime_for_path, model_graph_path, profile_trace_path, read_artifact_bytes, read_artifact_text, read_events, read_json_artifact,
    read_run_meta, safe_run_file,
};

/// Options for {@link InspectApiServer::bind}.
#[derive(Debug, Clone)]
pub struct InspectApiOptions {
    pub host: String,
    pub port: u16,
    pub runs_root: PathBuf,
}

impl Default for InspectApiOptions {
    fn default() -> Self {
        Self { host: "127.0.0.1".to_string(), port: 4310, runs_root: crate::store::default_runs_root() }
    }
}

/// Bound inspect HTTP server handle.
pub struct InspectApiServer {
    pub host: String,
    pub port: u16,
    pub url: String,
    pub runs_root: PathBuf,
    stop: Arc<AtomicBool>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl InspectApiServer {
    /// Bind loopback HTTP API. `port == 0` picks a free port.
    pub fn bind(options: InspectApiOptions) -> std::io::Result<Self> {
        let addr = format!("{}:{}", options.host, options.port);
        let listener = TcpListener::bind(&addr)?;
        listener.set_nonblocking(true)?;
        let bound = listener.local_addr()?;
        let host = bound.ip().to_string();
        let port = bound.port();
        let url = format!("http://{host}:{port}");
        let runs_root = options.runs_root;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let runs_for_thread = runs_root.clone();

        let join = thread::spawn(move || serve_loop(listener, runs_for_thread, stop_thread));

        Ok(Self { host, port, url, runs_root, stop, join: Mutex::new(Some(join)) })
    }

    /// Stop the server thread.
    pub fn close(self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.lock().ok().and_then(|mut g| g.take()) {
            let _ = join.join();
        }
    }
}

enum ApiResponse {
    Json { status: u16, body: serde_json::Value },
    Bytes { status: u16, content_type: String, body: Vec<u8> },
}

fn serve_loop(listener: TcpListener, runs_root: PathBuf, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let root = runs_root.clone();
                let _ = handle_connection(stream, &root);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }
}

fn handle_connection(mut stream: TcpStream, runs_root: &Path) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let request = read_request(&mut stream)?;
    let response = route_request(&request, runs_root);
    write_response(&mut stream, response)
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut buf = [0_u8; 8192];
    let n = stream.read(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf[..n]).into_owned())
}

fn split_path_query(path: &str) -> (&str, &str) {
    match path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (path, ""),
    }
}

fn parse_runs_query(query: &str) -> Vec<String> {
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            if key == "runs" && !value.is_empty() {
                return value.split(',').map(urlencoding_decode).filter(|s| !s.is_empty()).collect();
            }
        }
    }
    Vec::new()
}

fn route_request(request: &str, runs_root: &Path) -> ApiResponse {
    let first_line = request.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_path = parts.next().unwrap_or("/");
    let (path, query) = split_path_query(raw_path);

    if method != "GET" {
        return ApiResponse::Json { status: 405, body: serde_json::json!({ "error": "method_not_allowed" }) };
    }

    if path == "/api/health" {
        return ApiResponse::Json { status: 200, body: serde_json::json!({ "ok": true, "service": "dxo-studio-api" }) };
    }

    if path == "/api/runs" {
        return match list_runs(runs_root) {
            Ok(runs) => ApiResponse::Json { status: 200, body: serde_json::json!({ "runs": runs }) },
            Err(err) => ApiResponse::Json {
                status: 500,
                body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
            },
        };
    }

    if path == "/api/compare" {
        return handle_compare(query, runs_root);
    }

    if let Some(rest) = path.strip_prefix("/api/runs/") {
        return handle_run_path(rest, runs_root);
    }

    ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) }
}

fn handle_compare(query: &str, runs_root: &Path) -> ApiResponse {
    let run_ids = parse_runs_query(query);
    if run_ids.is_empty() {
        return ApiResponse::Json {
            status: 400,
            body: serde_json::json!({ "error": "invalid_request", "message": "missing runs query param" }),
        };
    }

    let mut series = Vec::new();
    for run_id in run_ids {
        let meta = match read_run_meta(runs_root, &run_id) {
            Ok(Some(m)) => m,
            Ok(None) => {
                return ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found", "runId": run_id }) };
            }
            Err(err) => {
                return ApiResponse::Json {
                    status: 500,
                    body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
                };
            }
        };
        let events = match read_events(runs_root, &run_id) {
            Ok(e) => e,
            Err(err) => {
                return ApiResponse::Json {
                    status: 500,
                    body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
                };
            }
        };
        series.push(serde_json::json!({
            "runId": run_id,
            "meta": meta,
            "metrics": metrics_from_events(&events),
        }));
    }

    ApiResponse::Json { status: 200, body: serde_json::json!({ "series": series }) }
}

fn handle_run_path(rest: &str, runs_root: &Path) -> ApiResponse {
    let (run_id, sub) = match rest.split_once('/') {
        Some((id, tail)) => (id, format!("/{tail}")),
        None => (rest, String::new()),
    };
    let run_id = urlencoding_decode(run_id);

    if sub.is_empty() || sub == "/meta" {
        return match read_run_meta(runs_root, &run_id) {
            Ok(Some(meta)) => ApiResponse::Json { status: 200, body: serde_json::json!({ "meta": meta }) },
            Ok(None) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
            Err(err) => ApiResponse::Json {
                status: 500,
                body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
            },
        };
    }

    if sub.starts_with("/files/") {
        let rel = urlencoding_decode(sub.trim_start_matches("/files/"));
        return match read_artifact_bytes(runs_root, &run_id, &rel) {
            Ok(bytes) => {
                let mime = safe_run_file(runs_root, &run_id, &rel)
                    .map(|p| mime_for_path(&p).to_string())
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                ApiResponse::Bytes { status: 200, content_type: mime, body: bytes }
            }
            Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
        };
    }

    let events = match read_events(runs_root, &run_id) {
        Ok(events) => events,
        Err(err) => {
            return ApiResponse::Json {
                status: 500,
                body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
            };
        }
    };

    match sub.as_str() {
        "/events" => ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "events": events }) },
        "/metrics" => ApiResponse::Json {
            status: 200,
            body: serde_json::json!({ "runId": run_id, "metrics": metrics_from_events(&events) }),
        },
        "/artifacts" => ApiResponse::Json {
            status: 200,
            body: serde_json::json!({ "runId": run_id, "artifacts": artifacts_from_events(&events) }),
        },
        "/checkpoints" => ApiResponse::Json {
            status: 200,
            body: serde_json::json!({ "runId": run_id, "checkpoints": checkpoints_from_events(&events) }),
        },
        "/confusion-matrix" => match read_json_artifact(&confusion_matrix_path(runs_root, &run_id)) {
            Ok(data) => {
                ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "confusionMatrix": data }) }
            }
            Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
        },
        "/image-samples" => match read_json_artifact(&image_samples_path(runs_root, &run_id)) {
            Ok(data) => ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "imageSamples": data }) },
            Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
        },
        "/model-graph" => match std::fs::read_to_string(model_graph_path(runs_root, &run_id)) {
            Ok(body) => match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(graph) => ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "graph": graph }) },
                Err(err) => ApiResponse::Json {
                    status: 500,
                    body: serde_json::json!({ "error": "internal_error", "message": err.to_string() }),
                },
            },
            Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
        },
        "/profile-trace" => match read_json_artifact(&profile_trace_path(runs_root, &run_id)) {
            Ok(data) => ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "profileTrace": data }) },
            Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
        },
        other if other.starts_with("/artifacts/") => {
            let rel = urlencoding_decode(other.trim_start_matches("/artifacts/"));
            match read_artifact_text(runs_root, &run_id, &rel) {
                Ok(body) => {
                    ApiResponse::Json { status: 200, body: serde_json::json!({ "runId": run_id, "path": rel, "body": body }) }
                }
                Err(_) => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
            }
        }
        _ => ApiResponse::Json { status: 404, body: serde_json::json!({ "error": "not_found" }) },
    }
}

fn urlencoding_decode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                out.push(byte as char);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn write_response(stream: &mut TcpStream, response: ApiResponse) -> std::io::Result<()> {
    match response {
        ApiResponse::Json { status, body } => write_json_response(stream, status, &body),
        ApiResponse::Bytes { status, content_type, body } => write_bytes_response(stream, status, &content_type, &body),
    }
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: &serde_json::Value) -> std::io::Result<()> {
    let text = body.to_string();
    let reason = status_reason(status);
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json; charset=utf-8\r\naccess-control-allow-origin: *\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}",
        text.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_bytes_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = status_reason(status);
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\naccess-control-allow-origin: *\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::default_runs_root;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[test]
    fn health_endpoint() {
        let server =
            InspectApiServer::bind(InspectApiOptions { host: "127.0.0.1".into(), port: 0, runs_root: default_runs_root() })
                .expect("bind");
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", server.port)).expect("connect");
        stream.write_all(b"GET /api/health HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut buf = String::new();
        stream.read_to_string(&mut buf).expect("read");
        assert!(buf.contains("\"ok\":true"));
        server.close();
    }

    #[test]
    fn lists_runs_from_store() {
        let root = std::env::temp_dir().join(format!("dxo-studio-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("run-a")).expect("mkdir");
        fs::write(
            root.join("run-a/meta.json"),
            r#"{"format":"dxo-inspect","version":0,"runId":"run-a","startedAtMs":1,"status":"ok"}"#,
        )
        .expect("meta");

        let server = InspectApiServer::bind(InspectApiOptions { host: "127.0.0.1".into(), port: 0, runs_root: root.clone() })
            .expect("bind");
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", server.port)).expect("connect");
        stream.write_all(b"GET /api/runs HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut buf = String::new();
        stream.read_to_string(&mut buf).expect("read");
        assert!(buf.contains("run-a"));
        server.close();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compare_and_wave2_artifacts() {
        let root = std::env::temp_dir().join(format!("dxo-studio-wave2-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("run-a/artifacts")).expect("mkdir");
        fs::write(
            root.join("run-a/meta.json"),
            r#"{"format":"dxo-inspect","version":0,"runId":"run-a","startedAtMs":1,"label":"a","status":"ok"}"#,
        )
        .expect("meta");
        fs::write(
            root.join("run-a/artifacts/image-samples.json"),
            r#"{"samples":[{"uri":"artifacts/samples/x.png","label":0,"pred":1}]}"#,
        )
        .expect("samples");
        fs::create_dir_all(root.join("run-a/artifacts/samples")).expect("samples dir");
        fs::write(root.join("run-a/artifacts/samples/x.png"), &[137, 80, 78, 71]).expect("png");

        let server = InspectApiServer::bind(InspectApiOptions { host: "127.0.0.1".into(), port: 0, runs_root: root.clone() })
            .expect("bind");
        let port = server.port;

        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        stream.write_all(b"GET /api/runs/run-a/confusion-matrix HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut buf = String::new();
        stream.read_to_string(&mut buf).expect("read");
        assert!(buf.contains("cat"));

        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        stream.write_all(b"GET /api/compare?runs=run-a HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut buf = String::new();
        stream.read_to_string(&mut buf).expect("read");
        assert!(buf.contains("series"));

        server.close();
        let _ = fs::remove_dir_all(root);
    }
}
