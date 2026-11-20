//! 合同 43 §4 验收：本地 HTTP 故障夹具。
//! 中途断流、忽略 Range、错误范围、416、429、5xx、坏镜像、摘要不符、离线复用、锁互斥、
//! 状态与实际长度对账、取消保留进度，均不允许发布错误字节。

use std::{
    io::{BufRead, BufReader, Write},
    net::{Shutdown, TcpListener},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use dxo_download::{AssetSpec, Cancel, DownloadConfig, DownloadError, Downloader, ResumeState, Source};
use sha2::{Digest, Sha256};

const FULL_LEN: usize = 100_000;
const HALF: usize = 40_000;

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    headers: Vec<(String, String)>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }
}

struct Fixture {
    port: u16,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Fixture {
    fn spawn<H>(handler: H) -> Self
    where
        H: Fn(&Seen, usize) -> Vec<u8> + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture");
        let port = listener.local_addr().expect("fixture addr").port();
        listener.set_nonblocking(true).expect("nonblocking accept");
        let stop = Arc::new(AtomicBool::new(false));
        let stop_loop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let mut connection = 0usize;
            loop {
                if stop_loop.load(Ordering::SeqCst) {
                    break;
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let index = connection;
                        connection += 1;
                        let Some(request) = read_request(&mut stream) else { continue };
                        let response = handler(&request, index);
                        let _ = stream.write_all(&response);
                        let _ = stream.shutdown(Shutdown::Both);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(5)),
                    Err(_) => break,
                }
            }
        });
        Self { port, stop, handle: Some(handle) }
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<Seen> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let path = line.split_whitespace().nth(1)?.to_string();
    let mut headers = Vec::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    Some(Seen { path, headers })
}

fn http(status: &str, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut text = format!("HTTP/1.1 {status}\r\n");
    for (name, value) in headers {
        text.push_str(&format!("{name}: {value}\r\n"));
    }
    text.push_str("connection: close\r\n\r\n");
    let mut out = text.into_bytes();
    out.extend_from_slice(body);
    out
}

fn serve_200(full: &[u8], etag: &str) -> Vec<u8> {
    http("200 OK", &[("etag", etag.into()), ("content-length", full.len().to_string())], full)
}

fn serve_206(full: &[u8], seen: &Seen) -> Vec<u8> {
    let range = seen.header("range").unwrap_or("bytes=0-");
    let start = range_start(range);
    let rest = &full[start as usize..];
    http(
        "206 Partial Content",
        &[
            ("etag", "\"v1\"".into()),
            ("content-range", format!("bytes {}-{}/{}", start, full.len() - 1, full.len())),
            ("content-length", rest.len().to_string()),
        ],
        rest,
    )
}

fn range_start(range: &str) -> usize {
    let body = range.strip_prefix("bytes=").expect("range form");
    body.split('-').next().expect("range start").parse().expect("range number")
}

fn payload() -> Arc<[u8]> {
    Arc::from((0..FULL_LEN).map(|index| (index as u64 * 31 + 7) as u8).collect::<Vec<u8>>())
}

fn sha_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn test_spec(id: &str, urls: &[String], full: &[u8]) -> AssetSpec {
    AssetSpec {
        id: id.into(),
        filename: "blob.bin".into(),
        expected_bytes: full.len() as u64,
        sha256: sha_hex(full),
        sources: urls.iter().map(|url| Source { url: url.clone(), label: "fixture".into(), headers: Vec::new() }).collect(),
    }
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_cache(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("dxo-dl-{tag}-{}-{}", std::process::id(), COUNTER.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp cache dir");
    dir
}

fn test_config(dir: &Path) -> DownloadConfig {
    DownloadConfig {
        max_attempts: 2,
        max_attempts_per_mirror: 2,
        backoff_initial: Duration::from_millis(10),
        backoff_max: Duration::from_millis(50),
        retry_after_cap: Duration::from_secs(2),
        chunk_bytes: 8 * 1024,
        request_timeout: Duration::from_secs(5),
        stream_idle_timeout: Duration::from_secs(5),
        ..DownloadConfig::new(dir)
    }
}

fn asset_dir(dir: &Path, id: &str) -> PathBuf {
    dir.join(id)
}

/// 第一次运行在 40% 处断流并失败；第二次全新进程（新 Downloader）从正确偏移续传。
#[test]
fn resume_after_midstream_disconnect_across_processes() {
    let full = payload();
    let served = Arc::clone(&full);
    let cut = http("200 OK", &[("etag", "\"v1\"".into()), ("content-length", FULL_LEN.to_string())], &full[..HALF]);
    let dir = temp_cache("resume");
    let fixture = Fixture::spawn(move |seen, index| {
        if index == 0 {
            cut.clone()
        } else {
            assert_eq!(seen.header("range"), Some("bytes=40000-"));
            assert_eq!(seen.header("if-range"), Some("\"v1\""));
            assert_eq!(seen.header("accept-encoding"), Some("identity"));
            serve_206(&served, seen)
        }
    });
    let spec = test_spec("resume", &[fixture.url("/blob")], &full);
    let mut config = test_config(&dir);
    config.max_attempts = 1;
    let error = Downloader::new(config).fetch(&spec, &Cancel::new()).expect_err("single attempt must fail mid-stream");
    assert!(matches!(error, DownloadError::Unreachable(_)), "{error}");
    let part = asset_dir(&dir, "resume").join("blob.bin.part");
    assert_eq!(std::fs::metadata(&part).expect("part kept").len(), HALF as u64);

    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("resume completes");
    assert_eq!(report.resumed_from, HALF as u64);
    assert_eq!(report.bytes_transferred, (FULL_LEN - HALF) as u64);
    assert!(!report.from_cache);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
    assert!(!part.exists());
    assert!(!asset_dir(&dir, "resume").join("blob.bin.state.json").exists());
}

/// 服务端忽略 Range 回 200：必须从头重写，不拼接旧字节。
#[test]
fn server_ignoring_range_rewrites_from_scratch() {
    let full = payload();
    let served = Arc::clone(&full);
    let cut = http("200 OK", &[("etag", "\"v1\"".into()), ("content-length", FULL_LEN.to_string())], &full[..HALF]);
    let dir = temp_cache("ignore");
    let fixture = Fixture::spawn(move |seen, index| {
        if index == 0 {
            cut.clone()
        } else {
            assert!(seen.header("range").is_some(), "downloader should attempt resume");
            serve_200(&served, "\"v2\"")
        }
    });
    let spec = test_spec("ignore", &[fixture.url("/blob")], &full);
    let mut config = test_config(&dir);
    config.max_attempts = 1;
    let _ = Downloader::new(config).fetch(&spec, &Cancel::new());

    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("200 rewrites from scratch");
    assert_eq!(report.resumed_from, 0);
    assert_eq!(report.bytes_transferred, FULL_LEN as u64);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 错误 Content-Range 起点不得追加；换镜像后从头取正确字节。
#[test]
fn wrong_content_range_never_appends() {
    let full = payload();
    let dir = temp_cache("badrange");
    let cut = http("200 OK", &[("etag", "\"v1\"".into()), ("content-length", FULL_LEN.to_string())], &full[..HALF]);
    let first = Fixture::spawn(move |_seen, _index| cut.clone());
    let spec = test_spec("badrange", &[first.url("/blob")], &full);
    let _ = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new());

    let lie = http(
        "206 Partial Content",
        &[
            ("etag", "\"v1\"".into()),
            ("content-range", format!("bytes {}-{}/{}", HALF - 1, FULL_LEN - 1, FULL_LEN)),
            ("content-length", (FULL_LEN - HALF + 1).to_string()),
        ],
        &full[HALF - 1..],
    );
    let served = Arc::clone(&full);
    let second = Fixture::spawn(move |seen, index| {
        if index == 0 {
            lie.clone()
        } else {
            assert_eq!(seen.header("range"), None, "mirror switch must restart");
            serve_200(&served, "\"v1\"")
        }
    });
    let spec = test_spec("badrange", &[second.url("/blob"), second.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("falls back to a fresh download");
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

fn plant_resume(dir: &Path, id: &str, url: &str, bytes: &[u8], committed: u64) {
    let dir = asset_dir(dir, id);
    std::fs::create_dir_all(&dir).expect("asset dir");
    std::fs::write(dir.join("blob.bin.part"), bytes).expect("plant part");
    let state = ResumeState {
        version: 1,
        id: id.into(),
        sha256: sha_hex(&payload()),
        expected_bytes: FULL_LEN as u64,
        source_url: url.into(),
        source_index: 0,
        committed_bytes: committed,
        etag: Some("\"v1\"".into()),
        last_modified: None,
    };
    std::fs::write(dir.join("blob.bin.state.json"), serde_json::to_vec(&state).expect("serialize state")).expect("plant state");
}

/// 416 且现有长度与全文件摘要都正确：直接收尾发布。
#[test]
fn complete_416_finalizes() {
    let full = payload();
    let dir = temp_cache("fin416");
    let fixture = Fixture::spawn(|_seen, _index| {
        http(
            "416 Range Not Satisfiable",
            &[("content-range", format!("bytes */{FULL_LEN}")), ("content-length", "0".into())],
            &[],
        )
    });
    plant_resume(&dir, "fin416", &fixture.url("/blob"), &full, FULL_LEN as u64);
    let spec = test_spec("fin416", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("416 with matching digest finalizes");
    assert_eq!(report.attempts, 1);
    assert_eq!(report.bytes_transferred, 0);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 416 但摘要不符：重置后重新下载。
#[test]
fn corrupt_416_resets_and_redownloads() {
    let full = payload();
    let dir = temp_cache("bad416");
    let wrong = vec![0u8; FULL_LEN];
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, index| {
        if index == 0 {
            http(
                "416 Range Not Satisfiable",
                &[("content-range", format!("bytes */{FULL_LEN}")), ("content-length", "0".into())],
                &[],
            )
        } else {
            assert_eq!(seen.header("range"), None, "reset must restart");
            serve_200(&served, "\"v1\"")
        }
    });
    plant_resume(&dir, "bad416", &fixture.url("/blob"), &wrong, FULL_LEN as u64);
    let spec = test_spec("bad416", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("416 with wrong digest restarts");
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 429 遵守 Retry-After 后重试成功。
#[test]
fn rate_limit_honors_retry_after() {
    let full = payload();
    let dir = temp_cache("retry429");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |_seen, index| {
        if index == 0 {
            http("429 Too Many Requests", &[("retry-after", "1".into()), ("content-length", "0".into())], &[])
        } else {
            serve_200(&served, "\"v1\"")
        }
    });
    let spec = test_spec("retry429", &[fixture.url("/blob")], &full);
    let started = Instant::now();
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("429 then success");
    assert!(started.elapsed() >= Duration::from_millis(900), "must wait out Retry-After");
    assert_eq!(report.attempts, 2);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 暂时性 5xx 退避后重试成功。
#[test]
fn transient_5xx_retries() {
    let full = payload();
    let dir = temp_cache("err5xx");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |_seen, index| {
        if index == 0 {
            http("503 Service Unavailable", &[("content-length", "0".into())], &[])
        } else {
            serve_200(&served, "\"v1\"")
        }
    });
    let spec = test_spec("err5xx", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("503 then success");
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 坏镜像 404 后切换备用源，且切换后从头下载。
#[test]
fn mirror_failover_on_permanent_status() {
    let full = payload();
    let dir = temp_cache("failover");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, _index| {
        if seen.path == "/bad" {
            http("404 Not Found", &[("content-length", "0".into())], &[])
        } else {
            assert_eq!(seen.header("range"), None);
            assert_eq!(seen.header("x-token"), Some("s3cret"));
            serve_200(&served, "\"v1\"")
        }
    });
    let mut spec = test_spec("failover", &[fixture.url("/bad"), fixture.url("/good")], &full);
    spec.sources[1].headers.push(("x-token".into(), "s3cret".into()));
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("second mirror serves");
    assert_eq!(report.attempts, 2);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 摘要不符绝不发布，报可诊断错误。
#[test]
fn checksum_mismatch_never_publishes() {
    let full = payload();
    let mut corrupted = full.to_vec();
    corrupted[1234] ^= 0xff;
    let dir = temp_cache("badsum");
    let fixture = Fixture::spawn(move |_seen, _index| serve_200(&corrupted, "\"v1\""));
    let spec = test_spec("badsum", &[fixture.url("/blob")], &full);
    let error = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect_err("corrupt bytes must not publish");
    assert!(matches!(error, DownloadError::ChecksumMismatch { .. }), "{error}");
    assert!(!asset_dir(&dir, "badsum").join("blob.bin").exists());
}

/// 已验证发布件（含人工离线副本）无网复用。
#[test]
fn verified_cache_hit_needs_no_network() {
    let full = payload();
    let dir = temp_cache("cached");
    let asset = asset_dir(&dir, "cached");
    std::fs::create_dir_all(&asset).expect("asset dir");
    std::fs::write(asset.join("blob.bin"), &full).expect("offline copy");
    let spec = test_spec("cached", &["http://127.0.0.1:1/unreachable".into()], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("cache hit");
    assert!(report.from_cache);
    assert_eq!(report.attempts, 0);
}

/// 发布件损坏时删除重下，不静默使用。
#[test]
fn corrupt_published_copy_is_redownloaded() {
    let full = payload();
    let dir = temp_cache("redl");
    let asset = asset_dir(&dir, "redl");
    std::fs::create_dir_all(&asset).expect("asset dir");
    std::fs::write(asset.join("blob.bin"), vec![9u8; FULL_LEN]).expect("corrupt copy");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |_seen, _index| serve_200(&served, "\"v1\""));
    let spec = test_spec("redl", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("redownload replaces corrupt copy");
    assert!(!report.from_cache);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 同一缓存键只允许一个写入者。
#[test]
fn single_writer_per_cache_key() {
    let full = payload();
    let dir = temp_cache("lock");
    let asset = asset_dir(&dir, "lock");
    std::fs::create_dir_all(&asset).expect("asset dir");
    std::fs::write(asset.join(".write.lock"), b"999999").expect("plant lock");
    let spec = test_spec("lock", &["http://127.0.0.1:1/x".into()], &full);
    let error =
        Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect_err("busy lock must reject second writer");
    assert!(matches!(error, DownloadError::Busy { .. }), "{error}");
}

/// 文件长度超出已提交记录：截断多余尾部后按记录偏移续传。
#[test]
fn file_ahead_of_state_is_truncated() {
    let full = payload();
    let dir = temp_cache("ahead");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, _index| {
        assert_eq!(seen.header("range"), Some("bytes=40000-"), "must resume at committed length");
        serve_206(&served, seen)
    });
    plant_resume(&dir, "ahead", &fixture.url("/blob"), &full[..HALF + 10_000], HALF as u64);
    let spec = test_spec("ahead", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("truncates then resumes");
    assert_eq!(report.resumed_from, HALF as u64);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 记录长度超过文件实际长度：不可信，整体重置从头下载。
#[test]
fn state_ahead_of_file_resets() {
    let full = payload();
    let dir = temp_cache("stale");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, _index| {
        assert_eq!(seen.header("range"), None, "stale state must restart");
        serve_200(&served, "\"v1\"")
    });
    plant_resume(&dir, "stale", &fixture.url("/blob"), &full[..HALF], 80_000);
    let spec = test_spec("stale", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("restarts after reconciliation");
    assert_eq!(report.resumed_from, 0);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 没有可靠验证器（无 ETag/Last-Modified）时重启该文件。
#[test]
fn missing_validator_forces_restart() {
    let full = payload();
    let dir = temp_cache("noval");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, _index| {
        assert_eq!(seen.header("range"), None, "no validator means no resume");
        serve_200(&served, "\"v1\"")
    });
    plant_resume(&dir, "noval", &fixture.url("/blob"), &full[..HALF], HALF as u64);
    let state_path = asset_dir(&dir, "noval").join("blob.bin.state.json");
    let mut state: ResumeState = serde_json::from_slice(&std::fs::read(&state_path).expect("state")).expect("parse");
    state.etag = None;
    std::fs::write(&state_path, serde_json::to_vec(&state).expect("serialize")).expect("rewrite");
    let spec = test_spec("noval", &[fixture.url("/blob")], &full);
    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("restart without validator");
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 取消保留可信进度，稍后可继续。
#[test]
fn cancel_keeps_resume_state() {
    let full = payload();
    let cut = http("200 OK", &[("etag", "\"v1\"".into()), ("content-length", FULL_LEN.to_string())], &full[..30_000]);
    let dir = temp_cache("cancel");
    let served = Arc::clone(&full);
    let fixture = Fixture::spawn(move |seen, index| if index == 0 { cut.clone() } else { serve_206(&served, seen) });
    let spec = test_spec("cancel", &[fixture.url("/blob")], &full);
    let cancel = Cancel::new();
    let handle = cancel.clone();
    let progress = move |done: u64, _total: u64| {
        if done >= 20_000 {
            handle.cancel();
        }
    };
    let error =
        Downloader::new(test_config(&dir)).fetch_with_progress(&spec, &cancel, Some(&progress)).expect_err("cancel must abort");
    assert!(matches!(error, DownloadError::Cancelled), "{error}");
    let state_path = asset_dir(&dir, "cancel").join("blob.bin.state.json");
    let state: ResumeState = serde_json::from_slice(&std::fs::read(&state_path).expect("kept state")).expect("parse");
    assert!((20_000..=30_000).contains(&state.committed_bytes));

    let report = Downloader::new(test_config(&dir)).fetch(&spec, &Cancel::new()).expect("later run continues");
    assert_eq!(report.resumed_from, state.committed_bytes);
    assert_eq!(std::fs::read(&report.path).expect("published"), &full[..]);
}

/// 规格校验拒绝不合法输入。
#[test]
fn invalid_specs_are_rejected() {
    let full = payload();
    let dir = temp_cache("invalid");
    let mut spec = test_spec("ok", &["https://example.invalid/blob".into()], &full);
    spec.validate().expect("baseline valid");

    spec.sha256 = "XYZ".into();
    assert!(matches!(spec.validate(), Err(DownloadError::InvalidSpec(_))));

    spec = test_spec("bad id/../..", &["https://example.invalid/blob".into()], &full);
    assert!(spec.validate().is_err());

    spec = test_spec("nosources", &[], &full);
    assert!(spec.validate().is_err());

    spec = test_spec("ftp", &["ftp://example.invalid/blob".into()], &full);
    assert!(spec.validate().is_err());
    let _ = dir;
}
