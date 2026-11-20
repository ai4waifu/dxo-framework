use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

use boost_net::{
    NetError, Url,
    http::{HttpTransport, HttpTransportConfig, Method, ProxyConfig, Request},
};
use sha2::{Digest, Sha256};

use crate::{
    error::{DownloadError, io_fail},
    httpdate::parse_retry_after,
    manifest::{AssetSpec, Source},
    state::{CacheLock, ResumeState, load_state},
};

/// 协作式取消：在分块边界生效，保留 `.part` 与续传状态。
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// 新建未取消的句柄。
    pub fn new() -> Self {
        Self::default()
    }

    /// 发出取消请求。
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// 是否已请求取消。
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// 下载进度回调（已落盘字节数，预期总字节数）。
pub type Progress<'a> = Option<&'a (dyn Fn(u64, u64) + Sync)>;

/// 下载器策略配置。
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// 下载缓存根，惯例为 `<项目根>/.cache/dxo/downloads`。
    pub cache_dir: PathBuf,
    /// 整次 `fetch` 的总 HTTP 尝试上限。
    pub max_attempts: u32,
    /// 同一镜像连续瞬时故障多少次后切换下一个镜像。
    pub max_attempts_per_mirror: u32,
    /// 首次退避时长。
    pub backoff_initial: Duration,
    /// 退避时长上限。
    pub backoff_max: Duration,
    /// `Retry-After` 的等待上限。
    pub retry_after_cap: Duration,
    /// 每次状态提交的字节粒度。
    pub chunk_bytes: usize,
    /// 连接、发送与等待响应头的超时。
    pub request_timeout: Duration,
    /// 响应正文两次读取之间的空闲超时。
    pub stream_idle_timeout: Duration,
    /// 传输层跟随重定向的上限。
    pub max_redirects: u32,
    /// 锁龄超过该值视为进程强杀遗留，允许接管。
    pub stale_lock_after: Duration,
    /// 显式代理；`None` 时若环境配置了代理可用 [`ProxyConfig::from_env`] 取值填入。
    pub proxy: Option<ProxyConfig>,
}

impl DownloadConfig {
    /// 以缓存根目录建立默认策略（合同 43 §4 的保守参数）。
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            max_attempts: 8,
            max_attempts_per_mirror: 3,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(30),
            retry_after_cap: Duration::from_secs(120),
            chunk_bytes: 256 * 1024,
            request_timeout: Duration::from_secs(30),
            stream_idle_timeout: Duration::from_secs(60),
            max_redirects: 8,
            stale_lock_after: Duration::from_secs(3600),
            proxy: None,
        }
    }
}

/// 一次 `fetch` 的结果与验收记录所需统计。
#[derive(Debug, Clone)]
pub struct DownloadReport {
    /// 发布后的正式缓存文件路径。
    pub path: PathBuf,
    /// 是否命中已验证缓存（含人工放入的离线副本），未走网络。
    pub from_cache: bool,
    /// 本次运行首个成功续传流的起始偏移；未续传为 0。
    pub resumed_from: u64,
    /// 本次运行实际从网络接收的字节数。
    pub bytes_transferred: u64,
    /// 本次运行的 HTTP 尝试次数。
    pub attempts: u32,
    /// 已验证的 SHA-256。
    pub sha256: String,
}

/// DXO 共用断点续传下载器：Boost 管 HTTPS/代理/流，这里管镜像顺序、重试、断点、校验与缓存发布。
#[derive(Debug, Clone)]
pub struct Downloader {
    /// 生效的策略。
    config: DownloadConfig,
}

/// 单次 HTTP 尝试的结局。
enum Outcome {
    Published,
    Transient { retry_after: Option<Duration> },
    SwitchMirror,
}

/// 一个缓存键下的文件布局。
struct Paths {
    dir: PathBuf,
    final_path: PathBuf,
    part_path: PathBuf,
    state_path: PathBuf,
}

impl Downloader {
    /// 以给定策略构造下载器。
    pub fn new(config: DownloadConfig) -> Self {
        Self { config }
    }

    /// 下载并校验一个资产；已验证缓存命中时不触网。
    pub fn fetch(&self, spec: &AssetSpec, cancel: &Cancel) -> Result<DownloadReport, DownloadError> {
        self.fetch_with_progress(spec, cancel, None)
    }

    /// 带进度回调的 [`Self::fetch`]。
    pub fn fetch_with_progress(
        &self,
        spec: &AssetSpec,
        cancel: &Cancel,
        progress: Progress<'_>,
    ) -> Result<DownloadReport, DownloadError> {
        spec.validate()?;
        let paths = self.paths(spec);
        fs::create_dir_all(&paths.dir).map_err(io_fail("create cache directory"))?;

        // 已验证缓存（或人工离线副本）可无网复用；摘要不符的发布件按损坏处理重下。
        if paths.final_path.exists() {
            if self.verify_file(&paths.final_path, spec)? {
                return Ok(DownloadReport {
                    path: paths.final_path,
                    from_cache: true,
                    resumed_from: 0,
                    bytes_transferred: 0,
                    attempts: 0,
                    sha256: spec.sha256.clone(),
                });
            }
            fs::remove_file(&paths.final_path).map_err(io_fail("remove corrupt published file"))?;
        }

        let _lock = CacheLock::acquire(&paths.dir, self.config.stale_lock_after)?;
        let mut state = self.load_and_reconcile(&paths, spec)?;

        let mut attempts = 0u32;
        let mut backoff = self.config.backoff_initial;
        let mut cursor = state.as_ref().map(|state| state.source_index).unwrap_or(0);
        let mut mirror_transient = 0u32;
        let mut transferred = 0u64;
        let mut resumed_from: Option<u64> = None;
        let mut last_detail: Option<String> = None;
        let mut last_checksum: Option<(String, String)> = None;

        loop {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            if attempts >= self.config.max_attempts {
                if let Some((expected, actual)) = last_checksum {
                    return Err(DownloadError::ChecksumMismatch { expected, actual });
                }
                return Err(DownloadError::Unreachable(last_detail.take().unwrap_or_else(|| "no attempt completed".into())));
            }
            attempts += 1;
            let source_index = cursor % spec.sources.len();
            let source = &spec.sources[source_index];
            let outcome = self.attempt(
                spec,
                source,
                source_index,
                &paths,
                &mut state,
                cancel,
                progress,
                &mut transferred,
                &mut resumed_from,
                &mut last_detail,
                &mut last_checksum,
            )?;
            match outcome {
                Outcome::Published => {
                    let _ = fs::remove_file(&paths.state_path);
                    return Ok(DownloadReport {
                        path: paths.final_path,
                        from_cache: false,
                        resumed_from: resumed_from.unwrap_or(0),
                        bytes_transferred: transferred,
                        attempts,
                        sha256: spec.sha256.clone(),
                    });
                }
                Outcome::Transient { retry_after } => {
                    mirror_transient += 1;
                    let wait = retry_after.map_or(backoff, |wait| wait.min(self.config.retry_after_cap));
                    backoff = (backoff * 2).min(self.config.backoff_max);
                    if mirror_transient >= self.config.max_attempts_per_mirror && spec.sources.len() > 1 {
                        // 换镜像才从头；只有一个来源时保留可信进度继续退避重试。
                        cursor += 1;
                        mirror_transient = 0;
                        self.reset(&paths, &mut state)?;
                    }
                    self.sleep_cancellable(wait, cancel)?;
                }
                Outcome::SwitchMirror => {
                    cursor += 1;
                    mirror_transient = 0;
                    backoff = self.config.backoff_initial;
                    self.reset(&paths, &mut state)?;
                }
            }
        }
    }

    fn paths(&self, spec: &AssetSpec) -> Paths {
        let dir = self.config.cache_dir.join(&spec.id);
        let final_path = dir.join(&spec.filename);
        Paths {
            part_path: dir.join(format!("{}.part", spec.filename)),
            state_path: dir.join(format!("{}.state.json", spec.filename)),
            final_path,
            dir,
        }
    }

    fn load_and_reconcile(&self, paths: &Paths, spec: &AssetSpec) -> Result<Option<ResumeState>, DownloadError> {
        let mut state = match load_state(&paths.state_path)? {
            Some(state)
                if state.version == 1
                    && state.id == spec.id
                    && state.sha256 == spec.sha256
                    && state.expected_bytes == spec.expected_bytes
                    && state.source_index < spec.sources.len() =>
            {
                Some(state)
            }
            Some(_) => {
                self.reset(paths, &mut None)?;
                return Ok(None);
            }
            None => None,
        };
        // 以实际文件长度对账：文件超出已提交长度则截断多余尾部，短于记录则整体重置。
        let actual = match fs::metadata(&paths.part_path) {
            Ok(meta) => meta.len(),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => 0,
            Err(source) => return Err(DownloadError::Io { context: "stat .part".into(), source }),
        };
        match state.as_mut() {
            None => {
                if actual > 0 {
                    truncate_to(&paths.part_path, 0)?;
                }
            }
            Some(state) => {
                if actual < state.committed_bytes {
                    self.reset(paths, &mut None)?;
                    return Ok(None);
                }
                if actual > state.committed_bytes {
                    truncate_to(&paths.part_path, state.committed_bytes)?;
                }
            }
        }
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn attempt(
        &self,
        spec: &AssetSpec,
        source: &Source,
        source_index: usize,
        paths: &Paths,
        state: &mut Option<ResumeState>,
        cancel: &Cancel,
        progress: Progress<'_>,
        transferred: &mut u64,
        resumed_from: &mut Option<u64>,
        last_detail: &mut Option<String>,
        last_checksum: &mut Option<(String, String)>,
    ) -> Result<Outcome, DownloadError> {
        // 续传仅当：同一镜像、已有落盘、且持有可靠验证器；否则从头写该文件。
        let resume = match state.as_ref() {
            Some(state)
                if state.committed_bytes > 0
                    && state.source_index == source_index
                    && state.source_url == source.url
                    && state.has_validator() =>
            {
                Some(state.clone())
            }
            _ => None,
        };
        if resume.is_none() && state.as_ref().is_some_and(|state| state.committed_bytes > 0) {
            self.reset(paths, state)?;
        }

        let url = Url::parse(&source.url)
            .map_err(|error| DownloadError::InvalidSpec(format!("source url `{}`: {error}", source.url)))?;
        let origin = source.url.clone();
        let mut request = Request::new(Method::Get, url).with_header("accept-encoding", "identity");
        if let Some(resume) = &resume {
            request = request.with_header("range", format!("bytes={}-", resume.committed_bytes));
            if let Some(validator) = resume.etag.as_ref().or(resume.last_modified.as_ref()) {
                request = request.with_header("if-range", validator.clone());
            }
        }
        for (name, value) in &source.headers {
            request = request.with_header(name.clone(), value.clone());
        }

        let transport = HttpTransport::with_config(HttpTransportConfig {
            max_body_bytes: spec.expected_bytes as usize,
            max_redirects: self.config.max_redirects,
            timeout: self.config.request_timeout,
            stream_idle_timeout: self.config.stream_idle_timeout,
            max_header_bytes: 64 * 1024,
            tls: Default::default(),
            proxy: self.config.proxy.clone(),
        });

        let mut response = match transport.send_stream(&request) {
            Ok(response) => response,
            Err(NetError::Unsupported(detail)) => {
                return Err(DownloadError::InvalidSpec(format!("source `{}`: {detail}", source.url)));
            }
            Err(error) => {
                *last_detail = Some(format!("{origin}: {error}"));
                return Ok(Outcome::Transient { retry_after: None });
            }
        };

        match response.status {
            200 | 206 => {}
            416 => return self.handle_416(spec, paths, state, last_detail),
            429 => {
                *last_detail = Some(format!("{origin}: HTTP 429"));
                return Ok(Outcome::Transient { retry_after: retry_after(&response.headers) });
            }
            500..=599 => {
                *last_detail = Some(format!("{origin}: HTTP {}", response.status));
                return Ok(Outcome::Transient { retry_after: retry_after(&response.headers) });
            }
            status => {
                *last_detail = Some(format!("{origin}: HTTP {status}"));
                return Ok(Outcome::SwitchMirror);
            }
        }

        let mut record = ResumeState {
            version: 1,
            id: spec.id.clone(),
            sha256: spec.sha256.clone(),
            expected_bytes: spec.expected_bytes,
            source_url: source.url.clone(),
            source_index,
            committed_bytes: 0,
            etag: response.headers.get("etag").filter(|etag| !etag.starts_with("W/")).map(str::to_string),
            last_modified: response.headers.get("last-modified").map(str::to_string),
        };

        let mut hash = Sha256::new();
        let mut file: File;
        if response.status == 206 {
            let Some(resume) = &resume else {
                *last_detail = Some(format!("{origin}: 206 without a resume request"));
                return Ok(Outcome::SwitchMirror);
            };
            let detail = match self.open_for_append(paths, resume, spec, &response.headers, &mut hash) {
                Ok(handle) => handle,
                Err(detail) => {
                    *last_detail = Some(format!("{origin}: {detail}"));
                    return Ok(Outcome::SwitchMirror);
                }
            };
            file = detail;
            record.committed_bytes = resume.committed_bytes;
            if resume.committed_bytes > 0 && resumed_from.is_none() {
                *resumed_from = Some(resume.committed_bytes);
            }
        } else {
            // 200：从头写（包括服务端忽略 Range 的情况）。
            file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&paths.part_path)
                .map_err(io_fail("open .part for fresh write"))?;
        }

        let mut committed = record.committed_bytes;
        let mut buffer = vec![0u8; self.config.chunk_bytes.max(4096)];
        loop {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            match response.body.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    file.write_all(&buffer[..read]).map_err(io_fail("write .part"))?;
                    hash.update(&buffer[..read]);
                    committed += read as u64;
                    *transferred += read as u64;
                    file.flush().map_err(io_fail("flush .part"))?;
                    file.sync_all().map_err(io_fail("sync .part"))?;
                    record.committed_bytes = committed;
                    record.save(&paths.state_path)?;
                    *state = Some(record.clone());
                    if let Some(progress) = progress {
                        progress(committed, spec.expected_bytes);
                    }
                }
                Err(NetError::BodyTooLarge { limit }) => {
                    *last_detail = Some(format!("{origin}: body exceeds {limit} bytes"));
                    return Ok(Outcome::SwitchMirror);
                }
                Err(error) => {
                    *last_detail = Some(format!("{origin}: stream broke: {error}"));
                    return Ok(Outcome::Transient { retry_after: None });
                }
            }
        }

        if committed < spec.expected_bytes {
            *last_detail = Some(format!("{origin}: early end of stream at {committed} bytes"));
            return Ok(Outcome::Transient { retry_after: None });
        }
        if committed > spec.expected_bytes {
            *last_detail = Some(format!("{origin}: oversize stream at {committed} bytes"));
            return Ok(Outcome::SwitchMirror);
        }
        let actual = hex_digest(hash.finalize().as_slice());
        if actual != spec.sha256 {
            *last_detail =
                Some(DownloadError::ChecksumMismatch { expected: spec.sha256.clone(), actual: actual.clone() }.to_string());
            *last_checksum = Some((spec.sha256.clone(), actual));
            return Ok(Outcome::SwitchMirror);
        }
        drop(file);
        self.publish(paths)?;
        *state = None;
        Ok(Outcome::Published)
    }

    /// 校验 206 的 `Content-Range` 与 `.part` 现状，重放已落盘前缀的哈希，返回可追加的文件句柄。
    fn open_for_append(
        &self,
        paths: &Paths,
        resume: &ResumeState,
        spec: &AssetSpec,
        headers: &boost_net::http::Headers,
        hash: &mut Sha256,
    ) -> Result<File, String> {
        let value = headers.get("content-range").ok_or_else(|| "206 without Content-Range".to_string())?;
        let (start, end, total) = parse_content_range(value).ok_or_else(|| format!("unparsable Content-Range `{value}`"))?;
        if start != resume.committed_bytes {
            return Err(format!("Content-Range starts at {start}, resume position is {}", resume.committed_bytes));
        }
        if end < start {
            return Err(format!("Content-Range ends before it starts: `{value}`"));
        }
        if total.is_some_and(|total| total != spec.expected_bytes) {
            return Err(format!("Content-Range total {:?} contradicts expected {}", total, spec.expected_bytes));
        }
        let actual_len = fs::metadata(&paths.part_path).map_err(|error| format!("stat .part: {error}"))?.len();
        if actual_len != start {
            return Err(format!(".part length {actual_len} contradicts resume position {start}"));
        }
        let mut file =
            OpenOptions::new().read(true).write(true).open(&paths.part_path).map_err(|error| format!("open .part: {error}"))?;
        let mut buffer = vec![0u8; self.config.chunk_bytes.max(4096)];
        loop {
            let read = file.read(&mut buffer).map_err(|error| format!("rehash .part: {error}"))?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        file.seek(SeekFrom::End(0)).map_err(|error| format!("seek .part: {error}"))?;
        Ok(file)
    }

    /// `416`：仅当现有长度与全文件摘要都正确才收尾，否则重置该文件。
    fn handle_416(
        &self,
        spec: &AssetSpec,
        paths: &Paths,
        state: &mut Option<ResumeState>,
        last_detail: &mut Option<String>,
    ) -> Result<Outcome, DownloadError> {
        let committed = state.as_ref().map_or(0, |state| state.committed_bytes);
        if committed > 0 && committed == spec.expected_bytes && self.verify_file(&paths.part_path, spec)? {
            self.publish(paths)?;
            *state = None;
            return Ok(Outcome::Published);
        }
        *last_detail = Some("server reported 416 for a stale resume position".into());
        self.reset(paths, state)?;
        Ok(Outcome::Transient { retry_after: None })
    }

    fn verify_file(&self, path: &Path, spec: &AssetSpec) -> Result<bool, DownloadError> {
        let meta = fs::metadata(path).map_err(io_fail("stat candidate file"))?;
        if meta.len() != spec.expected_bytes {
            return Ok(false);
        }
        let mut file = File::open(path).map_err(io_fail("open candidate file"))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(io_fail("hash candidate file"))?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        Ok(hex_digest(hash.finalize().as_slice()) == spec.sha256)
    }

    /// 完整 SHA-256 验证通过后原子发布。
    fn publish(&self, paths: &Paths) -> Result<(), DownloadError> {
        fs::rename(&paths.part_path, &paths.final_path).map_err(io_fail("atomically publish .part"))?;
        let _ = fs::remove_file(&paths.state_path);
        Ok(())
    }

    /// 从头开始：清空 `.part`、删除状态与内存记录。
    fn reset(&self, paths: &Paths, state: &mut Option<ResumeState>) -> Result<(), DownloadError> {
        *state = None;
        if paths.part_path.exists() {
            truncate_to(&paths.part_path, 0)?;
        }
        let _ = fs::remove_file(&paths.state_path);
        Ok(())
    }

    fn sleep_cancellable(&self, wait: Duration, cancel: &Cancel) -> Result<(), DownloadError> {
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            if cancel.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            let remaining = deadline - Instant::now();
            thread::sleep(remaining.min(Duration::from_millis(50)));
        }
        Ok(())
    }
}

fn truncate_to(path: &Path, len: u64) -> Result<(), DownloadError> {
    let file = OpenOptions::new().write(true).create(true).open(path).map_err(io_fail("truncate .part"))?;
    file.set_len(len).map_err(io_fail("set .part length"))
}

fn retry_after(headers: &boost_net::http::Headers) -> Option<Duration> {
    headers.get("retry-after").and_then(|value| parse_retry_after(value, SystemTime::now()))
}

/// 解析 `bytes start-end/total`（`total` 可为 `*`）。
fn parse_content_range(value: &str) -> Option<(u64, u64, Option<u64>)> {
    let rest = value.strip_prefix("bytes")?.trim();
    let (span, total) = rest.split_once('/')?;
    let (start, end) = span.split_once('-')?;
    let total = match total.trim() {
        "*" => None,
        other => Some(other.parse::<u64>().ok()?),
    };
    Some((start.trim().parse::<u64>().ok()?, end.trim().parse::<u64>().ok()?, total))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_range_with_total() {
        assert_eq!(parse_content_range("bytes 40000-99999/100000"), Some((40000, 99999, Some(100000))));
        assert_eq!(parse_content_range("bytes 0-9/*"), Some((0, 9, None)));
        assert_eq!(parse_content_range("items 0-9/10"), None);
        assert_eq!(parse_content_range("bytes abc-9/10"), None);
    }

    #[test]
    fn hex_digest_lowercases() {
        assert_eq!(hex_digest(&[0x0a, 0xff]), "0aff");
    }
}
