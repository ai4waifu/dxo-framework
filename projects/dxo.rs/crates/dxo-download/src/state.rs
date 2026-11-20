use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::error::{DownloadError, io_fail};

/// 单个 `.part` 的跨进程续传记录（合同 43 §4：已落盘长度、来源、强 ETag、预期长度/摘要）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeState {
    /// 状态文件格式版本。
    pub version: u32,
    /// 所属资产缓存键。
    pub id: String,
    /// 预期 SHA-256（与规格核对，防止状态串档）。
    pub sha256: String,
    /// 预期总长度。
    pub expected_bytes: u64,
    /// 上次写入进展的来源地址。
    pub source_url: String,
    /// 上次写入进展的来源序号。
    pub source_index: usize,
    /// 已落盘并通过 fsync 的字节数。
    pub committed_bytes: u64,
    /// 强 ETag（`W/` 弱摘要不记录）。
    #[serde(default)]
    pub etag: Option<String>,
    /// `Last-Modified` 值，作为 ETag 缺失时的验证器。
    #[serde(default)]
    pub last_modified: Option<String>,
}

impl ResumeState {
    /// 是否存在可放进 `If-Range` 的可靠验证器。
    pub(crate) fn has_validator(&self) -> bool {
        self.etag.is_some() || self.last_modified.is_some()
    }

    /// 原子写出状态文件（临时文件 + rename）。
    pub(crate) fn save(&self, path: &Path) -> Result<(), DownloadError> {
        let text = serde_json::to_string(self)
            .map_err(|error| DownloadError::StateCorrupt(format!("serialize resume state: {error}")))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, text).map_err(io_fail("write resume state tmp"))?;
        fs::rename(&tmp, path).map_err(io_fail("publish resume state"))
    }
}

/// 加载状态；缺失或属于其他资产规格时返回 `None`（视为从头开始）。
pub(crate) fn load_state(path: &Path) -> Result<Option<ResumeState>, DownloadError> {
    match fs::read_to_string(path) {
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(DownloadError::Io { context: "read resume state".into(), source }),
        Ok(text) => {
            let parsed: ResumeState = serde_json::from_str(&text)
                .map_err(|error| DownloadError::StateCorrupt(format!("parse {}: {error}", path.display())))?;
            Ok(Some(parsed))
        }
    }
}

/// 每个缓存键同一时间只允许一个写入者；锁文件记录 pid，过期锁可被接管。
pub(crate) struct CacheLock {
    path: PathBuf,
}

impl CacheLock {
    /// 获取目录锁；已有活跃锁时返回 [`DownloadError::Busy`]，
    /// 锁龄超过 `stale_after`（进程被强杀遗留）时删除后重试一次。
    pub(crate) fn acquire(dir: &Path, stale_after: Duration) -> Result<Self, DownloadError> {
        let path = dir.join(".write.lock");
        for attempt in 0..2u32 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = file.write_all(format!("{}", std::process::id()).as_bytes());
                    return Ok(Self { path });
                }
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                    if attempt == 0 {
                        let stale = fs::metadata(&path)
                            .and_then(|meta| meta.modified())
                            .map(|modified| modified.elapsed().unwrap_or(Duration::ZERO) > stale_after)
                            .unwrap_or(false);
                        if stale {
                            let _ = fs::remove_file(&path);
                            continue;
                        }
                    }
                    return Err(DownloadError::Busy { lock: path });
                }
                Err(source) => return Err(DownloadError::Io { context: "open cache lock".into(), source }),
            }
        }
        Err(DownloadError::Busy { lock: path })
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
