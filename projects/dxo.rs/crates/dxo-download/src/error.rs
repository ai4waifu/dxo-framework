use std::{fmt, io, path::PathBuf};

/// 共用下载器可能报告的全部失败类别（合同 43 §4）。
#[derive(Debug)]
pub enum DownloadError {
    /// 资产描述本身不合法：空来源、非法缓存键、摘要格式错误等。
    InvalidSpec(String),
    /// 同一缓存键已有另一个写入者持有锁。
    Busy {
        /// 冲突的锁文件路径。
        lock: PathBuf,
    },
    /// 调用方取消了下载；`.part` 与续传状态保留，可稍后继续。
    Cancelled,
    /// 本地文件系统失败。
    Io {
        /// 失败发生的环节说明。
        context: String,
        /// 底层 IO 错误。
        source: io::Error,
    },
    /// 落盘的续传状态无法解析。
    StateCorrupt(String),
    /// 服务端返回的 `Content-Range` 与续传位置或预期总长矛盾。
    ContentRange(String),
    /// 流结束时字节数与预期长度不符。
    SizeMismatch {
        /// 预期长度。
        expected: u64,
        /// 实际长度。
        actual: u64,
    },
    /// 字节数正确但 SHA-256 不符。
    ChecksumMismatch {
        /// 预期摘要（小写十六进制）。
        expected: String,
        /// 实际摘要（小写十六进制）。
        actual: String,
    },
    /// 所有镜像与尝试次数耗尽；保留可信进度，可稍后重试。
    Unreachable(String),
}

impl fmt::Display for DownloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(detail) => write!(formatter, "invalid asset spec: {detail}"),
            Self::Busy { lock } => write!(formatter, "another writer holds this cache key: {}", lock.display()),
            Self::Cancelled => formatter.write_str("download cancelled; resume state kept"),
            Self::Io { context, source } => write!(formatter, "io failure while {context}: {source}"),
            Self::StateCorrupt(detail) => write!(formatter, "resume state unreadable: {detail}"),
            Self::ContentRange(detail) => write!(formatter, "content-range violation: {detail}"),
            Self::SizeMismatch { expected, actual } => {
                write!(formatter, "expected {expected} bytes, stream delivered {actual}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(formatter, "sha256 mismatch: expected {expected}, got {actual}")
            }
            Self::Unreachable(detail) => write!(formatter, "all mirrors and attempts exhausted: {detail}"),
        }
    }
}

impl std::error::Error for DownloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// 把带环节的 IO 错误折叠进 [`DownloadError::Io`]。
pub(crate) fn io_fail(context: &str) -> impl FnOnce(io::Error) -> DownloadError + '_ {
    |source| DownloadError::Io { context: context.to_string(), source }
}
