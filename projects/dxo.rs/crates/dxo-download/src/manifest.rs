use boost_net::Url;
use serde::{Deserialize, Serialize};

use crate::error::DownloadError;

/// 一个可以提供资产字节的 HTTP 位置（主源或经核对的备用镜像）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// 绝对 `http`/`https` 地址。
    pub url: String,
    /// 人类可读的来源说明，用于诊断与验收记录。
    pub label: String,
    /// 随请求附带的额外头（例如访问令牌）；跨源重定向时由传输层负责不再转发凭据。
    #[serde(default)]
    pub headers: Vec<(String, String)>,
}

/// 单个可下载资产的锁定描述（合同 43 §4 的来源清单条目）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetSpec {
    /// 缓存键：一个安全的单路径组件，决定 `<cache_dir>/<id>/` 目录。
    pub id: String,
    /// 发布到缓存目录后的正式文件名（单个组件，不含分隔符）。
    pub filename: String,
    /// 预期字节长度；必须为正且与最终文件一致。
    pub expected_bytes: u64,
    /// 预期 SHA-256，64 位小写十六进制。
    pub sha256: String,
    /// 按优先级排列的来源；下载器依序使用镜像。
    pub sources: Vec<Source>,
}

impl AssetSpec {
    /// 校验规格的每个不变量，返回首个违规说明。
    pub fn validate(&self) -> Result<(), DownloadError> {
        check_component("id", &self.id)?;
        check_component("filename", &self.filename)?;
        if self.expected_bytes == 0 || self.expected_bytes > usize::MAX as u64 {
            return Err(DownloadError::InvalidSpec(format!(
                "`{}` expected_bytes {} out of range",
                self.id, self.expected_bytes
            )));
        }
        if self.sha256.len() != 64
            || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.sha256 != self.sha256.to_ascii_lowercase()
        {
            return Err(DownloadError::InvalidSpec(format!("`{}` sha256 must be 64 lowercase hex characters", self.id)));
        }
        if self.sources.is_empty() {
            return Err(DownloadError::InvalidSpec(format!("`{}` needs at least one source", self.id)));
        }
        for source in &self.sources {
            let url = Url::parse(&source.url)
                .map_err(|error| DownloadError::InvalidSpec(format!("`{}` source url `{}`: {error}", self.id, source.url)))?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(DownloadError::InvalidSpec(format!("`{}` source url `{}` must be http(s)", self.id, source.url)));
            }
        }
        Ok(())
    }
}

fn check_component(field: &str, value: &str) -> Result<(), DownloadError> {
    let ok = !value.is_empty()
        && value != "."
        && value != ".."
        && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(DownloadError::InvalidSpec(format!(
            "`{field}` must be a safe single path component (letters, digits, `.` `_` `-`)"
        )))
    }
}
