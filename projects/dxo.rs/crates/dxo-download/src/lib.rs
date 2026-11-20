//! DXO 共用断点续传下载器（合同 43 §4）。
//!
//! 链路为 `@dxo/hub → 本下载器 → boost.net`：Boost 管 HTTPS、显式代理、连接与响应流；
//! 这里管镜像顺序、重试退避、跨进程断点续传、SHA-256 校验与缓存原子发布。
//!
//! 缓存布局：`<cache_dir>/<id>/<filename>` 为发布件，`.part` 加 `.state.json` 承载
//! 续传进度；`.write.lock` 保证同一缓存键同一时间只有一个写入者。
//! 已验证发布件（含人工放入的离线副本）可无网复用；任何未通过完整校验的字节都不会发布。

mod downloader;
mod error;
mod httpdate;
mod manifest;
mod state;

pub use downloader::{Cancel, DownloadConfig, DownloadReport, Downloader, Progress};
pub use error::DownloadError;
pub use manifest::{AssetSpec, Source};
pub use state::ResumeState;

use std::path::{Path, PathBuf};

/// 项目根下约定的下载缓存目录（合同 43 §4）。
pub fn default_cache_dir(project_root: &Path) -> PathBuf {
    project_root.join(".cache").join("dxo").join("downloads")
}
