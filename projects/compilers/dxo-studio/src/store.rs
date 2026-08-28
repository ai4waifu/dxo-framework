//! Append-only inspect run store (read path).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const EVENTS_FILE: &str = "events.jsonl";
const META_FILE: &str = "meta.json";
const ARTIFACTS_DIR: &str = "artifacts";

/// Default runs root: `{cwd}/.dxo/runs` or `DXO_RUNS_DIR`.
pub fn default_runs_root() -> PathBuf {
    if let Ok(from_env) = std::env::var("DXO_RUNS_DIR") {
        let trimmed = from_env.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(".dxo").join("runs")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunMetaV0 {
    pub format: String,
    pub version: u32,
    pub run_id: String,
    pub started_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparams: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    pub run_id: String,
    pub meta: RunMetaV0,
}

fn run_dir(root: &Path, run_id: &str) -> PathBuf {
    root.join(run_id)
}

/// List runs sorted by `startedAtMs` descending.
pub fn list_runs(root: &Path) -> std::io::Result<Vec<RunSummary>> {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let run_id = entry.file_name().to_string_lossy().into_owned();
        if let Some(meta) = read_run_meta(root, &run_id)? {
            out.push(RunSummary { run_id, meta });
        }
    }
    out.sort_by(|a, b| b.meta.started_at_ms.cmp(&a.meta.started_at_ms));
    Ok(out)
}

pub fn read_run_meta(root: &Path, run_id: &str) -> std::io::Result<Option<RunMetaV0>> {
    let path = run_dir(root, run_id).join(META_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    serde_json::from_str(&text).map(Some).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

pub fn read_events(root: &Path, run_id: &str) -> std::io::Result<Vec<serde_json::Value>> {
    let path = run_dir(root, run_id).join(EVENTS_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        out.push(value);
    }
    Ok(out)
}

pub fn read_artifact_text(root: &Path, run_id: &str, rel: &str) -> std::io::Result<String> {
    let abs = safe_run_file(root, run_id, rel)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "artifact path escapes run dir"))?;
    fs::read_to_string(abs)
}

pub fn read_artifact_bytes(root: &Path, run_id: &str, rel: &str) -> std::io::Result<Vec<u8>> {
    let abs = safe_run_file(root, run_id, rel)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "artifact path escapes run dir"))?;
    fs::read(abs)
}

pub fn safe_run_file(root: &Path, run_id: &str, rel: &str) -> Option<PathBuf> {
    if rel.contains("..") {
        return None;
    }
    Some(run_dir(root, run_id).join(rel.replace('/', std::path::MAIN_SEPARATOR_STR)))
}

pub fn confusion_matrix_path(root: &Path, run_id: &str) -> PathBuf {
    run_dir(root, run_id).join(ARTIFACTS_DIR).join("confusion-matrix.json")
}

pub fn image_samples_path(root: &Path, run_id: &str) -> PathBuf {
    run_dir(root, run_id).join(ARTIFACTS_DIR).join("image-samples.json")
}

pub fn read_json_artifact(path: &Path) -> std::io::Result<serde_json::Value> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

pub fn mime_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

pub fn model_graph_path(root: &Path, run_id: &str) -> PathBuf {
    run_dir(root, run_id).join(ARTIFACTS_DIR).join("model-graph.json")
}

pub fn profile_trace_path(root: &Path, run_id: &str) -> PathBuf {
    run_dir(root, run_id).join(ARTIFACTS_DIR).join("profile-trace.json")
}

pub fn metrics_from_events(events: &[serde_json::Value]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| {
            if event.get("type")?.as_str()? != "metric/scalar" {
                return None;
            }
            event.get("metric").cloned()
        })
        .collect()
}

pub fn artifacts_from_events(events: &[serde_json::Value]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| {
            if event.get("type")?.as_str()? != "artifact/ref" {
                return None;
            }
            event.get("artifact").cloned()
        })
        .collect()
}

pub fn checkpoints_from_events(events: &[serde_json::Value]) -> Vec<serde_json::Value> {
    artifacts_from_events(events)
        .into_iter()
        .filter(|artifact| artifact.get("kind").and_then(|k| k.as_str()) == Some("checkpoint"))
        .collect()
}
