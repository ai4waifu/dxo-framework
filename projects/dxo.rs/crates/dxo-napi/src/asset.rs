//! `.dxo` 资产容器在 Node 侧的 napi 桥（`data-binary-dxo`）。
//!
//! TS 通过 [`DxoAsset`] 打开容器、读取索引元数据、做整体验收并读取张量 / 样本字节；
//! 解析、摘要校验与分块读取全部在 Rust 内完成，不再 spawn `dxo-build` CLI。

use napi::bindgen_prelude::*;
use napi_derive::napi;

fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn map_err(err: data_binary_dxo::DxoError) -> Error {
    Error::from_reason(format!("DXO_ASSET: {err}"))
}

/// 已打开并通过结构校验的 `.dxo` 资产（主文件 + 可选分卷）。
#[napi]
#[derive(Debug)]
pub struct DxoAsset {
    inner: data_binary_dxo::Asset,
}

#[napi]
impl DxoAsset {
    /// 打开主文件，校验外壳、索引摘要与完整索引结构；分卷在首次触达或 [`DxoAsset::verify_all`] 时校验。
    #[napi(constructor)]
    pub fn open(path: String) -> Result<Self> {
        Ok(Self { inner: data_binary_dxo::Asset::open(&path).map_err(map_err)? })
    }

    /// 索引元数据 JSON 字符串：kind、格式版本、semantic 摘要、张量、分卷、dataset / weights 描述与 provenance。
    #[napi]
    pub fn metadata_json(&self) -> Result<String> {
        let index = self.inner.index();
        let kind = match index.kind {
            data_binary_dxo::Kind::Dataset => "dataset",
            data_binary_dxo::Kind::Weights => "weights",
        };
        let tensors: Vec<serde_json::Value> = index
            .tensors
            .iter()
            .map(|t| {
                let dtype = match t.dtype {
                    data_binary_dxo::Dtype::U8 => "u8",
                    data_binary_dxo::Dtype::F32 => "f32",
                };
                serde_json::json!({
                    "key": t.key,
                    "dtype": dtype,
                    "shape": t.shape,
                    "logical_bytes": t.logical_bytes,
                    "logical_sha256": hex32(&t.logical_sha256),
                    "chunks": t
                        .chunks
                        .iter()
                        .map(|r| serde_json::json!({ "chunk_id": r.chunk_id, "logical_offset": r.logical_offset }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let volumes: Vec<serde_json::Value> = index
            .volumes
            .iter()
            .map(|v| serde_json::json!({ "volume_index": v.volume_index, "file_bytes": v.file_bytes }))
            .collect();
        let dataset = index.dataset.as_ref().map(|d| {
            serde_json::json!({
                "splits": d
                    .splits
                    .iter()
                    .map(|s| serde_json::json!({
                        "name": s.name,
                        "sample_count": s.sample_count,
                        "fields": s
                            .fields
                            .iter()
                            .map(|f| serde_json::json!({ "name": f.name, "tensor_key": f.tensor_key, "role": f.role }))
                            .collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>(),
            })
        });
        let weights = index.weights.as_ref().map(|w| {
            serde_json::json!({
                "architecture_id": w.architecture_id,
                "state_dict_contract": w.state_dict_contract,
                "entries": w
                    .entries
                    .iter()
                    .map(|e| serde_json::json!({ "tensor_key": e.tensor_key, "role": e.role }))
                    .collect::<Vec<_>>(),
            })
        });
        let provenance = index.provenance.as_ref().map(|p| {
            serde_json::json!({
                "upstream_uri": p.upstream_uri,
                "revision": p.revision,
                "input_sha256": p.input_sha256,
                "license": p.license,
                "tool_version": p.tool_version,
            })
        });
        let value = serde_json::json!({
            "kind": kind,
            "format": { "major": data_binary_dxo::FORMAT_MAJOR, "minor": data_binary_dxo::FORMAT_MINOR },
            "semantic_sha256": hex32(&index.semantic_sha256),
            "volume_width": index.volume_width,
            "volumes": volumes,
            "chunk_count": index.chunks.len(),
            "tensors": tensors,
            "dataset": dataset,
            "weights": weights,
            "provenance": provenance,
            "main_path": self.inner.main_path().to_string_lossy(),
        });
        serde_json::to_string(&value).map_err(|e| Error::from_reason(format!("DXO_ASSET metadata: {e}")))
    }

    /// 整体验收：逐分卷外壳、逐 chunk 摘要与逐张量逻辑摘要全部通过才返回 `Ok`。
    #[napi]
    pub fn verify_all(&mut self) -> Result<()> {
        self.inner.verify_all().map_err(map_err)
    }

    /// 读取一个张量的完整逻辑字节（Node `Buffer`），首次触达的 chunk 会做摘要校验。
    #[napi]
    pub fn read_tensor(&mut self, key: String) -> Result<Buffer> {
        self.inner.read_tensor(&key).map(Buffer::from).map_err(map_err)
    }

    /// 读取数据集某 split 某字段的连续样本区间（行主序打包字节），只触达覆盖范围的 chunk。
    #[napi]
    pub fn read_samples(&mut self, split: String, field: String, start_sample: u32, count: u32) -> Result<Buffer> {
        self.inner
            .read_samples(&split, &field, u64::from(start_sample), u64::from(count))
            .map(Buffer::from)
            .map_err(map_err)
    }
}
