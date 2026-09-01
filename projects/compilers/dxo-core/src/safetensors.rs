//! Safetensors F32 codec (HF layout) for DXO checkpoints and weight IO.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::tensor::TensorError;

const DTYPE_F32: &str = "F32";
const HEADER_ALIGN: usize = 8;
const METADATA_KEY: &str = "__metadata__";

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn align_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

/// Decoded safetensors payload.
pub type DecodedSafetensors = (BTreeMap<String, SafetensorSlice>, BTreeMap<String, String>);

/// Named f32 tensor slice for safetensors encode/decode.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorSlice {
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Host f32 payload (length == product(shape)).
    pub data: Vec<f32>,
}

/// Named f32 tensor as raw bytes (avoids an extra f32 vec on the encode hot path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorBufferSlice {
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Little-endian f32 bytes (length == product(shape) * 4).
    pub data: Vec<u8>,
}

impl SafetensorBufferSlice {
    fn validate(&self, name: &str) -> Result<(), TensorError> {
        if self.data.len() % 4 != 0 {
            return Err(TensorError::invalid_shape(format!(
                "safetensors '{name}': byte length {} is not a multiple of 4",
                self.data.len()
            )));
        }
        let expected = numel(&self.shape) * 4;
        if self.data.len() != expected {
            return Err(TensorError::invalid_shape(format!(
                "safetensors '{name}': byte length {} != product(shape)*4={expected}",
                self.data.len()
            )));
        }
        Ok(())
    }
}

fn f32_bytes_to_vec(data: &[u8]) -> Result<Vec<f32>, TensorError> {
    if data.len() % 4 != 0 {
        return Err(TensorError::invalid_shape("safetensors: f32 byte length must be a multiple of 4"));
    }
    data.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().map_err(|_| TensorError::invalid_shape("safetensors: bad f32 chunk"))?;
            Ok(f32::from_le_bytes(arr))
        })
        .collect()
}

/// Encode named F32 tensors (+ optional `__metadata__`) to safetensors bytes.
pub fn encode_safetensors_buffers(
    tensors: &BTreeMap<String, SafetensorBufferSlice>,
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<Vec<u8>, TensorError> {
    if tensors.is_empty() {
        return Err(TensorError::invalid_shape("encode_safetensors: empty tensor map"));
    }

    let mut header = Map::new();
    let mut data_chunks: Vec<&[u8]> = Vec::new();
    let mut offset = 0usize;

    for (name, slice) in tensors {
        slice.validate(name)?;
        let byte_len = slice.data.len();
        let mut entry = Map::new();
        entry.insert("dtype".into(), Value::String(DTYPE_F32.into()));
        entry.insert("shape".into(), Value::Array(slice.shape.iter().map(|&d| Value::from(d as u64)).collect()));
        entry.insert(
            "data_offsets".into(),
            Value::Array(vec![Value::from(offset as u64), Value::from((offset + byte_len) as u64)]),
        );
        header.insert(name.clone(), Value::Object(entry));
        data_chunks.push(slice.data.as_slice());
        offset += byte_len;
    }

    if let Some(meta) = metadata {
        let mut meta_obj = Map::new();
        for (k, v) in meta {
            meta_obj.insert(k.clone(), Value::String(v.clone()));
        }
        header.insert(METADATA_KEY.into(), Value::Object(meta_obj));
    }

    let header_json =
        serde_json::to_vec(&header).map_err(|e| TensorError::invalid_shape(format!("encode_safetensors: header JSON: {e}")))?;
    let header_len = align_up(header_json.len(), HEADER_ALIGN);
    let mut header_padded = vec![0x20u8; header_len];
    header_padded[..header_json.len()].copy_from_slice(&header_json);

    let mut out = Vec::with_capacity(8 + header_len + offset);
    out.extend_from_slice(&(header_len as u64).to_le_bytes());
    out.extend_from_slice(&header_padded);
    for chunk in data_chunks {
        out.extend_from_slice(chunk);
    }
    Ok(out)
}

/// Decode safetensors bytes into named F32 slices and optional metadata map.
pub fn decode_safetensors(bytes: &[u8]) -> Result<DecodedSafetensors, TensorError> {
    if bytes.len() < 8 {
        return Err(TensorError::invalid_shape("decode_safetensors: buffer too short"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if header_len == 0 || 8 + header_len > bytes.len() {
        return Err(TensorError::invalid_shape(format!("decode_safetensors: invalid header length {header_len}")));
    }

    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .map_err(|e| TensorError::invalid_shape(format!("decode_safetensors: bad header JSON: {e}")))?;
    let obj = header.as_object().ok_or_else(|| TensorError::invalid_shape("decode_safetensors: header must be an object"))?;

    let data_base = 8 + header_len;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();

    for (name, info) in obj {
        if name == METADATA_KEY {
            if let Some(meta_obj) = info.as_object() {
                for (k, v) in meta_obj {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let info = info
            .as_object()
            .ok_or_else(|| TensorError::invalid_shape(format!("decode_safetensors: invalid entry '{name}'")))?;
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorError::invalid_shape(format!("decode_safetensors: missing dtype for '{name}'")))?;
        if dtype != DTYPE_F32 {
            return Err(TensorError::invalid_shape(format!(
                "decode_safetensors: unsupported dtype '{dtype}' for '{name}' (v0 is F32-only)"
            )));
        }
        let shape = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorError::invalid_shape(format!("decode_safetensors: missing shape for '{name}'")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| TensorError::invalid_shape(format!("decode_safetensors: invalid shape for '{name}'")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorError::invalid_shape(format!("decode_safetensors: missing data_offsets for '{name}'")))?;
        if offsets.len() != 2 {
            return Err(TensorError::invalid_shape(format!("decode_safetensors: invalid data_offsets for '{name}'")));
        }
        let begin =
            offsets[0].as_u64().ok_or_else(|| TensorError::invalid_shape("decode_safetensors: invalid offset"))? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| TensorError::invalid_shape("decode_safetensors: invalid offset"))? as usize;
        if end < begin || data_base + end > bytes.len() {
            return Err(TensorError::invalid_shape(format!("decode_safetensors: bad data_offsets for '{name}'")));
        }
        let expected = numel(&shape) * 4;
        if end - begin != expected {
            return Err(TensorError::invalid_shape(format!(
                "decode_safetensors: '{name}' byte span {} != {expected}",
                end - begin
            )));
        }
        let raw = &bytes[data_base + begin..data_base + end];
        tensors.insert(name.clone(), SafetensorSlice { shape, data: f32_bytes_to_vec(raw)? });
    }

    Ok((tensors, metadata))
}

/// Convenience encode from host f32 vectors (tests and non-napi callers).
pub fn encode_safetensors(
    tensors: &BTreeMap<String, SafetensorSlice>,
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<Vec<u8>, TensorError> {
    let buffers = tensors
        .iter()
        .map(|(name, slice)| {
            let mut data = Vec::with_capacity(slice.data.len() * 4);
            for f in &slice.data {
                data.extend_from_slice(&f.to_le_bytes());
            }
            (name.clone(), SafetensorBufferSlice { shape: slice.shape.clone(), data })
        })
        .collect();
    encode_safetensors_buffers(&buffers, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_metadata() {
        let mut tensors = BTreeMap::new();
        tensors.insert("weight".into(), SafetensorSlice { shape: vec![2, 2], data: vec![1.0, 2.0, 3.0, 4.0] });
        let mut meta = BTreeMap::new();
        meta.insert("dxo.checkpoint.version".into(), "1".into());
        meta.insert("epoch".into(), "3".into());
        let bytes = encode_safetensors(&tensors, Some(&meta)).expect("encode");
        let (decoded, got_meta) = decode_safetensors(&bytes).expect("decode");
        assert_eq!(decoded, tensors);
        assert_eq!(got_meta.get("epoch"), Some(&"3".into()));
    }
}
