//! Browser WASM ABI for `@dxo/lite`.
//!
//! Exports opaque init + f32 kernels consumed by `@dxo/lite-unknown-wasm32`.
//! Does **not** own WebGPU devices; Titan wgpu dispatch lands later via `dxo-core`.

#![warn(missing_docs)]

use wasm_bindgen::prelude::*;

/// Package / ABI identity (not npm semver).
#[wasm_bindgen]
pub fn version() -> String {
    "dxo-lite-wasm@0.1.0-preview".to_string()
}

/// True when this module is the interim host-f32 path (not Titan wgpu yet).
#[wasm_bindgen(js_name = isInterimHostF32)]
pub fn is_interim_host_f32() -> bool {
    true
}

/// Row-major f32 matmul for rank-2 tensors: `(m,k) × (k,n) → (m,n)`.
///
/// `a` length must be `m * k`; `b` length must be `k * n`.
#[wasm_bindgen(js_name = matmulF32)]
pub fn matmul_f32(a: &[f32], a_rows: u32, a_cols: u32, b: &[f32], b_cols: u32) -> Result<Vec<f32>, JsValue> {
    let m = a_rows as usize;
    let k = a_cols as usize;
    let n = b_cols as usize;
    if a.len() != m * k {
        return Err(JsValue::from_str(&format!("matmul a length {} != {}×{}", a.len(), m, k)));
    }
    if b.len() != k * n {
        return Err(JsValue::from_str(&format!("matmul b length {} != {}×{}", b.len(), k, n)));
    }
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Ok(out)
}

/// Element-wise f32 add; lengths must match.
#[wasm_bindgen(js_name = addF32)]
pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, JsValue> {
    if a.len() != b.len() {
        return Err(JsValue::from_str(&format!("add length mismatch: {} vs {}", a.len(), b.len())));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}
