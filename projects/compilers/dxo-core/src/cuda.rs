//! CUDA GEMM adapter — dispatches through `titan-runtime` only.
//!
//! DXO must not compile kernels, cache artifacts, or orchestrate upload/launch/readback.
//! Device session bootstrap still uses `CudaDriver::open` until Titan exposes
//! `Runtime::open_cuda_primary()` (see handoff in workspace design docs).

use std::sync::{Mutex, OnceLock};

use titan_backend_cuda::CudaDriver;
use titan_graph::{EffectContract, OpRequest, TensorSpec};
use titan_hal::BackendDriver;
use titan_runtime::Runtime;
use titan_tensor::{Device, Tensor};
use titan_types::{AliasContract, AttrMap, DType, Layout, MemoryEffect, OperatorId, Shape, SourceSpan, Strides};

use crate::tensor::TensorError;

struct CudaState {
    runtime: Mutex<Runtime>,
    device: Device,
}

static CUDA_STATE: OnceLock<Result<CudaState, String>> = OnceLock::new();

fn dxo_source() -> SourceSpan {
    SourceSpan { file: "dxo-core".into(), line: 0, column: 0 }
}

fn gemm_request(lhs: titan_tensor::TensorHandle, rhs: titan_tensor::TensorHandle, output_shape: [u64; 2]) -> OpRequest {
    OpRequest {
        operator: OperatorId("gemm".into()),
        inputs: vec![lhs, rhs],
        outputs: vec![TensorSpec {
            dtype: DType::F32,
            strides: Strides(vec![output_shape[1] as i64, 1]),
            shape: Shape(output_shape.to_vec()),
            layout: Layout::Contiguous,
            alias: AliasContract::NoAlias,
        }],
        attrs: AttrMap::new(),
        effects: EffectContract { memory: MemoryEffect::Writes, deterministic: true },
        source: dxo_source(),
    }
}

fn cuda_state() -> Result<&'static CudaState, TensorError> {
    let init = CUDA_STATE.get_or_init(|| {
        let driver = CudaDriver::open().map_err(|e| e.to_string())?;
        let devices = driver.enumerate().map_err(|e| e.to_string())?;
        let fingerprint = devices.first().ok_or_else(|| "no CUDA devices enumerated".to_string())?;
        let session = driver.open(fingerprint.device).map_err(|e| e.to_string())?;
        let device = Device::from_session(session);
        let cache_path = std::env::temp_dir().join("dxo-titan-runtime-cache");
        let runtime = Runtime::open(cache_path);
        Ok(CudaState { runtime: Mutex::new(runtime), device })
    });
    init.as_ref().map_err(|msg| TensorError::Device(format!("CUDA unavailable: {msg}")))
}

/// Whether a Titan CUDA session can be opened on this machine.
pub fn is_available() -> bool {
    cuda_state().is_ok()
}

/// Contiguous row-major f32 GEMM on CUDA: `A[m,k] @ B[k,n] -> C[m,n]`.
pub fn gemm_f32(lhs: &[f32], m: usize, k: usize, rhs: &[f32], n: usize) -> Result<Vec<f32>, TensorError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TensorError::Shape("CUDA gemm requires non-zero M, K, N".into()));
    }
    if lhs.len() != m * k || rhs.len() != k * n {
        return Err(TensorError::Shape(format!(
            "CUDA gemm length mismatch: lhs={} (expect {}), rhs={} (expect {})",
            lhs.len(),
            m * k,
            rhs.len(),
            k * n
        )));
    }

    let state = cuda_state()?;
    let cuda_lhs = Tensor::<f32, 2>::from_slice(&state.device, [m, k], lhs)
        .map_err(|e| TensorError::Device(format!("CUDA upload lhs: {e}")))?;
    let cuda_rhs = Tensor::<f32, 2>::from_slice(&state.device, [k, n], rhs)
        .map_err(|e| TensorError::Device(format!("CUDA upload rhs: {e}")))?;

    let mut runtime = state.runtime.lock().map_err(|_| TensorError::Device("CUDA runtime lock poisoned".into()))?;

    let handle = runtime
        .execute(gemm_request(cuda_lhs.handle(), cuda_rhs.handle(), [m as u64, n as u64]))
        .map_err(|e| TensorError::Device(format!("CUDA gemm dispatch: {e}")))?
        .wait()
        .map_err(|e| TensorError::Device(format!("CUDA gemm completion: {e}")))?;

    handle
        .outputs
        .first()
        .ok_or_else(|| TensorError::Device("CUDA gemm returned no outputs".into()))?
        .to_vec_f32()
        .map_err(|e| TensorError::Device(format!("CUDA gemm readback: {e}")))
}
