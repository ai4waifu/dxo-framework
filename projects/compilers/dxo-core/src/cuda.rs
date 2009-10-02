//! Titan CUDA facade (G4 / 0.0.6 spike).
//!
//! Compiles without the CUDA Toolkit (Driver API via `libloading`). Runtime
//! availability depends on a loaded NVIDIA driver and at least one device.

use std::sync::{Arc, Mutex, OnceLock};

use titan_backend_cuda::{gemm_f32_abi, CudaCompiler, CudaDriver, GemmF32Descriptor};
use titan_hal::{BackendDriver, DeviceSession, LaunchGeometry};
use titan_kernel::{BasicBlock, BlockId, KernelAbi, KernelArg, KernelModule};
use titan_types::{BackendId, DType, DeviceId, KernelId};

use crate::tensor::TensorError;

static CUDA_SESSION: OnceLock<Result<Arc<dyn DeviceSession>, String>> = OnceLock::new();
static GEMM_CACHE: OnceLock<Mutex<Option<CachedGemm>>> = OnceLock::new();

struct CachedGemm {
    fingerprint_key: String,
    bytes: Vec<u8>,
    abi: KernelAbi,
    metadata: titan_types::KernelLaunchMetadata,
}

fn gemm_ir(abi: KernelAbi) -> KernelModule {
    KernelModule {
        kernel_id: KernelId("gemm.f32".into()),
        entry: BlockId(0),
        blocks: vec![BasicBlock { id: BlockId(0), params: vec![], instructions: vec![] }],
        abi,
    }
}

/// Whether a Titan CUDA session can be opened on this machine.
pub fn is_available() -> bool {
    matches!(cuda_session(), Ok(_))
}

fn cuda_session() -> Result<Arc<dyn DeviceSession>, TensorError> {
    let cached = CUDA_SESSION.get_or_init(|| {
        let driver = CudaDriver::open().map_err(|e| e.to_string())?;
        let devices = driver.enumerate().map_err(|e| e.to_string())?;
        if devices.is_empty() {
            return Err("no CUDA devices enumerated".into());
        }
        driver
            .open(DeviceId { backend: BackendId::Cuda, ordinal: 0 })
            .map_err(|e| e.to_string())
    });
    match cached {
        Ok(s) => Ok(s.clone()),
        Err(msg) => Err(TensorError::Device(format!("CUDA unavailable: {msg}"))),
    }
}

fn cached_gemm(session: &Arc<dyn DeviceSession>) -> Result<CachedGemm, TensorError> {
    let fp = session.fingerprint();
    let key = format!("{:?}:{}:{}", fp.device.backend, fp.device.ordinal, fp.capability_revision);
    let slot = GEMM_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|_| TensorError::Device("CUDA gemm cache poisoned".into()))?;
    if let Some(hit) = guard.as_ref() {
        if hit.fingerprint_key == key {
            return Ok(CachedGemm {
                fingerprint_key: hit.fingerprint_key.clone(),
                bytes: hit.bytes.clone(),
                abi: hit.abi.clone(),
                metadata: hit.metadata.clone(),
            });
        }
    }
    let abi = gemm_f32_abi();
    let artifact = CudaCompiler
        .compile_artifact(&gemm_ir(abi.clone()), &abi, fp)
        .map_err(|e| TensorError::Device(format!("CUDA gemm compile: {e}")))?;
    let cached = CachedGemm {
        fingerprint_key: key,
        bytes: artifact.ptx().to_vec(),
        abi,
        metadata: artifact.metadata().clone(),
    };
    *guard = Some(CachedGemm {
        fingerprint_key: cached.fingerprint_key.clone(),
        bytes: cached.bytes.clone(),
        abi: cached.abi.clone(),
        metadata: cached.metadata.clone(),
    });
    Ok(cached)
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    if bytes.len() % 4 != 0 {
        return Err(TensorError::Device("CUDA download length not multiple of 4".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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

    let session = cuda_session()?;
    let m_u = u32::try_from(m).map_err(|_| TensorError::Device("M exceeds u32".into()))?;
    let k_u = u32::try_from(k).map_err(|_| TensorError::Device("K exceeds u32".into()))?;
    let n_u = u32::try_from(n).map_err(|_| TensorError::Device("N exceeds u32".into()))?;

    GemmF32Descriptor {
        m: m_u,
        n: n_u,
        k: k_u,
        lhs_shape: [m_u, k_u],
        rhs_shape: [k_u, n_u],
        output_shape: [m_u, n_u],
        lhs_dtype: DType::F32,
        rhs_dtype: DType::F32,
        output_dtype: DType::F32,
        lhs_contiguous: true,
        rhs_contiguous: true,
        output_contiguous: true,
        transpose_lhs: false,
        transpose_rhs: false,
    }
    .validate()
    .map_err(|e| TensorError::Device(format!("CUDA gemm contract: {e}")))?;

    let lhs_bytes = f32_bytes(lhs);
    let rhs_bytes = f32_bytes(rhs);
    let out_bytes_len = m * n * 4;

    let lhs_buf = session
        .allocate(lhs_bytes.len(), 4)
        .map_err(|e| TensorError::Device(format!("CUDA allocate lhs: {e}")))?;
    let rhs_buf = session
        .allocate(rhs_bytes.len(), 4)
        .map_err(|e| TensorError::Device(format!("CUDA allocate rhs: {e}")))?;
    let out_buf = session
        .allocate(out_bytes_len, 4)
        .map_err(|e| TensorError::Device(format!("CUDA allocate out: {e}")))?;

    let stream = session
        .create_stream()
        .map_err(|e| TensorError::Device(format!("CUDA stream: {e}")))?;
    let up_l = session
        .upload(stream.as_ref(), lhs_buf.as_ref(), &lhs_bytes)
        .map_err(|e| TensorError::Device(format!("CUDA upload lhs: {e}")))?;
    session
        .wait(up_l.as_ref())
        .map_err(|e| TensorError::Device(format!("CUDA wait lhs: {e}")))?;
    let up_r = session
        .upload(stream.as_ref(), rhs_buf.as_ref(), &rhs_bytes)
        .map_err(|e| TensorError::Device(format!("CUDA upload rhs: {e}")))?;
    session
        .wait(up_r.as_ref())
        .map_err(|e| TensorError::Device(format!("CUDA wait rhs: {e}")))?;

    let compiled = cached_gemm(&session)?;
    let args = compiled
        .abi
        .encode(&[
            KernelArg::Buffer {
                slot: 0,
                dtype: DType::F32,
                writable: false,
                alignment: 4,
                buffer: lhs_buf.clone(),
            },
            KernelArg::Buffer {
                slot: 1,
                dtype: DType::F32,
                writable: false,
                alignment: 4,
                buffer: rhs_buf.clone(),
            },
            KernelArg::Buffer {
                slot: 2,
                dtype: DType::F32,
                writable: true,
                alignment: 4,
                buffer: out_buf.clone(),
            },
            KernelArg::Scalar {
                dtype: DType::I32,
                bytes: (m_u as i32).to_le_bytes().to_vec(),
            },
            KernelArg::Scalar {
                dtype: DType::I32,
                bytes: (n_u as i32).to_le_bytes().to_vec(),
            },
            KernelArg::Scalar {
                dtype: DType::I32,
                bytes: (k_u as i32).to_le_bytes().to_vec(),
            },
        ])
        .map_err(|e| TensorError::Device(format!("CUDA abi encode: {e}")))?;

    let kernel = session
        .load(&compiled.bytes, &compiled.abi.abi_hash(), compiled.metadata.clone())
        .map_err(|e| TensorError::Device(format!("CUDA load: {e}")))?;
    let block = compiled.metadata.block[0].max(1);
    let geometry = LaunchGeometry {
        grid: [((m * n) as u32).div_ceil(block), 1, 1],
        block: compiled.metadata.block,
        shared_bytes: compiled.metadata.shared_bytes,
    };
    let event = session
        .launch(stream.as_ref(), kernel.as_ref(), &args, &geometry)
        .map_err(|e| TensorError::Device(format!("CUDA launch: {e}")))?;
    session
        .wait(event.as_ref())
        .map_err(|e| TensorError::Device(format!("CUDA wait launch: {e}")))?;

    let mut out_bytes = vec![0u8; out_bytes_len];
    let down = session
        .download(stream.as_ref(), out_buf.as_ref(), &mut out_bytes)
        .map_err(|e| TensorError::Device(format!("CUDA download: {e}")))?;
    session
        .wait(down.as_ref())
        .map_err(|e| TensorError::Device(format!("CUDA wait download: {e}")))?;

    f32_from_bytes(&out_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_cpu_when_cuda_present() {
        if !is_available() {
            eprintln!("SKIP: CUDA Driver API unavailable");
            return;
        }
        let m = 2usize;
        let k = 3usize;
        let n = 2usize;
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = gemm_f32(&lhs, m, k, &rhs, n).expect("cuda gemm");
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += lhs[i * k + p] * rhs[p * n + j];
                }
                expected[i * n + j] = sum;
            }
        }
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "index {i}: cuda={a} cpu={e}");
        }
    }
}
