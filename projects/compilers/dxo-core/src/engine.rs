//! Titan-backed CPU engine facade.
//!
//! Host tensors in this slice still use shared CPU storage; the session proves
//! the Titan HAL boundary is wired for later device-buffer uploads.

use std::sync::{Arc, OnceLock};

use titan_backend_cpu::CpuDriver;
use titan_hal::{BackendDriver, DeviceSession};
use titan_types::{BackendId, DeviceId};

use crate::tensor::TensorError;

static CPU_SESSION: OnceLock<Arc<dyn DeviceSession>> = OnceLock::new();

/// Open (once) the shared Titan CPU `DeviceSession`.
pub fn cpu_session() -> Arc<dyn DeviceSession> {
    CPU_SESSION
        .get_or_init(|| {
            Arc::new(CpuDriver).open(DeviceId { backend: BackendId::Cpu, ordinal: 0 }).expect("titan CpuDriver session")
        })
        .clone()
}

/// Stable backend label surfaced to diagnostics.
pub const fn backend_label() -> &'static str {
    "titan-cpu"
}

/// Exercise Titan HAL `wait_event` (upload stream → compute stream) via the CPU session.
///
/// This proves DXO consumes the HAL dependency primitive rather than substituting a host
/// `await`. CUDA cross-stream proof remains a separate machine-gated check.
pub fn probe_event_dep() -> Result<(), TensorError> {
    let session = cpu_session();
    let upload_stream = session
        .create_stream()
        .map_err(|e| TensorError::Device(format!("create_stream: {e}")))?;
    let compute_stream = session
        .create_stream()
        .map_err(|e| TensorError::Device(format!("create_stream: {e}")))?;
    let buf = session
        .allocate(16, 4)
        .map_err(|e| TensorError::Device(format!("allocate: {e}")))?;
    let bytes = 1.0f32.to_le_bytes();
    let upload_event = session
        .upload(upload_stream.as_ref(), buf.as_ref(), &bytes)
        .map_err(|e| TensorError::Device(format!("upload: {e}")))?;
    session
        .wait_event(compute_stream.as_ref(), upload_event.as_ref())
        .map_err(|e| TensorError::Device(format!("wait_event: {e}")))?;
    session
        .wait(upload_event.as_ref())
        .map_err(|e| TensorError::Device(format!("wait: {e}")))?;
    Ok(())
}
