//! Titan-backed CPU engine facade + CUDA event probe entry points.

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

/// Stable backend label for user-facing diagnostics (`cpu` / `cuda` / … — never Titan brand).
pub const fn backend_label() -> &'static str {
    "cpu"
}

/// Exercise Titan HAL `wait_event` (upload stream → compute stream) via the CPU session.
pub fn probe_event_dep() -> Result<(), TensorError> {
    let session = cpu_session();
    let upload_stream = session.create_stream().map_err(|e| TensorError::from_hal(e, "cpu"))?;
    let compute_stream = session.create_stream().map_err(|e| TensorError::from_hal(e, "cpu"))?;
    let buf = session.allocate(16, 4).map_err(|e| TensorError::from_hal(e, "cpu"))?;
    let bytes = 1.0f32.to_le_bytes();
    let upload_event = session
        .upload(upload_stream.as_ref(), buf.as_ref(), &bytes)
        .map_err(|e| TensorError::from_hal(e, "cpu"))?;
    session
        .wait_event(compute_stream.as_ref(), upload_event.as_ref())
        .map_err(|e| TensorError::from_hal(e, "cpu"))?;
    session.wait(upload_event.as_ref()).map_err(|e| TensorError::from_hal(e, "cpu"))?;
    Ok(())
}

/// CUDA HAL `wait_event` probe (fails when CUDA unavailable).
pub fn probe_event_dep_cuda() -> Result<(), TensorError> {
    crate::cuda::probe_event_dep()
}
