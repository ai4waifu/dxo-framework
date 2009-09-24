//! Titan-backed CPU engine facade (G1 phase-1).
//!
//! Host tensors in this slice still use shared CPU storage; the session proves
//! the Titan HAL boundary is wired for later device-buffer uploads.

use std::sync::{Arc, OnceLock};

use titan_backend_cpu::CpuDriver;
use titan_hal::{BackendDriver, DeviceSession};
use titan_types::{BackendId, DeviceId};

static CPU_SESSION: OnceLock<Arc<dyn DeviceSession>> = OnceLock::new();

/// Open (once) the shared Titan CPU `DeviceSession`.
pub fn cpu_session() -> Arc<dyn DeviceSession> {
    CPU_SESSION
        .get_or_init(|| {
            Arc::new(CpuDriver)
                .open(DeviceId { backend: BackendId::Cpu, ordinal: 0 })
                .expect("titan CpuDriver session")
        })
        .clone()
}

/// Stable backend label surfaced to diagnostics.
pub const fn backend_label() -> &'static str {
    "titan-cpu"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_session_opens_once() {
        let a = cpu_session();
        let b = cpu_session();
        assert_eq!(a.device(), b.device());
        assert_eq!(a.device().backend, BackendId::Cpu);
    }
}
