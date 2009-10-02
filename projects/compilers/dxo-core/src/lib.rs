//! DXO dxo-core engine — Titan CPU tensor, eager autograd, CUDA matmul spike (G4).

#![deny(missing_docs)]

mod autograd;
mod broadcast;
mod cuda;
mod engine;
mod shape;
mod storage;
mod tensor;

pub use autograd::{is_grad_enabled, set_grad_enabled, without_grad};
pub use cuda::is_available as cuda_available;
pub use engine::{backend_label, cpu_session};
pub use shape::{Shape, Strides, contiguous_strides};
pub use storage::Storage;
pub use tensor::{DeviceKind, Tensor, TensorError};

/// Crate version string (mirrors npm `@dxo/core` semver during preview).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver_like() {
        assert!(VERSION.starts_with("0."));
    }

    #[test]
    fn titan_cpu_engine_wired() {
        let _ = cpu_session();
        assert_eq!(backend_label(), "titan-cpu");
    }

    #[test]
    fn device_parse() {
        assert_eq!(DeviceKind::parse("CPU").unwrap(), DeviceKind::Cpu);
        assert_eq!(DeviceKind::parse("cuda").unwrap(), DeviceKind::Cuda);
        assert!(DeviceKind::parse("metal").is_err());
    }
}
