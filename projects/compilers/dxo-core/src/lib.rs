//! DXO dxo-core engine — Titan CPU tensor, eager autograd, and CUDA matmul facade.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

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
