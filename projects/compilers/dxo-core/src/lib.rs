//! DXO dxo-core engine — Titan CPU tensor, eager autograd, and CUDA matmul facade.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod autograd;
mod broadcast;
mod conv;
mod cuda;
mod diagnostic;
mod dtype;
mod engine;
mod ops;
mod optim;
mod shape;
mod storage;
mod tensor;
mod transformer;

pub use autograd::{is_grad_enabled, set_grad_enabled, without_grad};
pub use diagnostic::{
    Diagnostic, DiagnosticValue, Severity, from_hal_error, from_titan_error, titan_kind_to_code,
};
pub use optim::{AdamState, adam_step, backward_sgd_step, sgd_step, zero_grads};
pub use cuda::{
    capability_fingerprint as cuda_capability_fingerprint, host_transfer_count, is_available as cuda_available,
    reset_host_transfer_count,
};
pub use dtype::DType;
pub use engine::{backend_label, cpu_session, probe_event_dep, probe_event_dep_cuda};
pub use shape::{Shape, Strides, contiguous_strides};
pub use storage::Storage;
pub use tensor::{DeviceKind, Tensor, TensorError};

/// Crate version string (mirrors npm `@dxo/core` semver during preview).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
