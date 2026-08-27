//! Node N-API bindings for DXO (M0: `version()` + empty `Tensor`).

#![deny(missing_docs)]

use napi_derive::napi;

/// Return the DXO engine version string.
#[napi]
pub fn version() -> String {
    dxo_core::VERSION.to_string()
}

/// Placeholder tensor handle exposed to JavaScript (M1 adds data/shape).
#[napi]
pub struct Tensor {}

#[napi]
impl Tensor {
    /// Create an empty tensor shell.
    #[napi(constructor)]
    pub fn new() -> Self {
        let _ = dxo_core::Tensor::new();
        Self {}
    }
}
