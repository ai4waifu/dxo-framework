//! DXO dxo-core engine — CPU tensor slice (M1); autograd/backends follow titan integration.

#![deny(missing_docs)]

mod tensor;

pub use tensor::{Tensor, TensorError};

/// Crate version string (mirrors npm `@dxo/core` semver during M0–M1).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver_like() {
        assert!(VERSION.starts_with("0."));
    }
}
