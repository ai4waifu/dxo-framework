//! DXO dxo-core engine — tensor storage, backends, autograd (M0: version + empty tensor shell).

#![deny(missing_docs)]

/// Crate version string (mirrors npm `@dxo/dxo-core` semver during M0).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder n-dimensional array handle (M1 will add storage/shape/strides).
#[derive(Debug, Clone, Default)]
pub struct Tensor {
    _private: (),
}

impl Tensor {
    /// Create an empty tensor shell (M0 scaffold).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver_like() {
        assert!(VERSION.starts_with("0."));
    }
}
