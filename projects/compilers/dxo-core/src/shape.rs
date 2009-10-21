//! Shape / stride helpers backed by `titan-types` protocol types.

use titan_types::{Shape as TitanShape, Strides as TitanStrides};

use crate::tensor::TensorError;

/// Dynamic row-major shape (elements are `usize` at the DXO facade).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Element count; errors on overflow.
    pub fn numel(&self) -> Result<usize, TensorError> {
        numel(&self.0)
    }

    /// Titan wire shape for protocol interop.
    pub fn to_titan(&self) -> TitanShape {
        TitanShape(self.0.iter().map(|&d| d as u64).collect())
    }

    /// Build from Titan wire shape.
    pub fn from_titan(shape: &TitanShape) -> Self {
        Self(shape.0.iter().map(|&d| d as usize).collect())
    }
}

/// Signed element strides (Titan protocol).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides(pub Vec<i64>);

impl Strides {
    /// Titan wire strides for protocol interop.
    pub fn to_titan(&self) -> TitanStrides {
        TitanStrides(self.0.clone())
    }

    /// Build from Titan wire strides.
    pub fn from_titan(strides: &TitanStrides) -> Self {
        Self(strides.0.clone())
    }
}

/// Row-major contiguous strides for `shape`.
pub fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut out = vec![0i64; shape.len()];
    let mut step = 1i64;
    for i in (0..shape.len()).rev() {
        out[i] = step;
        step = step.saturating_mul(shape[i] as i64);
    }
    out
}

/// Product of shape dimensions.
pub fn numel(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or_else(|| TensorError::Shape("shape product overflow".into()))
}

/// Whether `strides` describe a dense row-major layout for `shape`.
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    contiguous_strides(shape) == strides
}
