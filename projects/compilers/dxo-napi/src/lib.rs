//! Node N-API bindings for DXO (`version`, CPU `Tensor` factories and ops).

#![deny(missing_docs)]

use dxo_core::Tensor as CoreTensor;
use napi::bindgen_prelude::*;
use napi_derive::napi;

fn map_err(err: dxo_core::TensorError) -> Error {
    Error::from_reason(err.to_string())
}

/// Return the DXO engine version string.
#[napi]
pub fn version() -> String {
    dxo_core::VERSION.to_string()
}

/// Dense float32 tensor backed by `dxo-core`.
#[napi]
pub struct Tensor {
    inner: CoreTensor,
}

#[napi]
impl Tensor {
    /// Shape dimensions.
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| u32::try_from(d).unwrap_or(u32::MAX)).collect()
    }

    /// Element-wise add (with bias broadcast on the last axis).
    #[napi]
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.add(&other.inner).map_err(map_err)? })
    }

    /// Rank-2 matrix multiply.
    #[napi]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.matmul(&other.inner).map_err(map_err)? })
    }

    /// Reshape (same numel).
    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Tensor> {
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok(Tensor { inner: self.inner.reshape(&shape).map_err(map_err)? })
    }

    /// Element-wise ReLU.
    #[napi]
    pub fn relu(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.relu() })
    }

    /// Row-major float32 payload (copy-out for tests).
    #[napi]
    pub fn to_array(&self) -> Vec<f64> {
        self.inner.data().iter().map(|&x| f64::from(x)).collect()
    }
}

/// Create a tensor filled with zeros.
#[napi]
pub fn zeros(shape: Vec<u32>) -> Result<Tensor> {
    let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Tensor { inner: CoreTensor::zeros(&shape) })
}

/// Create a tensor from flat data and shape.
#[napi]
pub fn tensor(data: Vec<f64>, shape: Vec<u32>) -> Result<Tensor> {
    let data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Tensor { inner: CoreTensor::from_vec(data, shape).map_err(map_err)? })
}
