//! Node N-API bindings for DXO (`version`, CPU `Tensor` factories and ops).

#![deny(missing_docs)]

use dxo_core::Tensor as CoreTensor;
use napi::bindgen_prelude::*;
use napi_derive::napi;

fn map_err(err: dxo_core::TensorError) -> Error {
    Error::from_reason(err.to_string())
}

fn shape_to_usize(shape: Vec<u32>) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

/// Return the DXO engine version string.
#[napi]
pub fn version() -> String {
    dxo_core::VERSION.to_string()
}

/// Titan CPU backend label wired through `dxo-core` facade.
#[napi]
pub fn backend() -> String {
    dxo_core::backend_label().to_string()
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

    /// Element-wise add (NumPy-style broadcast).
    #[napi]
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.add(&other.inner).map_err(map_err)? })
    }

    /// Element-wise multiply (NumPy-style broadcast).
    #[napi]
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.mul(&other.inner).map_err(map_err)? })
    }

    /// Rank-2 matrix multiply.
    #[napi]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.matmul(&other.inner).map_err(map_err)? })
    }

    /// Reshape (zero-copy view when contiguous).
    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.reshape(&shape_to_usize(shape)).map_err(map_err)? })
    }

    /// Transpose rank-2 tensor (zero-copy view).
    #[napi]
    pub fn transpose(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.transpose().map_err(map_err)? })
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
    Ok(Tensor { inner: CoreTensor::zeros(&shape_to_usize(shape)) })
}

/// Create a tensor filled with ones.
#[napi]
pub fn ones(shape: Vec<u32>) -> Result<Tensor> {
    Ok(Tensor { inner: CoreTensor::ones(&shape_to_usize(shape)) })
}

/// Create a tensor with standard-normal samples.
#[napi]
pub fn randn(shape: Vec<u32>) -> Result<Tensor> {
    Ok(Tensor { inner: CoreTensor::randn(&shape_to_usize(shape)) })
}

/// Create a tensor from flat `f64` data and shape.
#[napi]
pub fn tensor(data: Vec<f64>, shape: Vec<u32>) -> Result<Tensor> {
    let data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Ok(Tensor { inner: CoreTensor::from_vec(data, shape_to_usize(shape)).map_err(map_err)? })
}

/// Create a tensor from a Node `Float32Array` byte buffer (no f64 cast).
#[napi]
pub fn tensor_f32(data: Buffer, shape: Vec<u32>) -> Result<Tensor> {
    if data.len() % 4 != 0 {
        return Err(Error::from_reason("tensor_f32 buffer byte length must be a multiple of 4"));
    }
    let f32_len = data.len() / 4;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, f32_len) };
    Ok(Tensor { inner: CoreTensor::from_vec(slice.to_vec(), shape_to_usize(shape)).map_err(map_err)? })
}
