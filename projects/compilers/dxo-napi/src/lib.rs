//! Node N-API bindings for DXO (`version`, CPU `Tensor`, eager autograd).

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![allow(unsafe_code)] // `tensor_f32` reads Node `Buffer` bytes as f32 via a validated slice view.

use std::fmt;

use dxo_core::{DeviceKind, Tensor as CoreTensor};
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

/// Whether Titan CUDA can open a Driver session on this machine.
#[napi]
pub fn cuda_available() -> bool {
    dxo_core::cuda_available()
}

/// Set whether the current thread records autograd ops; returns previous flag.
#[napi]
pub fn set_grad_enabled(enabled: bool) -> bool {
    dxo_core::set_grad_enabled(enabled)
}

/// Whether the current thread records autograd ops.
#[napi]
pub fn is_grad_enabled() -> bool {
    dxo_core::is_grad_enabled()
}

/// Dense float32 tensor backed by `dxo-core`.
#[napi]
pub struct Tensor {
    inner: CoreTensor,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor").field("inner", &self.inner).finish()
    }
}

#[napi]
impl Tensor {
    /// Shape dimensions.
    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| u32::try_from(d).unwrap_or(u32::MAX)).collect()
    }

    /// Whether this tensor tracks gradients.
    #[napi(getter)]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Device tag (`cpu` or `cuda`).
    #[napi(getter)]
    pub fn device(&self) -> String {
        self.inner.device().as_str().to_string()
    }

    /// Accumulated gradient as row-major f64, or `null` if absent.
    #[napi(getter)]
    pub fn grad(&self) -> Option<Vec<f64>> {
        self.inner.grad().map(|g| g.iter().map(|&x| f64::from(x)).collect())
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

    /// Logical dtype tag.
    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.inner.dtype().as_str().to_string()
    }

    /// Element-wise subtract (broadcast).
    #[napi]
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.sub(&other.inner).map_err(map_err)? })
    }

    /// Negate all elements.
    #[napi]
    pub fn neg(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.neg().map_err(map_err)? })
    }

    /// Element-wise divide (broadcast).
    #[napi]
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.div(&other.inner).map_err(map_err)? })
    }

    /// Mean of all elements â†?scalar `[1]`.
    #[napi]
    pub fn mean(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.mean() })
    }

    /// Max of all elements â†?scalar `[1]`.
    #[napi]
    pub fn max_all(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.max_all() })
    }

    /// Softmax along last dimension.
    #[napi]
    pub fn softmax(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.softmax().map_err(map_err)? })
    }

    /// Log-softmax along last dimension.
    #[napi]
    pub fn log_softmax(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.log_softmax().map_err(map_err)? })
    }

    /// Slice `[start..start+len)` along `dim`.
    #[napi]
    pub fn narrow(&self, dim: u32, start: u32, len: u32) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.narrow(dim as usize, start as usize, len as usize).map_err(map_err)? })
    }

    /// 2D convolution NCHW.
    #[napi]
    pub fn conv2d(&self, weight: &Tensor, bias: Option<&Tensor>, stride: u32, padding: u32) -> Result<Tensor> {
        Ok(Tensor {
            inner: self
                .inner
                .conv2d(&weight.inner, bias.map(|b| &b.inner), stride as usize, padding as usize)
                .map_err(map_err)?,
        })
    }

    /// Max pool 2D NCHW.
    #[napi]
    pub fn max_pool2d(&self, kernel: u32, stride: u32, padding: u32) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.max_pool2d(kernel as usize, stride as usize, padding as usize).map_err(map_err)? })
    }

    /// Batch norm 2D (training-style).
    #[napi]
    pub fn batch_norm2d(&self, gamma: &Tensor, beta: &Tensor, eps: Option<f64>) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.batch_norm2d(&gamma.inner, &beta.inner, eps.unwrap_or(1e-5) as f32).map_err(map_err)? })
    }

    /// Sum all elements to a scalar tensor of shape `[1]`.
    #[napi]
    pub fn sum(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.sum() })
    }

    /// Detach from the tape (values only).
    #[napi]
    pub fn detach(&self) -> Result<Tensor> {
        Ok(Tensor { inner: self.inner.detach() })
    }

    /// Move tensor to `cpu` or `cuda` (detached tensors only for CUDA in this preview).
    #[napi]
    pub fn to(&self, device: String) -> Result<Tensor> {
        let kind = DeviceKind::parse(&device).map_err(map_err)?;
        Ok(Tensor { inner: self.inner.to(kind).map_err(map_err)? })
    }

    /// Clear accumulated gradient.
    #[napi]
    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Reverse-mode autodiff from a scalar.
    #[napi]
    pub fn backward(&self) -> Result<()> {
        self.inner.backward().map_err(map_err)
    }

    /// Row-major float32 payload (copy-out for tests).
    #[napi]
    pub fn to_array(&self) -> Vec<f64> {
        self.inner.data().iter().map(|&x| f64::from(x)).collect()
    }
}

/// Concatenate tensors along `dim`.
#[napi]
pub fn cat(tensors: Vec<&Tensor>, dim: u32) -> Result<Tensor> {
    let owned: Vec<CoreTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    Ok(Tensor { inner: CoreTensor::cat(&owned, dim as usize).map_err(map_err)? })
}

/// Stack tensors along new dimension `dim`.
#[napi]
pub fn stack(tensors: Vec<&Tensor>, dim: u32) -> Result<Tensor> {
    let owned: Vec<CoreTensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    Ok(Tensor { inner: CoreTensor::stack(&owned, dim as usize).map_err(map_err)? })
}

/// Create a tensor filled with zeros.
#[napi]
pub fn zeros(shape: Vec<u32>, requires_grad: Option<bool>) -> Result<Tensor> {
    let t = CoreTensor::zeros(&shape_to_usize(shape));
    if requires_grad.unwrap_or(false) {
        let data = t.to_vec();
        let shape = t.shape().to_vec();
        return Ok(Tensor { inner: CoreTensor::from_vec_grad(data, shape, true).map_err(map_err)? });
    }
    Ok(Tensor { inner: t })
}

/// Create a tensor filled with ones.
#[napi]
pub fn ones(shape: Vec<u32>, requires_grad: Option<bool>) -> Result<Tensor> {
    let t = CoreTensor::ones(&shape_to_usize(shape));
    if requires_grad.unwrap_or(false) {
        let data = t.to_vec();
        let shape = t.shape().to_vec();
        return Ok(Tensor { inner: CoreTensor::from_vec_grad(data, shape, true).map_err(map_err)? });
    }
    Ok(Tensor { inner: t })
}

/// Create a tensor with standard-normal samples.
#[napi]
pub fn randn(shape: Vec<u32>, requires_grad: Option<bool>) -> Result<Tensor> {
    let t = CoreTensor::randn(&shape_to_usize(shape));
    if requires_grad.unwrap_or(false) {
        let data = t.to_vec();
        let shape = t.shape().to_vec();
        return Ok(Tensor { inner: CoreTensor::from_vec_grad(data, shape, true).map_err(map_err)? });
    }
    Ok(Tensor { inner: t })
}

/// Create a tensor from flat `f64` data and shape.
#[napi]
pub fn tensor(data: Vec<f64>, shape: Vec<u32>, requires_grad: Option<bool>) -> Result<Tensor> {
    let data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Ok(Tensor {
        inner: CoreTensor::from_vec_grad(data, shape_to_usize(shape), requires_grad.unwrap_or(false)).map_err(map_err)?,
    })
}

/// Create a tensor from a Node `Float32Array` byte buffer (no f64 cast).
#[napi]
pub fn tensor_f32(data: Buffer, shape: Vec<u32>, requires_grad: Option<bool>) -> Result<Tensor> {
    if data.len() % 4 != 0 {
        return Err(Error::from_reason("tensor_f32 buffer byte length must be a multiple of 4"));
    }
    let f32_len = data.len() / 4;
    // SAFETY: length is a multiple of four; `Buffer` outlives this call and `as_ptr` is valid for `f32_len` elements.
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, f32_len) };
    Ok(Tensor {
        inner: CoreTensor::from_vec_grad(slice.to_vec(), shape_to_usize(shape), requires_grad.unwrap_or(false))
            .map_err(map_err)?,
    })
}

