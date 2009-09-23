//! CPU float32 tensor with shared storage and stride views.

use std::fmt;

use rand::Rng;

use crate::broadcast::{broadcast_offset, broadcast_shapes, for_each_index};
use crate::shape::{Shape, Strides, contiguous_strides, is_contiguous, numel};
use crate::storage::Storage;

/// Shape / broadcast / matmul errors surfaced to napi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Invalid shape or numel mismatch.
    Shape(String),
    /// Incompatible broadcast.
    Broadcast(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(msg) | Self::Broadcast(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TensorError {}

/// Row-major float32 tensor view over shared [`Storage`].
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    offset: usize,
    shape: Shape,
    strides: Strides,
}

impl Tensor {
    /// Zeros tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let n = numel(shape).unwrap_or(0);
        Self::from_storage(Storage::zeros(n), 0, shape)
    }

    /// Ones tensor with the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        let n = numel(shape).unwrap_or(0);
        Self::from_storage(Storage::from_vec(vec![1.0; n]), 0, shape)
    }

    /// Standard-normal samples (Box–Muller); G1 uses host RNG only.
    pub fn randn(shape: &[usize]) -> Self {
        let n = numel(shape).unwrap_or(0);
        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let u1 = rng.random::<f32>().max(f32::EPSILON);
            let u2 = rng.random::<f32>();
            let mag = (-2.0 * u1.ln()).sqrt();
            data.push(mag * (2.0 * std::f32::consts::PI * u2).cos());
        }
        Self::from_storage(Storage::from_vec(data), 0, shape)
    }

    /// Construct from flat data; `data.len()` must equal shape product.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let n = numel(&shape)?;
        if data.len() != n {
            return Err(TensorError::Shape(format!(
                "data length {} does not match shape product {} (shape={shape:?})",
                data.len(),
                n
            )));
        }
        Ok(Self::from_storage(Storage::from_vec(data), 0, &shape))
    }

    fn from_storage(storage: Storage, offset: usize, shape: &[usize]) -> Self {
        let strides = contiguous_strides(shape);
        Self { storage, offset, shape: Shape(shape.to_vec()), strides: Strides(strides) }
    }

    /// Shape dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape.0
    }

    /// Element strides in elements.
    pub fn strides(&self) -> &[i64] {
        &self.strides.0
    }

    /// Titan protocol shape view.
    pub fn titan_shape(&self) -> titan_types::Shape {
        self.shape.to_titan()
    }

    /// Titan protocol strides view.
    pub fn titan_strides(&self) -> titan_types::Strides {
        self.strides.to_titan()
    }

    /// Underlying row-major payload (materializes non-contiguous tensors).
    pub fn data(&self) -> Vec<f32> {
        self.to_vec()
    }

    /// Element count.
    pub fn numel(&self) -> usize {
        numel(self.shape()).unwrap_or(0)
    }

    /// Whether this tensor is a dense row-major view.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(self.shape(), self.strides())
    }

    /// Shared storage identity (for view tests).
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.storage.ptr_eq(&other.storage)
    }

    /// Dense copy of tensor elements.
    pub fn to_vec(&self) -> Vec<f32> {
        let out_len = self.numel();
        if self.is_contiguous() && self.offset == 0 {
            let slice = &self.storage.data()[..out_len];
            return slice.to_vec();
        }
        let mut out = vec![0.0; out_len];
        self.for_each(|idx, value| {
            out[idx] = value;
        });
        out
    }

    fn linear_offset(&self, coords: &[usize]) -> usize {
        let mut off = self.offset as i64;
        for (c, s) in coords.iter().zip(self.strides()) {
            off += *c as i64 * *s;
        }
        off.max(0) as usize
    }

    fn read(&self, coords: &[usize]) -> f32 {
        self.storage.data()[self.linear_offset(coords)]
    }

    fn for_each(&self, mut f: impl FnMut(usize, f32)) {
        let shape = self.shape();
        let mut idx = 0usize;
        for_each_index(shape, |coords| {
            f(idx, self.read(coords));
            idx += 1;
        });
    }

    /// View with the same storage and a new shape (zero-copy when contiguous).
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let n = numel(shape)?;
        if n != self.numel() {
            return Err(TensorError::Shape(format!(
                "cannot reshape numel {} into product {} (shape={shape:?})",
                self.numel(),
                n
            )));
        }
        if !self.is_contiguous() {
            return Ok(Self::from_vec(self.to_vec(), shape.to_vec())?);
        }
        Ok(Self {
            storage: self.storage.clone(),
            offset: self.offset,
            shape: Shape(shape.to_vec()),
            strides: Strides(contiguous_strides(shape)),
        })
    }

    /// Transpose rank-2 tensor (zero-copy view).
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape().len() != 2 {
            return Err(TensorError::Shape("transpose requires rank-2 tensor".into()));
        }
        let shape = vec![self.shape()[1], self.shape()[0]];
        let strides = vec![self.strides()[1], self.strides()[0]];
        Ok(Self { storage: self.storage.clone(), offset: self.offset, shape: Shape(shape), strides: Strides(strides) })
    }

    /// Element-wise add with NumPy-style broadcast.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(other, |a, b| a + b)
    }

    /// Element-wise multiply with NumPy-style broadcast.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(other, |a, b| a * b)
    }

    fn binary_broadcast(&self, other: &Self, op: fn(f32, f32) -> f32) -> Result<Self, TensorError> {
        let out_shape = broadcast_shapes(self.shape(), other.shape())?;
        let rank = out_shape.len();
        let mut out = vec![0.0; numel(&out_shape)?];
        let mut oi = 0usize;
        for_each_index(&out_shape, |index| {
            let li = self.offset + broadcast_offset(index, self.shape(), self.strides(), rank);
            let ri = other.offset + broadcast_offset(index, other.shape(), other.strides(), rank);
            out[oi] = op(self.storage.data()[li], other.storage.data()[ri]);
            oi += 1;
        });
        Ok(Self::from_storage(Storage::from_vec(out), 0, &out_shape))
    }

    /// Matrix multiply for rank-2 tensors: `[m,k] @ [k,n] -> [m,n]`.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(TensorError::Shape("matmul requires rank-2 tensors".into()));
        }
        let a = if self.is_contiguous() { self.to_vec() } else { self.to_vec() };
        let b = other.to_vec();
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = other.shape()[0];
        let n = other.shape()[1];
        if k != k2 {
            return Err(TensorError::Shape(format!("matmul inner dims mismatch: {k} vs {k2}")));
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Ok(Self::from_storage(Storage::from_vec(out), 0, &[m, n]))
    }

    /// Element-wise ReLU.
    pub fn relu(&self) -> Self {
        let data = self.to_vec().into_iter().map(|x| x.max(0.0)).collect();
        Self::from_storage(Storage::from_vec(data), 0, self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.data()[0] - 19.0).abs() < 1e-5);
        assert!((c.data()[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn add_bias_broadcast() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        let y = x.add(&b).unwrap();
        assert_eq!(y.data(), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn reshape_is_zero_copy_view() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let v = t.reshape(&[4]).unwrap();
        assert!(t.shares_storage_with(&v));
        assert_eq!(v.shape(), &[4]);
    }

    #[test]
    fn transpose_is_zero_copy_view() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let v = t.transpose().unwrap();
        assert!(t.shares_storage_with(&v));
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.strides(), &[1, 2]);
    }

    #[test]
    fn mul_broadcast() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        let y = a.mul(&b).unwrap();
        assert_eq!(y.data(), vec![20.0, 60.0, 40.0, 100.0]);
    }

    #[test]
    fn ones_and_randn_shapes() {
        let o = Tensor::ones(&[2, 3]);
        assert_eq!(o.shape(), &[2, 3]);
        assert!(o.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
        let r = Tensor::randn(&[4]);
        assert_eq!(r.shape(), &[4]);
        assert_eq!(r.numel(), 4);
    }
}
