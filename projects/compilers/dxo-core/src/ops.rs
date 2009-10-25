//! Extended tensor ops: reductions, softmax, cat/stack, slice, sub/div.

use std::sync::Arc;

use crate::autograd::{GradFn, propagate};
use crate::broadcast::{broadcast_shapes, for_each_index};
use crate::shape::numel;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorError};

impl Tensor {
    /// Element-wise subtract (broadcast).
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        let neg = other.neg()?;
        self.add(&neg)
    }

    /// Negate all elements.
    pub fn neg(&self) -> Result<Self, TensorError> {
        let data = self.to_vec().into_iter().map(|x| -x).collect();
        let base = Self::from_storage_on(Storage::from_vec(data), 0, self.shape(), false, None, None, self.device());
        Ok(Self::maybe_attach(base, &[self], Arc::new(NegBackward { input: self.clone() })))
    }

    /// Element-wise divide (broadcast).
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        let out = self.binary_broadcast(other, |a, b| if b.abs() < f32::EPSILON { f32::NAN } else { a / b })?;
        Ok(Self::maybe_attach(out, &[self, other], Arc::new(DivBackward { a: self.clone(), b: other.clone() })))
    }

    /// Mean of all elements → scalar `[1]`.
    pub fn mean(&self) -> Self {
        let n = self.numel().max(1) as f32;
        let s = self.to_vec().iter().sum::<f32>() / n;
        let base = Self::from_storage_on(Storage::from_vec(vec![s]), 0, &[1], false, None, None, self.device());
        Self::maybe_attach(base, &[self], Arc::new(MeanBackward { input: self.clone(), n: self.numel() }))
    }

    /// Max of all elements → scalar `[1]`.
    pub fn max_all(&self) -> Self {
        let m = self.to_vec().into_iter().fold(f32::NEG_INFINITY, f32::max);
        let base = Self::from_storage_on(Storage::from_vec(vec![m]), 0, &[1], false, None, None, self.device());
        Self::maybe_attach(base, &[self], Arc::new(MaxBackward { input: self.clone() }))
    }

    /// Softmax along the last dimension (stable).
    pub fn softmax(&self) -> Result<Self, TensorError> {
        let rank = self.shape().len();
        if rank == 0 {
            return Err(TensorError::Shape("softmax requires rank >= 1".into()));
        }
        let axis = rank - 1;
        let outer: usize = self.shape()[..axis].iter().product();
        let inner = self.shape()[axis];
        let data = self.to_vec();
        let mut out = vec![0.0f32; data.len()];
        for o in 0..outer.max(1) {
            let base = o * inner;
            let slice = &data[base..base + inner];
            let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = slice.iter().map(|&x| (x - max_v).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for (i, e) in exp.iter().enumerate() {
                out[base + i] = e / sum;
            }
        }
        let base = Self::from_storage_on(Storage::from_vec(out.clone()), 0, self.shape(), false, None, None, self.device());
        Ok(Self::maybe_attach(base, &[self], Arc::new(SoftmaxBackward { input: self.clone(), output: out })))
    }

    /// Log-softmax along last dimension.
    pub fn log_softmax(&self) -> Result<Self, TensorError> {
        let sm = self.softmax()?;
        let data = sm.to_vec().into_iter().map(|x| x.ln()).collect();
        Ok(Self::from_storage_on(Storage::from_vec(data), 0, self.shape(), false, None, None, self.device()))
    }

    /// Concatenate tensors along `dim`.
    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::Shape("cat requires at least one tensor".into()));
        }
        let rank = tensors[0].shape().len();
        if dim >= rank {
            return Err(TensorError::Shape(format!("cat dim {dim} out of range for rank {rank}")));
        }
        let device = tensors[0].device();
        for t in tensors.iter().skip(1) {
            if t.shape().len() != rank {
                return Err(TensorError::Shape("cat requires same rank".into()));
            }
            if t.device() != device {
                return Err(TensorError::Device("cat device mismatch".into()));
            }
            for (i, (&a, &b)) in tensors[0].shape().iter().zip(t.shape()).enumerate() {
                if i != dim && a != b {
                    return Err(TensorError::Shape(format!("cat shape mismatch at dim {i}: {a} vs {b}")));
                }
            }
        }
        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();
        let mut out_data = Vec::new();
        for t in tensors {
            out_data.extend(t.to_vec());
        }
        let parents: Vec<&Tensor> = tensors.iter().collect();
        let base = Self::from_storage_on(Storage::from_vec(out_data), 0, &out_shape, false, None, None, device);
        Ok(Self::maybe_attach(base, &parents, Arc::new(CatBackward { inputs: tensors.to_vec(), dim })))
    }

    /// Stack tensors along new dimension `dim`.
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::Shape("stack requires at least one tensor".into()));
        }
        let ref_shape = tensors[0].shape();
        for t in tensors.iter().skip(1) {
            if t.shape() != ref_shape {
                return Err(TensorError::Shape("stack requires identical shapes".into()));
            }
            if t.device() != tensors[0].device() {
                return Err(TensorError::Device("stack device mismatch".into()));
            }
        }
        let mut out_shape = ref_shape.to_vec();
        if dim > out_shape.len() {
            out_shape.push(tensors.len());
        } else {
            out_shape.insert(dim, tensors.len());
        }
        let chunk = numel(ref_shape)?;
        let n = numel(&out_shape)?;
        let mut out_data = vec![0.0f32; n];
        for (i, t) in tensors.iter().enumerate() {
            let start = i * chunk;
            out_data[start..start + chunk].copy_from_slice(&t.to_vec());
        }
        Ok(Self::from_storage_on(Storage::from_vec(out_data), 0, &out_shape, false, None, None, tensors[0].device()))
    }

    /// Slice `[start..start+len)` along `dim`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, TensorError> {
        let rank = self.shape().len();
        if dim >= rank {
            return Err(TensorError::Shape(format!("narrow dim {dim} out of range")));
        }
        if start + len > self.shape()[dim] {
            return Err(TensorError::Shape(format!(
                "narrow range {}..{} exceeds dim size {}",
                start,
                start + len,
                self.shape()[dim]
            )));
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[dim] = len;
        let data = self.to_vec();
        let in_shape = self.shape();
        let out_n = numel(&out_shape)?;
        let mut out = vec![0.0f32; out_n];
        let mut oi = 0usize;
        for_each_index(&out_shape, |coords| {
            let mut in_coords = coords.to_vec();
            in_coords[dim] += start;
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in (0..rank).rev() {
                idx += in_coords[i] * stride;
                stride *= in_shape[i];
            }
            out[oi] = data[idx];
            oi += 1;
        });
        let base = Self::from_storage_on(Storage::from_vec(out), 0, &out_shape, false, None, None, self.device());
        Ok(Self::maybe_attach(base, &[self], Arc::new(NarrowBackward { input: self.clone(), dim, start, len })))
    }
}

struct NegBackward {
    input: Tensor,
}

impl GradFn for NegBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let gx: Vec<f32> = grad_output.iter().map(|&g| -g).collect();
        propagate(&self.input, &gx)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct DivBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for DivBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let a_v = self.a.to_vec();
        let b_v = self.b.to_vec();
        let out_shape = broadcast_shapes(self.a.shape(), self.b.shape())?;
        let out_n = numel(&out_shape)?;
        let mut ga = vec![0.0f32; self.a.numel()];
        let mut gb = vec![0.0f32; self.b.numel()];
        let ga_len = ga.len();
        let gb_len = gb.len();
        for oi in 0..out_n {
            let g = grad_output[oi];
            let av = a_v[oi % a_v.len()];
            let bv = b_v[oi % b_v.len()].max(f32::EPSILON);
            ga[oi % ga_len] += g / bv;
            gb[oi % gb_len] += -g * av / (bv * bv);
        }
        propagate(&self.a, &ga)?;
        propagate(&self.b, &gb)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

struct MeanBackward {
    input: Tensor,
    n: usize,
}

impl GradFn for MeanBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let g = grad_output.first().copied().unwrap_or(0.0) / self.n.max(1) as f32;
        propagate(&self.input, &vec![g; self.input.numel()])
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct MaxBackward {
    input: Tensor,
}

impl GradFn for MaxBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let x = self.input.to_vec();
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let g = grad_output.first().copied().unwrap_or(0.0);
        let gx: Vec<f32> = x.iter().map(|&v| if (v - m).abs() < 1e-6 { g } else { 0.0 }).collect();
        propagate(&self.input, &gx)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct SoftmaxBackward {
    input: Tensor,
    output: Vec<f32>,
}

impl GradFn for SoftmaxBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let y = &self.output;
        let g = grad_output;
        let rank = self.input.shape().len();
        let axis = rank.saturating_sub(1);
        let outer: usize = self.input.shape()[..axis].iter().product::<usize>().max(1);
        let inner = self.input.shape()[axis];
        let mut gx = vec![0.0f32; g.len()];
        for o in 0..outer {
            let base = o * inner;
            let dot: f32 = (0..inner).map(|i| g[base + i] * y[base + i]).sum();
            for i in 0..inner {
                gx[base + i] = y[base + i] * (g[base + i] - dot);
            }
        }
        propagate(&self.input, &gx)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct CatBackward {
    inputs: Vec<Tensor>,
    dim: usize,
}

impl GradFn for CatBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let _ = self.dim;
        let mut offset = 0usize;
        for input in &self.inputs {
            let chunk = input.numel();
            propagate(input, &grad_output[offset..offset + chunk])?;
            offset += chunk;
        }
        Ok(())
    }
    fn parents(&self) -> Vec<&Tensor> {
        self.inputs.iter().collect()
    }
}

struct NarrowBackward {
    input: Tensor,
    dim: usize,
    start: usize,
    len: usize,
}

impl GradFn for NarrowBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let mut gx = vec![0.0f32; self.input.numel()];
        let in_shape = self.input.shape();
        let out_shape = {
            let mut s = in_shape.to_vec();
            s[self.dim] = self.len;
            s
        };
        let rank = in_shape.len();
        let mut oi = 0usize;
        for_each_index(&out_shape, |coords| {
            let mut in_coords = coords.to_vec();
            in_coords[self.dim] += self.start;
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in (0..rank).rev() {
                idx += in_coords[i] * stride;
                stride *= in_shape[i];
            }
            gx[idx] += grad_output[oi];
            oi += 1;
        });
        propagate(&self.input, &gx)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}
