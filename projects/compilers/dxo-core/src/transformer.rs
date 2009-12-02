//! Embedding, LayerNorm, batched matmul, and scaled-dot-product attention (CPU reference).

use std::sync::Arc;

use crate::autograd::{GradFn, propagate};
use crate::dtype::DType;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorError};

// GradFn impls use `propagate(parent, &grad)` only (accumulate lives inside propagate).

impl Tensor {
    /// Gather rows from `weight` `[vocab, dim]` by integer token ids in `indices`.
    ///
    /// `indices` values are truncated toward zero from f32 storage. Output shape is
    /// `indices.shape + [dim]`.
    pub fn embedding(weight: &Self, indices: &Self) -> Result<Self, TensorError> {
        let ws = weight.shape();
        if ws.len() != 2 {
            return Err(TensorError::Shape("embedding weight expects [vocab, dim]".into()));
        }
        let (vocab, dim) = (ws[0], ws[1]);
        let idx = indices.to_vec();
        let mut out = vec![0.0f32; idx.len() * dim];
        let wd = weight.to_vec();
        for (i, &raw) in idx.iter().enumerate() {
            let id = raw as isize;
            if id < 0 || (id as usize) >= vocab {
                return Err(TensorError::Shape(format!("embedding index {id} out of range for vocab {vocab}")));
            }
            let row = id as usize;
            let src = row * dim;
            let dst = i * dim;
            out[dst..dst + dim].copy_from_slice(&wd[src..src + dim]);
        }
        let mut out_shape = indices.shape().to_vec();
        out_shape.push(dim);
        let base = Self::from_storage_on(Storage::from_vec(out), 0, &out_shape, false, None, None, weight.device());
        Ok(Self::maybe_attach(base, &[weight], Arc::new(EmbeddingBackward { weight: weight.clone(), indices: idx, dim })))
    }

    /// LayerNorm over the last dimension: `(x - mean) / sqrt(var + eps) * weight + bias`.
    pub fn layer_norm(&self, weight: &Self, bias: &Self, eps: f32) -> Result<Self, TensorError> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(TensorError::Shape("layer_norm requires rank >= 1".into()));
        }
        let dim = *shape.last().unwrap();
        if weight.shape() != [dim] || bias.shape() != [dim] {
            return Err(TensorError::Shape(format!(
                "layer_norm weight/bias expect [{dim}], got {:?} / {:?}",
                weight.shape(),
                bias.shape()
            )));
        }
        let data = self.to_vec();
        let w = weight.to_vec();
        let b = bias.to_vec();
        let outer = data.len() / dim;
        let mut out = vec![0.0f32; data.len()];
        let mut means = vec![0.0f32; outer];
        let mut inv_stds = vec![0.0f32; outer];
        for o in 0..outer {
            let base = o * dim;
            let slice = &data[base..base + dim];
            let mean = slice.iter().sum::<f32>() / dim as f32;
            let var = slice
                .iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f32>()
                / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            means[o] = mean;
            inv_stds[o] = inv;
            for i in 0..dim {
                let xn = (data[base + i] - mean) * inv;
                out[base + i] = xn * w[i] + b[i];
            }
        }
        let base = Self::from_storage_on(Storage::from_vec(out), 0, shape, false, None, None, self.device());
        Ok(Self::maybe_attach(
            base,
            &[self, weight, bias],
            Arc::new(LayerNormBackward {
                input: self.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                means,
                inv_stds,
                dim,
            }),
        ))
    }

    /// Batch matrix multiply: `[B, M, K] @ [B, K, N] -> [B, M, N]`.
    pub fn bmm(&self, other: &Self) -> Result<Self, TensorError> {
        let a = self.shape();
        let b = other.shape();
        if a.len() != 3 || b.len() != 3 {
            return Err(TensorError::Shape("bmm expects rank-3 tensors".into()));
        }
        if a[0] != b[0] || a[2] != b[1] {
            return Err(TensorError::Shape(format!("bmm shape mismatch {:?} @ {:?}", a, b)));
        }
        let (batch, m, k, n) = (a[0], a[1], a[2], b[2]);
        let ad = self.to_vec();
        let bd = other.to_vec();
        let mut out = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for t in 0..k {
                        s += ad[(bi * m + i) * k + t] * bd[(bi * k + t) * n + j];
                    }
                    out[(bi * m + i) * n + j] = s;
                }
            }
        }
        let base = Self::from_storage_on(Storage::from_vec(out), 0, &[batch, m, n], false, None, None, self.device());
        Ok(Self::maybe_attach(base, &[self, other], Arc::new(BmmBackward { a: self.clone(), b: other.clone() })))
    }

    /// Swap the last two dimensions (contiguous copy).
    pub fn transpose_last(&self) -> Result<Self, TensorError> {
        let s = self.shape();
        if s.len() < 2 {
            return Err(TensorError::Shape("transpose_last requires rank >= 2".into()));
        }
        let r = s.len();
        self.transpose_dims(r - 2, r - 1)
    }

    /// Swap two axes (contiguous copy). Used for attention layout `[B,T,H,D] ↔ [B,H,T,D]`.
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        let s = self.shape();
        let rank = s.len();
        if dim0 >= rank || dim1 >= rank {
            return Err(TensorError::Shape(format!("transpose_dims dims {dim0},{dim1} out of range for rank {rank}")));
        }
        if dim0 == dim1 {
            return Ok(self.clone());
        }
        let mut out_shape = s.to_vec();
        out_shape.swap(dim0, dim1);
        let data = self.to_vec();
        let mut out = vec![0.0f32; data.len()];
        let mut coord = vec![0usize; rank];
        for flat in 0..data.len() {
            let mut rem = flat;
            for d in (0..rank).rev() {
                coord[d] = rem % s[d];
                rem /= s[d];
            }
            coord.swap(dim0, dim1);
            let mut dst = 0usize;
            for d in 0..rank {
                dst = dst * out_shape[d] + coord[d];
            }
            out[dst] = data[flat];
        }
        let base = Self::from_storage_on(Storage::from_vec(out), 0, &out_shape, false, None, None, self.device());
        Ok(Self::maybe_attach(base, &[self], Arc::new(TransposeDimsBackward { input: self.clone(), dim0, dim1 })))
    }

    /// Scaled dot-product attention with optional causal mask.
    ///
    /// `q`, `k`, `v` shapes `[B, H, T, D]` (or broadcastable batch/head via reshape before call).
    pub fn scaled_dot_product_attention(q: &Self, k: &Self, v: &Self, causal: bool) -> Result<Self, TensorError> {
        let qs = q.shape();
        let ks = k.shape();
        let vs = v.shape();
        if qs.len() != 4 || ks.len() != 4 || vs.len() != 4 {
            return Err(TensorError::Shape("scaled_dot_product_attention expects [B,H,T,D]".into()));
        }
        if qs[0] != ks[0] || qs[0] != vs[0] || qs[1] != ks[1] || qs[1] != vs[1] {
            return Err(TensorError::Shape("attention batch/head mismatch".into()));
        }
        if qs[3] != ks[3] || vs[3] != qs[3] || ks[2] != vs[2] {
            return Err(TensorError::Shape(format!("attention dim mismatch q={:?} k={:?} v={:?}", qs, ks, vs)));
        }
        let (b, h, tq, d) = (qs[0], qs[1], qs[2], qs[3]);
        let tk = ks[2];
        let scale = 1.0 / (d as f32).sqrt();
        // Fold B*H into batch for bmm: [BH, Tq, D] @ [BH, D, Tk]
        let q2 = q.reshape(&[b * h, tq, d])?;
        let k2 = k.reshape(&[b * h, tk, d])?.transpose_last()?; // [BH, D, Tk]
        let scores = q2.bmm(&k2)?; // [BH, Tq, Tk]
        let scale_t = Self::from_vec(vec![scale; scores.numel()], scores.shape().to_vec())?;
        let mut scaled = scores.mul(&scale_t)?;
        if causal {
            let mut mask = vec![0.0f32; scores.numel()];
            for bi in 0..(b * h) {
                for i in 0..tq {
                    for j in 0..tk {
                        if j > i {
                            mask[(bi * tq + i) * tk + j] = -1e9;
                        }
                    }
                }
            }
            let mask_t = Self::from_vec(mask, scores.shape().to_vec())?;
            scaled = scaled.add(&mask_t)?;
        }
        let attn = scaled.softmax()?;
        let v2 = v.reshape(&[b * h, tk, d])?;
        let out = attn.bmm(&v2)?; // [BH, Tq, D]
        out.reshape(&[b, h, tq, d])
    }

    /// Retag logical dtype without changing host f32 payload (honest AMP preview).
    pub fn cast_dtype(&self, dtype: DType) -> Self {
        if !self.requires_grad() {
            let mut t = self.detach();
            t.set_dtype(dtype);
            return t;
        }
        let base = Self::from_storage_on(Storage::from_vec(self.to_vec()), 0, self.shape(), false, None, None, self.device());
        let mut out = Self::maybe_attach(base, &[self], Arc::new(CastDtypeBackward { input: self.clone() }));
        out.set_dtype(dtype);
        out
    }
}

struct EmbeddingBackward {
    weight: Tensor,
    indices: Vec<f32>,
    dim: usize,
}

impl GradFn for EmbeddingBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let mut gw = vec![0.0f32; self.weight.numel()];
        for (i, &raw) in self.indices.iter().enumerate() {
            let row = raw as usize;
            let src = i * self.dim;
            let dst = row * self.dim;
            for j in 0..self.dim {
                gw[dst + j] += grad_output[src + j];
            }
        }
        propagate(&self.weight, &gw)
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

struct LayerNormBackward {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    means: Vec<f32>,
    inv_stds: Vec<f32>,
    dim: usize,
}

impl GradFn for LayerNormBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let x = self.input.to_vec();
        let w = self.weight.to_vec();
        let outer = self.means.len();
        let mut gx = vec![0.0f32; x.len()];
        let mut gw = vec![0.0f32; self.dim];
        let mut gb = vec![0.0f32; self.dim];
        for o in 0..outer {
            let base = o * self.dim;
            let mean = self.means[o];
            let inv = self.inv_stds[o];
            let mut dot_dy_xhat = 0.0f32;
            let mut sum_dy = 0.0f32;
            let mut xhat = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                xhat[i] = (x[base + i] - mean) * inv;
                let dy = grad_output[base + i];
                gw[i] += dy * xhat[i];
                gb[i] += dy;
                let dy_w = dy * w[i];
                sum_dy += dy_w;
                dot_dy_xhat += dy_w * xhat[i];
            }
            let n = self.dim as f32;
            for i in 0..self.dim {
                let dy_w = grad_output[base + i] * w[i];
                gx[base + i] = inv / n * (n * dy_w - sum_dy - xhat[i] * dot_dy_xhat);
            }
        }
        propagate(&self.input, &gx)?;
        propagate(&self.weight, &gw)?;
        propagate(&self.bias, &gb)
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input, &self.weight, &self.bias]
    }
}

struct BmmBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for BmmBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let as_ = self.a.shape();
        let bs = self.b.shape();
        let (batch, m, k, n) = (as_[0], as_[1], as_[2], bs[2]);
        let ad = self.a.to_vec();
        let bd = self.b.to_vec();
        let mut ga = vec![0.0f32; self.a.numel()];
        let mut gb = vec![0.0f32; self.b.numel()];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let g = grad_output[(bi * m + i) * n + j];
                    for t in 0..k {
                        ga[(bi * m + i) * k + t] += g * bd[(bi * k + t) * n + j];
                        gb[(bi * k + t) * n + j] += g * ad[(bi * m + i) * k + t];
                    }
                }
            }
        }
        propagate(&self.a, &ga)?;
        propagate(&self.b, &gb)
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

struct TransposeDimsBackward {
    input: Tensor,
    dim0: usize,
    dim1: usize,
}

impl GradFn for TransposeDimsBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let out_shape = {
            let mut s = self.input.shape().to_vec();
            s.swap(self.dim0, self.dim1);
            s
        };
        let in_shape = self.input.shape();
        let rank = in_shape.len();
        let mut gin = vec![0.0f32; self.input.numel()];
        let mut coord = vec![0usize; rank];
        for flat in 0..grad_output.len() {
            let mut rem = flat;
            for d in (0..rank).rev() {
                coord[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            coord.swap(self.dim0, self.dim1);
            let mut dst = 0usize;
            for d in 0..rank {
                dst = dst * in_shape[d] + coord[d];
            }
            gin[dst] = grad_output[flat];
        }
        propagate(&self.input, &gin)
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct CastDtypeBackward {
    input: Tensor,
}

impl GradFn for CastDtypeBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        propagate(&self.input, grad_output)
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_gathers_rows() {
        let w = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]).unwrap();
        let ids = Tensor::from_vec(vec![2., 0.], vec![2]).unwrap();
        let out = Tensor::embedding(&w, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.to_vec(), vec![5., 6., 1., 2.]);
    }

    #[test]
    fn layer_norm_unit_var() {
        let x = Tensor::from_vec(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let w = Tensor::ones(&[2]);
        let b = Tensor::zeros(&[2]);
        let y = x.layer_norm(&w, &b, 1e-5).unwrap();
        let v = y.to_vec();
        // each row mean ~0
        assert!((v[0] + v[1]).abs() < 1e-4);
        assert!((v[2] + v[3]).abs() < 1e-4);
    }

    #[test]
    fn causal_attention_masks_future() {
        let q = Tensor::from_vec(vec![1., 0., 0., 1.], vec![1, 1, 2, 2]).unwrap();
        let k = q.clone();
        let v = Tensor::from_vec(vec![10., 20., 30., 40.], vec![1, 1, 2, 2]).unwrap();
        let out = Tensor::scaled_dot_product_attention(&q, &k, &v, true).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        // position 0 can only attend to itself → v[0]
        let data = out.to_vec();
        assert!((data[0] - 10.).abs() < 1e-3);
        assert!((data[1] - 20.).abs() < 1e-3);
    }
}
