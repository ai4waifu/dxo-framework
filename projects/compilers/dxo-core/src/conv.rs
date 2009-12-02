//! Conv2d, pooling, and batch norm (NCHW, CPU reference).

use std::sync::Arc;

use crate::autograd::{GradFn, propagate};
use crate::tensor::{Tensor, TensorError};

impl Tensor {
    /// 2D convolution NCHW: input `[N,C,H,W]`, weight `[O,C,kH,kW]`, optional bias `[O]`.
    pub fn conv2d(&self, weight: &Self, bias: Option<&Self>, stride: usize, padding: usize) -> Result<Self, TensorError> {
        let (out, _) = conv2d_forward(self, weight, bias, stride, padding)?;
        let parents: Vec<&Tensor> =
            if bias.is_some() { vec![self, weight, bias.as_ref().unwrap()] } else { vec![self, weight] };
        Ok(Self::maybe_attach(
            out,
            &parents,
            Arc::new(Conv2dBackward { input: self.clone(), weight: weight.clone(), bias: bias.cloned(), stride, padding }),
        ))
    }

    /// Max pool 2D NCHW.
    pub fn max_pool2d(&self, kernel: usize, stride: usize, padding: usize) -> Result<Self, TensorError> {
        let (out, indices) = max_pool2d_forward(self, kernel, stride, padding)?;
        Ok(Self::maybe_attach(
            out,
            &[self],
            Arc::new(MaxPool2dBackward { input: self.clone(), indices, kernel, stride, padding }),
        ))
    }

    /// Batch norm 2D (training-style single batch): `[N,C,H,W]`.
    pub fn batch_norm2d(&self, gamma: &Self, beta: &Self, eps: f32) -> Result<Self, TensorError> {
        let (out, mean, var) = batch_norm2d_forward(self, gamma, beta, eps)?;
        Ok(Self::maybe_attach(
            out,
            &[self, gamma, beta],
            Arc::new(BatchNorm2dBackward { input: self.clone(), gamma: gamma.clone(), beta: beta.clone(), mean, var, eps }),
        ))
    }
}

fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<(Tensor, ()), TensorError> {
    let is = input.shape();
    let ws = weight.shape();
    if is.len() != 4 || ws.len() != 4 {
        return Err(TensorError::Shape("conv2d expects NCHW input and OIHW weight".into()));
    }
    let (n, c_in, h, w) = (is[0], is[1], is[2], is[3]);
    let (c_out, c_w, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    if c_in != c_w {
        return Err(TensorError::Shape(format!("conv2d channel mismatch {c_in} vs {c_w}")));
    }
    let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
    let id = input.to_vec();
    let wd = weight.to_vec();
    let bias_v = bias.map(|b| b.to_vec());
    let mut out = vec![0.0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let hi = ho * stride + kh_i;
                                let wi = wo * stride + kw_i;
                                if hi < padding || wi < padding || hi >= padding + h || wi >= padding + w {
                                    continue;
                                }
                                let hi_in = hi - padding;
                                let wi_in = wi - padding;
                                let i_idx = ((ni * c_in + ci) * h + hi_in) * w + wi_in;
                                let w_idx = ((co * c_w + ci) * kh + kh_i) * kw + kw_i;
                                sum += id[i_idx] * wd[w_idx];
                            }
                        }
                    }
                    if let Some(ref b) = bias_v {
                        sum += b[co];
                    }
                    let o_idx = ((ni * c_out + co) * h_out + ho) * w_out + wo;
                    out[o_idx] = sum;
                }
            }
        }
    }
    let t = Tensor::from_vec(out, vec![n, c_out, h_out, w_out])?;
    Ok((t, ()))
}

fn conv2d_backward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    grad_output: &[f32],
    stride: usize,
    padding: usize,
) -> Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>), TensorError> {
    let is = input.shape();
    let ws = weight.shape();
    let (n, c_in, h, w) = (is[0], is[1], is[2], is[3]);
    let (c_out, _, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
    if grad_output.len() != n * c_out * h_out * w_out {
        return Err(TensorError::Shape("conv2d backward: grad_output size mismatch".into()));
    }
    let id = input.to_vec();
    let wd = weight.to_vec();
    let mut gx = vec![0.0f32; n * c_in * h * w];
    let mut gw = vec![0.0f32; c_out * c_in * kh * kw];
    let mut gb = bias.map(|_| vec![0.0f32; c_out]);

    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let o_idx = ((ni * c_out + co) * h_out + ho) * w_out + wo;
                    let g = grad_output[o_idx];
                    if let Some(ref mut b) = gb {
                        b[co] += g;
                    }
                    for ci in 0..c_in {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let hi = ho * stride + kh_i;
                                let wi = wo * stride + kw_i;
                                if hi < padding || wi < padding || hi >= padding + h || wi >= padding + w {
                                    continue;
                                }
                                let hi_in = hi - padding;
                                let wi_in = wi - padding;
                                let i_idx = ((ni * c_in + ci) * h + hi_in) * w + wi_in;
                                let w_idx = ((co * c_in + ci) * kh + kh_i) * kw + kw_i;
                                gx[i_idx] += wd[w_idx] * g;
                                gw[w_idx] += id[i_idx] * g;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((gx, gw, gb))
}

fn max_pool2d_forward(
    input: &Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<(Tensor, Vec<usize>), TensorError> {
    let is = input.shape();
    if is.len() != 4 {
        return Err(TensorError::Shape("max_pool2d expects NCHW".into()));
    }
    let (n, c, h, w) = (is[0], is[1], is[2], is[3]);
    let h_out = (h + 2 * padding).saturating_sub(kernel) / stride + 1;
    let w_out = (w + 2 * padding).saturating_sub(kernel) / stride + 1;
    let id = input.to_vec();
    let mut out = vec![0.0f32; n * c * h_out * w_out];
    let mut indices = vec![0usize; out.len()];
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut max_v = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for kh in 0..kernel {
                        for kw in 0..kernel {
                            let hi = ho * stride + kh;
                            let wi = wo * stride + kw;
                            if hi < padding || wi < padding || hi >= padding + h || wi >= padding + w {
                                continue;
                            }
                            let hi_in = hi - padding;
                            let wi_in = wi - padding;
                            let idx = ((ni * c + ci) * h + hi_in) * w + wi_in;
                            if id[idx] > max_v {
                                max_v = id[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    let o_idx = ((ni * c + ci) * h_out + ho) * w_out + wo;
                    out[o_idx] = max_v;
                    indices[o_idx] = max_idx;
                }
            }
        }
    }
    Ok((Tensor::from_vec(out, vec![n, c, h_out, w_out])?, indices))
}

fn batch_norm2d_forward(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Vec<f32>, Vec<f32>), TensorError> {
    let is = input.shape();
    if is.len() != 4 {
        return Err(TensorError::Shape("batch_norm2d expects NCHW".into()));
    }
    let (n, c, h, w) = (is[0], is[1], is[2], is[3]);
    if gamma.numel() != c || beta.numel() != c {
        return Err(TensorError::Shape("batch_norm2d gamma/beta must be [C]".into()));
    }
    let spatial = (h * w).max(1);
    let id = input.to_vec();
    let g = gamma.to_vec();
    let b = beta.to_vec();
    let mut mean = vec![0.0f32; c];
    let mut var = vec![0.0f32; c];
    let m_count = (n * spatial) as f32;
    for ci in 0..c {
        let mut m = 0.0f32;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    m += id[idx];
                }
            }
        }
        m /= m_count;
        mean[ci] = m;
        let mut v = 0.0f32;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let d = id[idx] - m;
                    v += d * d;
                }
            }
        }
        var[ci] = v / m_count;
    }
    let mut out = id.clone();
    for ci in 0..c {
        let inv_std = 1.0 / (var[ci] + eps).sqrt();
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    out[idx] = (id[idx] - mean[ci]) * inv_std * g[ci] + b[ci];
                }
            }
        }
    }
    Ok((Tensor::from_vec(out, is.to_vec())?, mean, var))
}

fn batch_norm2d_backward(
    input: &Tensor,
    gamma: &Tensor,
    grad_output: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), TensorError> {
    let is = input.shape();
    let (n, c, h, w) = (is[0], is[1], is[2], is[3]);
    let spatial = (h * w).max(1);
    let m = (n * spatial) as f32;
    let id = input.to_vec();
    let g = gamma.to_vec();
    let mut gx = vec![0.0f32; id.len()];
    let mut dgamma = vec![0.0f32; c];
    let mut dbeta = vec![0.0f32; c];

    for ci in 0..c {
        let inv_std = 1.0 / (var[ci] + eps).sqrt();
        let mut sum_dy = 0.0f32;
        let mut sum_dy_xhat = 0.0f32;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let xhat = (id[idx] - mean[ci]) * inv_std;
                    let dy = grad_output[idx];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                    dbeta[ci] += dy;
                    dgamma[ci] += dy * xhat;
                }
            }
        }
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let xhat = (id[idx] - mean[ci]) * inv_std;
                    let dy = grad_output[idx];
                    gx[idx] = (g[ci] * inv_std / m) * (m * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }
        }
    }
    Ok((gx, dgamma, dbeta))
}

struct Conv2dBackward {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl GradFn for Conv2dBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let (gx, gw, gb) =
            conv2d_backward(&self.input, &self.weight, self.bias.as_ref(), grad_output, self.stride, self.padding)?;
        propagate(&self.input, &gx)?;
        propagate(&self.weight, &gw)?;
        if let (Some(bias), Some(gb)) = (&self.bias, gb) {
            propagate(bias, &gb)?;
        }
        Ok(())
    }
    fn parents(&self) -> Vec<&Tensor> {
        if let Some(ref b) = self.bias { vec![&self.input, &self.weight, b] } else { vec![&self.input, &self.weight] }
    }
}

struct MaxPool2dBackward {
    input: Tensor,
    indices: Vec<usize>,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl GradFn for MaxPool2dBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let _ = (self.kernel, self.stride, self.padding);
        let mut gx = vec![0.0f32; self.input.numel()];
        for (oi, &g) in grad_output.iter().enumerate() {
            if oi < self.indices.len() {
                gx[self.indices[oi]] += g;
            }
        }
        propagate(&self.input, &gx)
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct BatchNorm2dBackward {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    mean: Vec<f32>,
    var: Vec<f32>,
    eps: f32,
}

impl GradFn for BatchNorm2dBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let (gx, dgamma, dbeta) =
            batch_norm2d_backward(&self.input, &self.gamma, grad_output, &self.mean, &self.var, self.eps)?;
        propagate(&self.input, &gx)?;
        propagate(&self.gamma, &dgamma)?;
        propagate(&self.beta, &dbeta)?;
        Ok(())
    }
    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input, &self.gamma, &self.beta]
    }
}
