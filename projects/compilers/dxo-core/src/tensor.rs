//! CPU tensor storage and elementary ops (M1 thin slice).

use std::fmt;

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

/// Row-major float32 tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Zeros tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    /// Construct from flat data; `data.len()` must equal shape product.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(TensorError::Shape(format!(
                "data length {} does not match shape product {n} (shape={shape:?})",
                data.len()
            )));
        }
        Ok(Self { data, shape })
    }

    /// Shape dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Underlying storage (row-major).
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// View with the same storage and a new shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        if n != self.numel() {
            return Err(TensorError::Shape(format!(
                "cannot reshape numel {} into product {n} (shape={shape:?})",
                self.numel()
            )));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: shape.to_vec(),
        })
    }

    /// Element-wise add with broadcast (trailing dims; bias vector on last axis).
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape == other.shape {
            return Ok(Self {
                data: self
                    .data
                    .iter()
                    .zip(&other.data)
                    .map(|(a, b)| a + b)
                    .collect(),
                shape: self.shape.clone(),
            });
        }
        if other.shape.len() == 1 && !self.shape.is_empty() {
            let last = *self.shape.last().expect("non-empty shape");
            if other.shape[0] == last {
                let mut out = self.data.clone();
                for chunk in out.chunks_mut(last) {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        *slot += other.data[j];
                    }
                }
                return Ok(Self {
                    data: out,
                    shape: self.shape.clone(),
                });
            }
        }
        Err(TensorError::Broadcast(format!(
            "add: incompatible shapes {:?} and {:?}",
            self.shape, other.shape
        )))
    }

    /// Matrix multiply for rank-2 tensors: `[m,k] @ [k,n] -> [m,n]`.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::Shape(
                "matmul requires rank-2 tensors".into(),
            ));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];
        if k != k2 {
            return Err(TensorError::Shape(format!(
                "matmul inner dims mismatch: {k} vs {k2}"
            )));
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Ok(Self {
            data: out,
            shape: vec![m, n],
        })
    }

    /// Element-wise ReLU.
    #[must_use]
    pub fn relu(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.max(0.0)).collect(),
            shape: self.shape.clone(),
        }
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
        assert_eq!(y.data(), &[11.0, 22.0, 13.0, 24.0]);
    }
}
