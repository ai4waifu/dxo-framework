//! Batch optimizer steps in Rust (avoid per-parameter host roundtrips from TypeScript).

use std::cmp::Ordering;

use crate::tensor::{DeviceKind, Tensor, TensorError};

fn is_positive_finite(x: f32) -> bool {
    x.is_finite() && x.partial_cmp(&0.0) == Some(Ordering::Greater)
}

fn is_open_unit_interval(x: f32) -> bool {
    x.is_finite() && x.partial_cmp(&0.0) == Some(Ordering::Greater) && x.partial_cmp(&1.0) == Some(Ordering::Less)
}

/// Clear gradients on every parameter leaf in one pass.
pub fn zero_grads(params: &[&Tensor]) {
    for p in params {
        p.zero_grad();
    }
}

/// Vanilla SGD: `p ← p - lr * grad` for each leaf that has a gradient.
///
/// Returns **new** `requires_grad` leaves (G3 contract). Leaves without grad are
/// cloned as fresh autograd leaves with the same values. CPU-only in this slice.
pub fn sgd_step(params: &[&Tensor], lr: f32) -> Result<Vec<Tensor>, TensorError> {
    if !is_positive_finite(lr) {
        return Err(TensorError::Autograd("SGD lr must be positive".into()));
    }
    let mut out = Vec::with_capacity(params.len());
    for p in params {
        ensure_cpu_leaf(p, "sgd_step")?;
        match p.grad() {
            None => out.push(clone_leaf(p)?),
            Some(g) => {
                let data = p.to_vec();
                if data.len() != g.len() {
                    return Err(TensorError::Shape(format!("SGD grad length mismatch: param={} grad={}", data.len(), g.len())));
                }
                let next: Vec<f32> = data.iter().zip(g.iter()).map(|(x, gi)| x - lr * gi).collect();
                out.push(Tensor::from_vec_grad(next, p.shape().to_vec(), true)?);
            }
        }
    }
    Ok(out)
}

/// Adam moment buffers keyed by parameter index (matches preview TS Adam).
#[derive(Debug, Default, Clone)]
pub struct AdamState {
    /// First moment per parameter.
    pub m: Vec<Vec<f32>>,
    /// Second moment per parameter.
    pub v: Vec<Vec<f32>>,
    /// Step count.
    pub t: u64,
}

/// Adam update; mutates `state` and returns new `requires_grad` leaves. CPU-only.
pub fn adam_step(
    params: &[&Tensor],
    state: &mut AdamState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) -> Result<Vec<Tensor>, TensorError> {
    if !is_positive_finite(lr) {
        return Err(TensorError::Autograd("Adam lr must be positive".into()));
    }
    if !is_open_unit_interval(beta1) || !is_open_unit_interval(beta2) {
        return Err(TensorError::Autograd("Adam betas must be in (0, 1)".into()));
    }
    if !is_positive_finite(eps) {
        return Err(TensorError::Autograd("Adam eps must be positive".into()));
    }

    state.t = state.t.saturating_add(1);
    let t = state.t as f32;
    if state.m.len() < params.len() {
        state.m.resize_with(params.len(), Vec::new);
        state.v.resize_with(params.len(), Vec::new);
    }

    let mut out = Vec::with_capacity(params.len());
    for (idx, p) in params.iter().enumerate() {
        ensure_cpu_leaf(p, "adam_step")?;
        match p.grad() {
            None => out.push(clone_leaf(p)?),
            Some(g) => {
                let data = p.to_vec();
                if data.len() != g.len() {
                    return Err(TensorError::Shape(format!(
                        "Adam grad length mismatch: param={} grad={}",
                        data.len(),
                        g.len()
                    )));
                }
                let m = &mut state.m[idx];
                let v = &mut state.v[idx];
                if m.len() != data.len() {
                    *m = vec![0.0; data.len()];
                    *v = vec![0.0; data.len()];
                }
                let mut next = vec![0.0; data.len()];
                for i in 0..data.len() {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                    let m_hat = m[i] / (1.0 - beta1.powf(t));
                    let v_hat = v[i] / (1.0 - beta2.powf(t));
                    next[i] = data[i] - (lr * m_hat) / (v_hat.sqrt() + eps);
                }
                out.push(Tensor::from_vec_grad(next, p.shape().to_vec(), true)?);
            }
        }
    }
    Ok(out)
}

/// `loss.backward()` then [`sgd_step`] in one engine call.
pub fn backward_sgd_step(loss: &Tensor, params: &[&Tensor], lr: f32) -> Result<Vec<Tensor>, TensorError> {
    loss.backward()?;
    sgd_step(params, lr)
}

fn ensure_cpu_leaf(p: &Tensor, op: &str) -> Result<(), TensorError> {
    if p.device() != DeviceKind::Cpu {
        return Err(TensorError::Device(format!("{op} is CPU-only in this preview slice (got {})", p.device().as_str())));
    }
    Ok(())
}

fn clone_leaf(p: &Tensor) -> Result<Tensor, TensorError> {
    Tensor::from_vec_grad(p.to_vec(), p.shape().to_vec(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn sgd_step_updates_leaf() {
        let w = Tensor::from_vec_grad(vec![1.0, 2.0], vec![2], true).unwrap();
        // Manually seed grad via a tiny graph: loss = sum(w) → grad ones.
        let loss = w.sum();
        loss.backward().unwrap();
        let next = sgd_step(&[&w], 0.5).unwrap();
        assert_eq!(next[0].to_vec(), vec![0.5, 1.5]);
        assert!(next[0].requires_grad());
    }

    #[test]
    fn zero_grads_clears() {
        let w = Tensor::from_vec_grad(vec![1.0], vec![1], true).unwrap();
        let loss = w.sum();
        loss.backward().unwrap();
        assert!(w.grad().is_some());
        zero_grads(&[&w]);
        assert!(w.grad().is_none());
    }

    #[test]
    fn adam_step_moves() {
        let w = Tensor::from_vec_grad(vec![1.0], vec![1], true).unwrap();
        let loss = w.sum();
        loss.backward().unwrap();
        let mut state = AdamState::default();
        let next = adam_step(&[&w], &mut state, 0.1, 0.9, 0.999, 1e-8).unwrap();
        assert!(next[0].to_vec()[0] < 1.0);
        assert_eq!(state.t, 1);
    }
}
