//! Eager Define-by-Run tape helpers.
//!
//! Grad enablement uses a thread-local scope stack so `without_grad` restores
//! the previous flag after the closure; it is not a permanent process global.

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use crate::tensor::{Tensor, TensorError};

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether the current thread records ops onto the tape.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Set grad recording for this thread; returns the previous value.
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|cell| {
        let prev = cell.get();
        cell.set(enabled);
        prev
    })
}

/// Run `f` with tape recording disabled, then restore the prior flag.
pub fn without_grad<R>(f: impl FnOnce() -> R) -> R {
    let prev = set_grad_enabled(false);
    let out = f();
    set_grad_enabled(prev);
    out
}

/// Shared accumulator for a logical tensor's `.grad` buffer.
pub type GradSlot = Arc<Mutex<Option<Vec<f32>>>>;

/// Reverse-mode rule recorded on intermediate tensors.
pub trait GradFn: Send + Sync {
    /// Apply VJP given upstream gradient matching this node's output shape.
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError>;

    /// Parent tensors for topological walk (Define-by-Run).
    fn parents(&self) -> Vec<&Tensor>;
}

/// Allocate an empty grad slot for a leaf or intermediate that tracks grads.
pub fn new_grad_slot() -> GradSlot {
    Arc::new(Mutex::new(None))
}

/// Accumulate `delta` into `slot` (same length as the tensor).
pub fn accumulate_grad(slot: &GradSlot, delta: &[f32]) -> Result<(), TensorError> {
    let mut guard = slot.lock().map_err(|_| TensorError::Autograd("grad slot poisoned".into()))?;
    match guard.as_mut() {
        Some(buf) => {
            if buf.len() != delta.len() {
                return Err(TensorError::Autograd(format!(
                    "grad length mismatch: have {} vs delta {}",
                    buf.len(),
                    delta.len()
                )));
            }
            for (a, b) in buf.iter_mut().zip(delta.iter()) {
                *a += *b;
            }
        }
        None => *guard = Some(delta.to_vec()),
    }
    Ok(())
}

/// Reduce a broadcasted gradient back to `target_shape`.
pub fn sum_to_shape(grad: &[f32], out_shape: &[usize], target_shape: &[usize]) -> Result<Vec<f32>, TensorError> {
    use crate::broadcast::for_each_index;
    use crate::shape::{contiguous_strides, numel};

    let out_n = numel(out_shape)?;
    if grad.len() != out_n {
        return Err(TensorError::Autograd(format!("sum_to_shape: grad len {} != out numel {}", grad.len(), out_n)));
    }
    if out_shape == target_shape {
        return Ok(grad.to_vec());
    }

    let target_n = numel(target_shape)?;
    let mut acc = vec![0.0f32; target_n];
    let target_strides = contiguous_strides(target_shape);
    let out_rank = out_shape.len();
    let in_rank = target_shape.len();
    let mut flat = 0usize;
    for_each_index(out_shape, |index| {
        let mut offset = 0i64;
        for (out_dim, &coord_out) in index.iter().enumerate().take(out_rank) {
            let in_dim = out_dim as isize - out_rank as isize + in_rank as isize;
            if in_dim < 0 {
                continue;
            }
            let in_dim = in_dim as usize;
            let in_size = target_shape[in_dim];
            let coord = if in_size == 1 { 0 } else { coord_out };
            offset += coord as i64 * target_strides[in_dim];
        }
        acc[offset.max(0) as usize] += grad[flat];
        flat += 1;
    });
    Ok(acc)
}

/// Push `grad_output` into a parent tensor's slot and queue its `grad_fn`.
pub fn propagate(parent: &Tensor, grad_output: &[f32]) -> Result<(), TensorError> {
    if !parent.tracks_grad() {
        return Ok(());
    }
    if let Some(slot) = parent.grad_slot() {
        accumulate_grad(slot, grad_output)?;
    }
    Ok(())
}
