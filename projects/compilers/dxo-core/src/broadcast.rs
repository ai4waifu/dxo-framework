//! NumPy-style trailing-dimension broadcast helpers.

use crate::shape::numel;
use crate::tensor::TensorError;

/// Broadcast `a` and `b` shapes; trailing dimensions must be compatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = a.get(a.len().wrapping_sub(rank - i)).copied().unwrap_or(1);
        let db = b.get(b.len().wrapping_sub(rank - i)).copied().unwrap_or(1);
        if da != db && da != 1 && db != 1 {
            return Err(TensorError::Broadcast(format!("incompatible shapes for broadcast: {a:?} vs {b:?}")));
        }
        out.push(da.max(db));
    }
    Ok(out)
}

/// Linear offset into an operand expanded by broadcast rules.
pub fn broadcast_offset(index: &[usize], shape: &[usize], strides: &[i64], out_rank: usize) -> usize {
    let in_rank = shape.len();
    let mut offset = 0i64;
    for out_dim in 0..out_rank {
        let in_dim = out_dim as isize - out_rank as isize + in_rank as isize;
        if in_dim < 0 {
            continue;
        }
        let in_dim = in_dim as usize;
        let in_size = shape[in_dim];
        let coord = if in_size == 1 { 0 } else { index[out_dim] };
        offset += coord as i64 * strides[in_dim];
    }
    offset.max(0) as usize
}

/// Iterate all multi-indices for `shape` in row-major order.
pub fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let n = numel(shape).unwrap_or(0);
    if n == 0 {
        return;
    }
    let rank = shape.len();
    let mut index = vec![0usize; rank];
    for _ in 0..n {
        f(&index);
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_bias() {
        let out = broadcast_shapes(&[2, 2], &[2]).unwrap();
        assert_eq!(out, vec![2, 2]);
    }
}
