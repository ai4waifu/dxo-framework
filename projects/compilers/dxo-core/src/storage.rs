//! Shared host storage for zero-copy tensor views.

use std::sync::Arc;

/// Reference-counted `f32` payload shared by tensor views.
#[derive(Clone, Debug)]
pub struct Storage {
    data: Arc<Vec<f32>>,
}

impl Storage {
    /// Allocate zero-filled storage with `len` elements.
    pub fn zeros(len: usize) -> Self {
        Self { data: Arc::new(vec![0.0; len]) }
    }

    /// Take ownership of a vector as shared storage.
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data: Arc::new(data) }
    }

    /// Shared backing buffer.
    pub fn data(&self) -> &Arc<Vec<f32>> {
        &self.data
    }

    /// Whether `other` points at the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}
