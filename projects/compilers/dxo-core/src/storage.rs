//! Shared tensor storage: host `f32` vectors or opaque Titan device handles.

use std::fmt;
use std::sync::Arc;

use titan_tensor::TensorHandle;

/// Reference-counted payload shared by tensor views.
#[derive(Clone)]
pub enum Storage {
    /// Contiguous host `f32` buffer.
    Host(Arc<Vec<f32>>),
    /// Opaque Titan device buffer (CUDA residency path).
    Device(TensorHandle),
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(data) => f.debug_tuple("Host").field(&data.len()).finish(),
            Self::Device(handle) => f
                .debug_struct("Device")
                .field("shape", &handle.shape())
                .field("dtype", &handle.dtype())
                .finish(),
        }
    }
}

impl Storage {
    /// Allocate zero-filled host storage with `len` elements.
    pub fn zeros(len: usize) -> Self {
        Self::Host(Arc::new(vec![0.0; len]))
    }

    /// Take ownership of a vector as shared host storage.
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self::Host(Arc::new(data))
    }

    /// Wrap an opaque Titan device handle.
    pub fn from_device(handle: TensorHandle) -> Self {
        Self::Device(handle)
    }

    /// Host payload when resident on CPU.
    pub fn host_data(&self) -> Option<&Arc<Vec<f32>>> {
        match self {
            Self::Host(data) => Some(data),
            Self::Device(_) => None,
        }
    }

    /// Device handle when resident on GPU.
    pub fn device_handle(&self) -> Option<&TensorHandle> {
        match self {
            Self::Host(_) => None,
            Self::Device(handle) => Some(handle),
        }
    }

    /// Whether `other` points at the same host allocation (device handles never alias as equal).
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Host(a), Self::Host(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
