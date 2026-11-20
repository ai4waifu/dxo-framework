//! CPU float32 tensor with shared storage, stride views, eager autograd, and CUDA residency.

use std::fmt;
use std::sync::Arc;

use rand::RngExt;

use crate::autograd::{GradFn, GradSlot, accumulate_grad, is_grad_enabled, new_grad_slot, propagate, sum_to_shape};
use crate::broadcast::{broadcast_offset, broadcast_shapes, for_each_index};
use crate::cuda;
use crate::diagnostic::Diagnostic;
use crate::dtype::DType;
use crate::shape::{Shape, Strides, contiguous_strides, is_contiguous, numel};
use crate::storage::Storage;

/// Placement for tensor values.
///
/// CUDA tensors prefer opaque Titan device buffers; host materialization happens
/// only on explicit readback (`to_vec` / `to('cpu')`) or CPU-only ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Host CPU storage.
    Cpu,
    /// NVIDIA CUDA (Driver API via titan-backend-cuda).
    Cuda,
}

impl DeviceKind {
    /// Parse `'cpu' | 'cuda'` (case-insensitive). Metal is not in this spike.
    pub fn parse(s: &str) -> Result<Self, TensorError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(TensorError::unknown_device(other)),
        }
    }

    /// Stable label for napi / diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// Shape / broadcast / matmul / autograd / device errors surfaced to napi.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Invalid shape or numel mismatch.
    Shape(String),
    /// Incompatible broadcast.
    Broadcast(String),
    /// Autograd graph / backward errors.
    Autograd(String),
    /// Device placement / CUDA facade errors.
    Device(String),
    /// Structured diagnostic (preferred wire path).
    Diagnostic(Box<Diagnostic>),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(msg) | Self::Broadcast(msg) | Self::Autograd(msg) | Self::Device(msg) => f.write_str(msg),
            Self::Diagnostic(d) => f.write_str(&d.message_dev),
        }
    }
}

impl std::error::Error for TensorError {}

/// Row-major float32 tensor view over shared [`Storage`], with optional tape edges.
#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    offset: usize,
    shape: Shape,
    strides: Strides,
    requires_grad: bool,
    grad_slot: Option<GradSlot>,
    grad_fn: Option<Arc<dyn GradFn>>,
    device: DeviceKind,
    dtype: DType,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape.0)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish_non_exhaustive()
    }
}

impl Tensor {
    /// Zeros tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let n = numel(shape).unwrap_or(0);
        Self::from_storage(Storage::zeros(n), 0, shape, false, None, None)
    }

    /// Ones tensor with the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        let n = numel(shape).unwrap_or(0);
        Self::from_storage(Storage::from_vec(vec![1.0; n]), 0, shape, false, None, None)
    }

    /// Standard-normal samples (Box–Muller); host RNG only.
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
        Self::from_storage(Storage::from_vec(data), 0, shape, false, None, None)
    }

    /// Construct from flat data; `data.len()` must equal shape product.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let n = numel(&shape)?;
        if data.len() != n {
            return Err(TensorError::invalid_shape(format!(
                "data length {} does not match shape product {} (shape={shape:?})",
                data.len(),
                n
            )));
        }
        Ok(Self::from_storage(Storage::from_vec(data), 0, &shape, false, None, None))
    }

    /// Leaf tensor that participates in autograd when `requires_grad` is true.
    pub fn from_vec_grad(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self, TensorError> {
        let mut t = Self::from_vec(data, shape)?;
        if requires_grad {
            t.requires_grad = true;
            t.grad_slot = Some(new_grad_slot());
        }
        Ok(t)
    }

    fn from_storage(
        storage: Storage,
        offset: usize,
        shape: &[usize],
        requires_grad: bool,
        grad_slot: Option<GradSlot>,
        grad_fn: Option<Arc<dyn GradFn>>,
    ) -> Self {
        Self::from_storage_on(storage, offset, shape, requires_grad, grad_slot, grad_fn, DeviceKind::Cpu)
    }

    pub(crate) fn from_storage_on(
        storage: Storage,
        offset: usize,
        shape: &[usize],
        requires_grad: bool,
        grad_slot: Option<GradSlot>,
        grad_fn: Option<Arc<dyn GradFn>>,
        device: DeviceKind,
    ) -> Self {
        let strides = contiguous_strides(shape);
        Self {
            storage,
            offset,
            shape: Shape(shape.to_vec()),
            strides: Strides(strides),
            requires_grad,
            grad_slot,
            grad_fn,
            device,
            dtype: DType::F32,
        }
    }

    /// Logical dtype tag.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Retag logical dtype (host payload stays f32 in this preview).
    pub fn set_dtype(&mut self, dtype: DType) {
        self.dtype = dtype;
    }

    /// Current device tag (`cpu` / `cuda`).
    pub fn device(&self) -> DeviceKind {
        self.device
    }

    /// Explicit device migration. CUDA path uploads into opaque Titan storage.
    /// Autograd leaves must be detached before moving to CUDA in this spike.
    pub fn to(&self, device: DeviceKind) -> Result<Self, TensorError> {
        if device == self.device {
            return Ok(self.clone());
        }
        match device {
            DeviceKind::Cpu => {
                let data = self.to_vec();
                Ok(Self::from_storage_on(Storage::from_vec(data), 0, self.shape(), false, None, None, DeviceKind::Cpu))
            }
            DeviceKind::Cuda => {
                if self.requires_grad || self.grad_fn.is_some() {
                    return Err(TensorError::from_diagnostic(
                        crate::diagnostic::Diagnostic::error(
                            "DXO_TENSOR_REQUIRES_DETACH",
                            "to('cuda') requires a detached tensor without requiresGrad in this preview",
                        )
                        .with_arg("device", "cuda")
                        .with_operation("to"),
                    ));
                }
                if !cuda::is_available() {
                    return Err(TensorError::from_diagnostic(
                        crate::diagnostic::Diagnostic::error(
                            "DXO_TITAN_BACKEND_UNAVAILABLE",
                            "CUDA unavailable (no driver/device)",
                        )
                        .with_arg("requested", "cuda")
                        .with_arg("available", "cpu")
                        .with_backend("cuda")
                        .with_operation("to"),
                    ));
                }
                let data = self.to_vec();
                let handle = cuda::upload_f32(self.shape(), &data)?;
                Ok(Self::from_storage_on(Storage::from_device(handle), 0, self.shape(), false, None, None, DeviceKind::Cuda))
            }
        }
    }

    /// Shape dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape.0
    }

    /// Element strides in elements.
    pub fn strides(&self) -> &[i64] {
        &self.strides.0
    }

    /// Whether this tensor (or its parents) require gradient tracking.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Shared grad slot, if any.
    pub fn grad_slot(&self) -> Option<&GradSlot> {
        self.grad_slot.as_ref()
    }

    /// True when this node should receive gradients.
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad && self.grad_slot.is_some()
    }

    /// Dense copy of accumulated gradient, if present.
    pub fn grad(&self) -> Option<Vec<f32>> {
        self.grad_slot.as_ref().and_then(|slot| slot.lock().ok().and_then(|g| g.clone()))
    }

    /// Clear accumulated gradient on this tensor.
    pub fn zero_grad(&self) {
        if let Some(slot) = &self.grad_slot {
            if let Ok(mut g) = slot.lock() {
                *g = None;
            }
        }
    }

    /// Clone values without tape / requires_grad (keeps device tag).
    pub fn detach(&self) -> Self {
        Self::from_storage_on(self.storage.clone(), self.offset, self.shape(), false, None, None, self.device)
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

    /// Dense copy of tensor elements (device tensors perform explicit readback).
    pub fn to_vec(&self) -> Vec<f32> {
        match &self.storage {
            Storage::Device(handle) if self.is_contiguous() && self.offset == 0 => {
                cuda::download_f32(handle).unwrap_or_else(|_| Vec::new())
            }
            Storage::Host(data) if self.is_contiguous() && self.offset == 0 => data[..self.numel()].to_vec(),
            Storage::Device(_) => {
                // Non-contiguous device views: download dense then gather (rare in preview).
                let dense = match self.storage.device_handle() {
                    Some(h) => cuda::download_f32(h).unwrap_or_default(),
                    None => Vec::new(),
                };
                let host = Self::from_storage_on(Storage::from_vec(dense), 0, self.shape(), false, None, None, DeviceKind::Cpu);
                let mut out = vec![0.0; self.numel()];
                let mut idx = 0usize;
                for_each_index(self.shape(), |coords| {
                    out[idx] = host.read(coords);
                    idx += 1;
                });
                out
            }
            Storage::Host(_) => {
                let out_len = self.numel();
                let mut out = vec![0.0; out_len];
                self.for_each(|idx, value| {
                    out[idx] = value;
                });
                out
            }
        }
    }

    /// Host buffer for CPU kernels; downloads once when storage is device-resident.
    pub(crate) fn host_arc(&self) -> Result<Arc<Vec<f32>>, TensorError> {
        match &self.storage {
            Storage::Host(data) => Ok(data.clone()),
            Storage::Device(handle) => {
                let data = cuda::download_f32(handle)?;
                Ok(Arc::new(data))
            }
        }
    }

    fn linear_offset(&self, coords: &[usize]) -> usize {
        let mut off = self.offset as i64;
        for (c, s) in coords.iter().zip(self.strides()) {
            off += *c as i64 * *s;
        }
        off.max(0) as usize
    }

    fn read(&self, coords: &[usize]) -> f32 {
        match &self.storage {
            Storage::Host(data) => data[self.linear_offset(coords)],
            Storage::Device(_) => {
                // Fallback path — prefer contiguous device ops that avoid per-element reads.
                let host = self.host_arc().unwrap_or_else(|_| Arc::new(Vec::new()));
                host.get(self.linear_offset(coords)).copied().unwrap_or(0.0)
            }
        }
    }

    fn for_each(&self, mut f: impl FnMut(usize, f32)) {
        let shape = self.shape();
        let mut idx = 0usize;
        for_each_index(shape, |coords| {
            f(idx, self.read(coords));
            idx += 1;
        });
    }

    /// Storage offset in elements (crate-internal).
    pub(crate) fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Ensure a device handle for CUDA ops (uploads host storage if needed).
    pub(crate) fn device_handle_for_op(&self) -> Result<titan_tensor::TensorHandle, TensorError> {
        if self.device != DeviceKind::Cuda {
            return Err(TensorError::Device("device handle requested for non-CUDA tensor".into()));
        }
        if !self.is_contiguous() || self.offset != 0 {
            return Err(TensorError::Device("CUDA op requires contiguous tensor (offset 0)".into()));
        }
        match &self.storage {
            Storage::Device(handle) => Ok(handle.clone()),
            Storage::Host(data) => cuda::upload_f32(self.shape(), data.as_slice()),
        }
    }

    pub(crate) fn maybe_attach(mut out: Self, parents: &[&Self], grad_fn: Arc<dyn GradFn>) -> Self {
        let track = is_grad_enabled() && parents.iter().any(|p| p.requires_grad);
        if track {
            out.requires_grad = true;
            out.grad_slot = Some(new_grad_slot());
            out.grad_fn = Some(grad_fn);
        }
        out
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
        let base = if !self.is_contiguous() {
            Self::from_storage_on(Storage::from_vec(self.to_vec()), 0, shape, false, None, None, self.device)
        } else {
            Self {
                storage: self.storage.clone(),
                offset: self.offset,
                shape: Shape(shape.to_vec()),
                strides: Strides(contiguous_strides(shape)),
                requires_grad: false,
                grad_slot: None,
                grad_fn: None,
                device: self.device,
                dtype: self.dtype,
            }
        };
        Ok(Self::maybe_attach(base, &[self], Arc::new(ReshapeBackward { input: self.clone(), out_shape: shape.to_vec() })))
    }

    /// Transpose rank-2 tensor (zero-copy view).
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape().len() != 2 {
            return Err(TensorError::Shape("transpose requires rank-2 tensor".into()));
        }
        let shape = vec![self.shape()[1], self.shape()[0]];
        let strides = vec![self.strides()[1], self.strides()[0]];
        let base = Self {
            storage: self.storage.clone(),
            offset: self.offset,
            shape: Shape(shape),
            strides: Strides(strides),
            requires_grad: false,
            grad_slot: None,
            grad_fn: None,
            device: self.device,
            dtype: self.dtype,
        };
        Ok(Self::maybe_attach(base, &[self], Arc::new(TransposeBackward { input: self.clone() })))
    }

    /// Element-wise add with NumPy-style broadcast.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.device == DeviceKind::Cuda
            && other.device == DeviceKind::Cuda
            && self.shape() == other.shape()
            && self.is_contiguous()
            && other.is_contiguous()
            && self.offset == 0
            && other.offset == 0
            && !self.requires_grad
            && !other.requires_grad
        {
            let a = self.device_handle_for_op()?;
            let b = other.device_handle_for_op()?;
            let handle = cuda::add_handles(&a, &b, self.shape())?;
            return Ok(Self::from_storage_on(
                Storage::from_device(handle),
                0,
                self.shape(),
                false,
                None,
                None,
                DeviceKind::Cuda,
            ));
        }
        if self.device == DeviceKind::Cuda
            && other.device == DeviceKind::Cuda
            && self.shape() != other.shape()
            && !self.requires_grad
            && !other.requires_grad
        {
            let out_shape = broadcast_shapes(self.shape(), other.shape())?;
            let a = self.device_handle_for_op()?;
            let b = other.device_handle_for_op()?;
            let handle = cuda::broadcast_add_handles(&a, &b, &out_shape)?;
            return Ok(Self::from_storage_on(Storage::from_device(handle), 0, &out_shape, false, None, None, DeviceKind::Cuda));
        }
        let out = self.binary_broadcast(other, |a, b| a + b)?;
        Ok(Self::maybe_attach(out, &[self, other], Arc::new(AddBackward { a: self.clone(), b: other.clone() })))
    }

    /// Element-wise multiply with NumPy-style broadcast.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        let out = self.binary_broadcast(other, |a, b| a * b)?;
        Ok(Self::maybe_attach(out, &[self, other], Arc::new(MulBackward { a: self.clone(), b: other.clone() })))
    }

    pub(crate) fn binary_broadcast(&self, other: &Self, op: fn(f32, f32) -> f32) -> Result<Self, TensorError> {
        if self.device != other.device {
            return Err(TensorError::Device(format!(
                "broadcast op device mismatch: {} vs {}",
                self.device.as_str(),
                other.device.as_str()
            )));
        }
        let a = self.host_arc()?;
        let b = other.host_arc()?;
        let out_shape = broadcast_shapes(self.shape(), other.shape())?;
        let rank = out_shape.len();
        let mut out = vec![0.0; numel(&out_shape)?];
        let mut oi = 0usize;
        for_each_index(&out_shape, |index| {
            let li = self.offset + broadcast_offset(index, self.shape(), self.strides(), rank);
            let ri = other.offset + broadcast_offset(index, other.shape(), other.strides(), rank);
            out[oi] = op(a[li], b[ri]);
            oi += 1;
        });
        // CPU kernels materialize host results even when tagged cuda (autograd path).
        Ok(Self::from_storage_on(Storage::from_vec(out), 0, &out_shape, false, None, None, DeviceKind::Cpu)
            .pipe_device(self.device))
    }

    fn pipe_device(mut self, device: DeviceKind) -> Self {
        if device == DeviceKind::Cuda {
            // Keep host mirror but retag — prefer explicit device ops above for residency.
            self.device = DeviceKind::Cuda;
        }
        self
    }

    /// Matrix multiply for rank-2 tensors: `[m,k] @ [k,n] -> [m,n]`.
    /// When both operands are on `cuda`, runs titan CUDA `gemm.f32` (contiguous f32 only).
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(TensorError::Shape("matmul requires rank-2 tensors".into()));
        }
        if self.device != other.device {
            return Err(TensorError::Device(format!(
                "matmul device mismatch: {} vs {} (call .to on both)",
                self.device.as_str(),
                other.device.as_str()
            )));
        }
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = other.shape()[0];
        let n = other.shape()[1];
        if k != k2 {
            return Err(TensorError::Shape(format!("matmul inner dims mismatch: {k} vs {k2}")));
        }

        if self.device == DeviceKind::Cuda {
            if !self.is_contiguous() || self.offset != 0 || !other.is_contiguous() || other.offset != 0 {
                return Err(TensorError::Device("CUDA matmul requires contiguous rank-2 inputs (offset 0)".into()));
            }
            let a = self.device_handle_for_op()?;
            let b = other.device_handle_for_op()?;
            let handle = cuda::gemm_handles(&a, &b, m, n)?;
            return Ok(Self::from_storage_on(Storage::from_device(handle), 0, &[m, n], false, None, None, DeviceKind::Cuda));
        }

        let a = self.to_vec();
        let b = other.to_vec();
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
        let base = Self::from_storage_on(Storage::from_vec(out), 0, &[m, n], false, None, None, self.device);
        Ok(Self::maybe_attach(base, &[self, other], Arc::new(MatmulBackward { a: self.clone(), b: other.clone() })))
    }

    /// Element-wise ReLU.
    pub fn relu(&self) -> Self {
        let data = self.to_vec().into_iter().map(|x| x.max(0.0)).collect();
        let base = Self::from_storage_on(Storage::from_vec(data), 0, self.shape(), false, None, None, self.device);
        Self::maybe_attach(base, &[self], Arc::new(ReluBackward { input: self.clone() }))
    }

    /// Sum all elements to a scalar tensor of shape `[1]`.
    pub fn sum(&self) -> Self {
        let s: f32 = self.to_vec().iter().sum();
        let base = Self::from_storage_on(Storage::from_vec(vec![s]), 0, &[1], false, None, None, self.device);
        Self::maybe_attach(base, &[self], Arc::new(SumBackward { input: self.clone() }))
    }

    /// Reverse-mode autodiff from a scalar output (shape `[1]` or numel 1).
    pub fn backward(&self) -> Result<(), TensorError> {
        if self.numel() != 1 {
            return Err(TensorError::non_scalar(self.shape(), "backward"));
        }
        if !self.requires_grad {
            return Err(TensorError::Autograd("tensor does not require grad".into()));
        }
        if let Some(slot) = &self.grad_slot {
            accumulate_grad(slot, &[1.0])?;
        }
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: Vec<*const ()> = Vec::new();
        topo_collect(self, &mut order, &mut visited);
        for node in order.into_iter().rev() {
            let gy = node
                .grad_slot
                .as_ref()
                .and_then(|s| s.lock().ok().and_then(|g| g.clone()))
                .unwrap_or_else(|| vec![0.0; node.numel()]);
            if let Some(gf) = &node.grad_fn {
                gf.apply(&gy)?;
            }
        }
        Ok(())
    }
}

fn topo_collect(t: &Tensor, order: &mut Vec<Tensor>, visited: &mut Vec<*const ()>) {
    let Some(gf) = &t.grad_fn else {
        return;
    };
    let ptr = Arc::as_ptr(gf).cast::<()>();
    if visited.contains(&ptr) {
        return;
    }
    visited.push(ptr);
    for parent in gf.parents() {
        topo_collect(parent, order, visited);
    }
    order.push(t.clone());
}

// --- concrete VJPs ---------------------------------------------------------

struct AddBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for AddBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let out_shape = broadcast_shapes(self.a.shape(), self.b.shape())?;
        let ga = sum_to_shape(grad_output, &out_shape, self.a.shape())?;
        let gb = sum_to_shape(grad_output, &out_shape, self.b.shape())?;
        propagate(&self.a, &ga)?;
        propagate(&self.b, &gb)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

struct MulBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let out_shape = broadcast_shapes(self.a.shape(), self.b.shape())?;
        let rank = out_shape.len();
        let a_host = self.a.host_arc()?;
        let b_host = self.b.host_arc()?;
        let mut ga_full = vec![0.0f32; grad_output.len()];
        let mut gb_full = vec![0.0f32; grad_output.len()];
        let mut oi = 0usize;
        for_each_index(&out_shape, |index| {
            let li = self.a.offset + broadcast_offset(index, self.a.shape(), self.a.strides(), rank);
            let ri = self.b.offset + broadcast_offset(index, self.b.shape(), self.b.strides(), rank);
            let av = a_host[li];
            let bv = b_host[ri];
            let g = grad_output[oi];
            ga_full[oi] = g * bv;
            gb_full[oi] = g * av;
            oi += 1;
        });
        let ga = sum_to_shape(&ga_full, &out_shape, self.a.shape())?;
        let gb = sum_to_shape(&gb_full, &out_shape, self.b.shape())?;
        propagate(&self.a, &ga)?;
        propagate(&self.b, &gb)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

struct MatmulBackward {
    a: Tensor,
    b: Tensor,
}

impl GradFn for MatmulBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let m = self.a.shape()[0];
        let k = self.a.shape()[1];
        let n = self.b.shape()[1];
        let a = self.a.to_vec();
        let b = self.b.to_vec();
        let mut ga = vec![0.0f32; m * k];
        let mut gb = vec![0.0f32; k * n];
        for i in 0..m {
            for p in 0..k {
                let mut s = 0.0f32;
                for j in 0..n {
                    s += grad_output[i * n + j] * b[p * n + j];
                }
                ga[i * k + p] = s;
            }
        }
        for p in 0..k {
            for j in 0..n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += a[i * k + p] * grad_output[i * n + j];
                }
                gb[p * n + j] = s;
            }
        }
        propagate(&self.a, &ga)?;
        propagate(&self.b, &gb)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.a, &self.b]
    }
}

struct ReluBackward {
    input: Tensor,
}

impl GradFn for ReluBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let x = self.input.to_vec();
        let mut gx = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            gx[i] = if x[i] > 0.0 { grad_output[i] } else { 0.0 };
        }
        propagate(&self.input, &gx)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct SumBackward {
    input: Tensor,
}

impl GradFn for SumBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let g = grad_output.first().copied().unwrap_or(0.0);
        let gx = vec![g; self.input.numel()];
        propagate(&self.input, &gx)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct ReshapeBackward {
    input: Tensor,
    out_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let _ = &self.out_shape;
        if grad_output.len() != self.input.numel() {
            return Err(TensorError::Autograd("reshape backward numel mismatch".into()));
        }
        propagate(&self.input, grad_output)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}

struct TransposeBackward {
    input: Tensor,
}

impl GradFn for TransposeBackward {
    fn apply(&self, grad_output: &[f32]) -> Result<(), TensorError> {
        let m = self.input.shape()[0];
        let n = self.input.shape()[1];
        let mut gx = vec![0.0f32; m * n];
        for i in 0..n {
            for j in 0..m {
                gx[j * n + i] = grad_output[i * m + j];
            }
        }
        propagate(&self.input, &gx)?;
        Ok(())
    }

    fn parents(&self) -> Vec<&Tensor> {
        vec![&self.input]
    }
}
