//! Language-agnostic diagnostics for napi / TS (`DxoDiagnostic` wire).

use std::collections::BTreeMap;
use std::fmt;

use titan_hal::HalError;
use titan_types::{TitanError, TitanErrorKind};

use crate::tensor::TensorError;

/// Diagnostic severity (Living 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Hard failure.
    Error,
    /// Recoverable warning.
    Warning,
    /// Non-blocking advice.
    Advice,
}

impl Severity {
    /// Stable wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Advice => "advice",
        }
    }
}

/// Scalar argument values for catalog interpolation.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticValue {
    /// String argument.
    String(String),
    /// Integer argument.
    Int(i64),
    /// Floating argument.
    Float(f64),
    /// Boolean argument.
    Bool(bool),
}

impl DiagnosticValue {
    /// JSON-ish display for napi / catalogs (not localized).
    pub fn to_display(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
        }
    }
}

/// Stable diagnostic payload shared by core / lite / napi / CLI `--json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    /// Stable code (`DXO_*` / `DXO_TITAN_*`).
    pub code: &'static str,
    /// Severity.
    pub severity: Severity,
    /// Catalog interpolation args.
    pub args: BTreeMap<String, DiagnosticValue>,
    /// Machine-readable extras (shape, dtype, debug detail, …).
    pub details: BTreeMap<String, String>,
    /// Backend tag when relevant.
    pub backend: Option<&'static str>,
    /// Failing operation name when relevant.
    pub operation: Option<&'static str>,
    /// English developer fallback message (not a second translation source).
    pub message_dev: String,
}

impl Diagnostic {
    /// Build an error-severity diagnostic with a developer message.
    pub fn error(code: &'static str, message_dev: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Error,
            args: BTreeMap::new(),
            details: BTreeMap::new(),
            backend: None,
            operation: None,
            message_dev: message_dev.into(),
        }
    }

    /// Attach a string arg.
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.insert(key.into(), DiagnosticValue::String(value.into()));
        self
    }

    /// Attach a detail entry.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Set backend label.
    pub fn with_backend(mut self, backend: &'static str) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set operation label.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message_dev)
    }
}

/// Map [`TitanErrorKind`] → stable `DXO_TITAN_*` code.
pub fn titan_kind_to_code(kind: TitanErrorKind) -> &'static str {
    match kind {
        TitanErrorKind::DeviceNotFound => "DXO_TITAN_DEVICE_NOT_FOUND",
        TitanErrorKind::DeviceLost => "DXO_TITAN_DEVICE_LOST",
        TitanErrorKind::CrossDevice => "DXO_TITAN_CROSS_DEVICE",
        TitanErrorKind::CrossStream => "DXO_TITAN_CROSS_STREAM",
        TitanErrorKind::EventWaitFailed => "DXO_TITAN_EVENT_WAIT_FAILED",
        TitanErrorKind::AllocationFailed => "DXO_TITAN_ALLOCATION_FAILED",
        TitanErrorKind::InvalidAbi => "DXO_TITAN_INVALID_ABI",
        TitanErrorKind::KernelUnsupported => "DXO_TITAN_KERNEL_UNSUPPORTED",
        TitanErrorKind::KernelLaunchFailed => "DXO_TITAN_KERNEL_LAUNCH_FAILED",
        TitanErrorKind::ReadbackFailed => "DXO_TITAN_READBACK_FAILED",
        TitanErrorKind::UploadFailed => "DXO_TITAN_UPLOAD_FAILED",
        TitanErrorKind::BackendUnavailable => "DXO_TITAN_BACKEND_UNAVAILABLE",
        TitanErrorKind::UnknownError => "DXO_TITAN_UNKNOWN",
    }
}

/// Build a diagnostic from a Titan error.
pub fn from_titan_error(err: &TitanError, backend: Option<&'static str>) -> Diagnostic {
    let mut d = Diagnostic::error(titan_kind_to_code(err.kind()), err.to_string()).with_arg("kind", err.kind().as_str());
    if let Some(detail) = err.detail() {
        d = d.with_detail("debug", detail);
    }
    if let Some(b) = backend {
        d = d.with_backend(b);
    }
    d
}

/// Build a diagnostic from a HAL error via Titan kind heuristics.
pub fn from_hal_error(err: &HalError, backend: Option<&'static str>) -> Diagnostic {
    let titan = err.to_titan_error();
    let mut d = from_titan_error(&titan, backend);
    d.operation = Some(err.operation);
    d
}

impl TensorError {
    /// Prefer structured diagnostics for new call sites.
    pub fn from_diagnostic(d: Diagnostic) -> Self {
        Self::Diagnostic(Box::new(d))
    }

    /// Map HAL failure through Titan kinds.
    pub fn from_hal(err: HalError, backend: &'static str) -> Self {
        Self::from_diagnostic(from_hal_error(&err, Some(backend)))
    }

    /// Map Titan error.
    pub fn from_titan(err: TitanError, backend: &'static str) -> Self {
        Self::from_diagnostic(from_titan_error(&err, Some(backend)))
    }

    /// Device / backend unavailable (preview CPU-only or missing adapter).
    pub fn device_unavailable(device: &str) -> Self {
        Self::from_diagnostic(
            Diagnostic::error(
                "DXO_DEVICE_UNAVAILABLE",
                format!("device '{device}' is not available in this preview"),
            )
            .with_arg("device", device)
            .with_detail("requestedDevice", device)
            .with_operation("to"),
        )
    }

    /// Unknown device tag in parse path.
    pub fn unknown_device(tag: &str) -> Self {
        Self::from_diagnostic(
            Diagnostic::error(
                "DXO_BACKEND_UNAVAILABLE",
                format!("unknown device '{tag}' (supported: cpu, cuda)"),
            )
            .with_arg("requested", tag)
            .with_arg("available", "cpu,cuda")
            .with_operation("parse"),
        )
    }

    /// Non-scalar tensor where a scalar is required.
    pub fn non_scalar(shape: &[usize], operation: &'static str) -> Self {
        let shape_s = format!("{shape:?}");
        Self::from_diagnostic(
            Diagnostic::error(
                "DXO_TENSOR_NON_SCALAR",
                format!("{operation} requires a scalar tensor, got shape {shape_s}"),
            )
            .with_arg("shape", shape_s.clone())
            .with_arg("operation", operation)
            .with_detail("shape", shape_s)
            .with_operation(operation),
        )
    }

    /// Shape / numel mismatch.
    pub fn invalid_shape(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::from_diagnostic(
            Diagnostic::error("DXO_TENSOR_INVALID_SHAPE", message.clone()).with_detail("debug", message),
        )
    }

    /// Project any `TensorError` to a [`Diagnostic`] (legacy variants get default codes).
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::Diagnostic(d) => d.as_ref().clone(),
            Self::Shape(msg) => Diagnostic::error("DXO_TENSOR_INVALID_SHAPE", msg.clone()).with_detail("debug", msg.clone()),
            Self::Broadcast(msg) => {
                Diagnostic::error("DXO_TENSOR_BROADCAST_INCOMPATIBLE", msg.clone()).with_detail("debug", msg.clone())
            }
            Self::Autograd(msg) => {
                let code = if msg.to_ascii_lowercase().contains("scalar") {
                    "DXO_TENSOR_NON_SCALAR"
                } else {
                    "DXO_AUTOGRAD_FAILED"
                };
                Diagnostic::error(code, msg.clone()).with_detail("debug", msg.clone())
            }
            Self::Device(msg) => {
                let lower = msg.to_ascii_lowercase();
                let code = if lower.contains("unavailable") || lower.contains("no cuda") {
                    "DXO_TITAN_BACKEND_UNAVAILABLE"
                } else if lower.contains("upload") {
                    "DXO_TITAN_UPLOAD_FAILED"
                } else if lower.contains("readback") || lower.contains("download") {
                    "DXO_TITAN_READBACK_FAILED"
                } else if lower.contains("mismatch") {
                    "DXO_TITAN_CROSS_DEVICE"
                } else {
                    "DXO_DEVICE_ERROR"
                };
                Diagnostic::error(code, msg.clone())
                    .with_detail("debug", msg.clone())
                    .with_backend("cuda")
            }
        }
    }
}
