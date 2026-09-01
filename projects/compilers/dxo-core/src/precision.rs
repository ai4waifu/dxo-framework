//! Precision / quantization capability reporting (Living `12` — no silent fallback).

use crate::cuda;
use crate::diagnostic::Diagnostic;
use crate::tensor::TensorError;

/// Supported dtype labels for a capability group (stable strings, not localized).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecisionCapabilities {
    /// Active backend label (`cpu` / `cuda`).
    pub backend: &'static str,
    /// CUDA session probe.
    pub cuda_available: bool,
    /// Weight storage/compute dtypes actually available on this build.
    pub weights: Vec<&'static str>,
    /// Activation dtypes available for compute (honest — tags ≠ kernels).
    pub activations: Vec<&'static str>,
    /// Accumulation dtypes for matmul/reduction style ops.
    pub accumulation: Vec<&'static str>,
}

/// Probe runtime precision capabilities without fabricating int8/f16 kernels.
pub fn precision_capabilities() -> PrecisionCapabilities {
    let cuda_available = cuda::is_available();
    let backend = if cuda_available { "cuda" } else { "cpu" };
    // Preview: compute kernels are f32-hosted; f16/bf16 are dtype tags only until mixed-precision parity lands.
    PrecisionCapabilities {
        backend,
        cuda_available,
        weights: vec!["f32"],
        activations: vec!["f32"],
        accumulation: vec!["f32"],
    }
}

fn canonical_weight_dtype(requested: &str) -> Result<&'static str, TensorError> {
    match requested.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok("f32"),
        "f16" | "float16" => Ok("f16"),
        "bf16" | "bfloat16" => Ok("bf16"),
        "int8" | "i8" => Ok("int8"),
        "int4" | "i4" => Ok("int4"),
        other => Err(TensorError::from_diagnostic(
            Diagnostic::error("DXO_PRECISION_DTYPE_UNSUPPORTED", format!("unsupported weight dtype '{other}'"))
                .with_arg("requested", other),
        )),
    }
}

/// Resolve a requested weight dtype against capabilities.
pub fn resolve_weight_dtype(requested: &str, caps: &PrecisionCapabilities) -> Result<&'static str, TensorError> {
    let canonical = canonical_weight_dtype(requested)?;
    if caps.weights.iter().any(|d| *d == canonical) {
        return Ok(canonical);
    }
    Err(TensorError::from_diagnostic(
        Diagnostic::error(
            "DXO_PRECISION_DTYPE_UNAVAILABLE",
            format!(
                "weight dtype '{requested}' unavailable on backend '{}'",
                caps.backend
            ),
        )
        .with_arg("requested", requested)
        .with_arg("canonical", canonical)
        .with_arg("backend", caps.backend)
        .with_arg("supported", caps.weights.join(","))
        .with_backend(caps.backend),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_f32_only_on_preview() {
        let caps = precision_capabilities();
        assert_eq!(caps.weights, vec!["f32"]);
        assert_eq!(caps.activations, vec!["f32"]);
    }

    #[test]
    fn rejects_unavailable_weight_dtype() {
        let caps = precision_capabilities();
        assert!(resolve_weight_dtype("f16", &caps).is_err());
        assert_eq!(resolve_weight_dtype("f32", &caps).unwrap(), "f32");
        assert_eq!(resolve_weight_dtype("float32", &caps).unwrap(), "f32");
    }
}
