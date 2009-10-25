//! Logical dtype tags for tensors (compute path still f32-hosted in this preview).

use std::fmt;

/// Supported logical dtypes (storage may still be f32-backed until mixed-precision lands).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit float (default compute dtype).
    #[default]
    F32,
    /// 16-bit float tag.
    F16,
    /// brain float16 tag.
    BF16,
    /// 64-bit integer tag.
    I64,
    /// Boolean tag.
    Bool,
}

impl DType {
    /// Stable label for napi / diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I64 => "i64",
            Self::Bool => "bool",
        }
    }

    /// Parse dtype string (case-insensitive).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "f16" | "float16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "i64" | "int64" => Ok(Self::I64),
            "bool" | "boolean" => Ok(Self::Bool),
            other => Err(format!("unknown dtype '{other}'")),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
