//! Typed host image pixel carrier (Living `11` / napi bridge).

use crate::diagnostic::Diagnostic;
use crate::tensor::{Tensor, TensorError};

/// Pixel storage dtype for [`HostImageBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDtype {
    /// 8-bit unsigned.
    U8,
    /// 16-bit unsigned.
    U16,
    /// 32-bit float.
    F32,
}

impl ImageDtype {
    /// Stable wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::F32 => "f32",
        }
    }

    /// Bytes per pixel channel.
    pub fn bytes_per_channel(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Memory layout of pixel channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    /// Height × width × channels.
    Hwc,
    /// Channels × height × width.
    Chw,
}

impl ImageLayout {
    /// Stable wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hwc => "HWC",
            Self::Chw => "CHW",
        }
    }
}

/// Interpretation hint — not a full ICC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// RGB interleaved.
    Rgb,
    /// RGBA interleaved.
    Rgba,
    /// Single channel luminance.
    Gray,
    /// BGR interleaved.
    Bgr,
    /// Unknown / unspecified.
    Unknown,
}

impl ColorSpace {
    /// Stable wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Rgba => "rgba",
            Self::Gray => "gray",
            Self::Bgr => "bgr",
            Self::Unknown => "unknown",
        }
    }
}

/// Alpha channel interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    /// Fully opaque (no alpha channel).
    Opaque,
    /// Straight alpha.
    Straight,
    /// Premultiplied alpha.
    Premultiplied,
    /// Explicitly no alpha.
    None,
}

impl AlphaMode {
    /// Stable wire label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Opaque => "opaque",
            Self::Straight => "straight",
            Self::Premultiplied => "premultiplied",
            Self::None => "none",
        }
    }
}

/// Host-resident image pixels + layout metadata.
#[derive(Debug, Clone)]
pub struct HostImageBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channel count (e.g. 3 for RGB).
    pub channels: u32,
    /// Pixel dtype.
    pub dtype: ImageDtype,
    /// Channel layout.
    pub layout: ImageLayout,
    /// Color space hint.
    pub color_space: ColorSpace,
    /// Alpha interpretation.
    pub alpha_mode: AlphaMode,
    /// Row-major pixel bytes (length validated in constructors).
    pub data: Vec<u8>,
}

impl HostImageBuffer {
    fn expected_len(width: u32, height: u32, channels: u32, dtype: ImageDtype) -> Result<usize, TensorError> {
        let w = width as usize;
        let h = height as usize;
        let c = channels as usize;
        if w == 0 || h == 0 || c == 0 {
            return Err(TensorError::invalid_shape("ImageBuffer: width/height/channels must be positive"));
        }
        w.checked_mul(h)
            .and_then(|n| n.checked_mul(c))
            .and_then(|n| n.checked_mul(dtype.bytes_per_channel()))
            .ok_or_else(|| TensorError::invalid_shape("ImageBuffer: pixel byte length overflow"))
    }

    /// Construct from explicit HWC (or CHW) pixel bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pixels(
        width: u32,
        height: u32,
        channels: u32,
        dtype: ImageDtype,
        layout: ImageLayout,
        color_space: ColorSpace,
        alpha_mode: AlphaMode,
        data: Vec<u8>,
    ) -> Result<Self, TensorError> {
        let expected = Self::expected_len(width, height, channels, dtype)?;
        if data.len() != expected {
            return Err(TensorError::invalid_shape(format!(
                "ImageBuffer: byte length {} != expected {expected}",
                data.len()
            )));
        }
        Ok(Self { width, height, channels, dtype, layout, color_space, alpha_mode, data })
    }

    /// Decode PNG bytes into RGB/RGBA u8 HWC (first codec provider slice).
    pub fn decode_png(bytes: &[u8]) -> Result<Self, TensorError> {
        use png::{ColorType, Decoder, Transformations};
        let mut decoder = Decoder::new(bytes);
        decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|e| TensorError::invalid_shape(format!("PNG decode: {e}")))?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| TensorError::invalid_shape(format!("PNG decode: {e}")))?;
        buf.truncate(info.buffer_size());
        let width = info.width;
        let height = info.height;
        let (channels, color_space, alpha_mode) = match info.color_type {
            ColorType::Rgb => (3, ColorSpace::Rgb, AlphaMode::Opaque),
            ColorType::Rgba => (4, ColorSpace::Rgba, AlphaMode::Straight),
            ColorType::Grayscale => (1, ColorSpace::Gray, AlphaMode::None),
            ColorType::GrayscaleAlpha => (2, ColorSpace::Gray, AlphaMode::Straight),
            other => {
                return Err(TensorError::invalid_shape(format!("PNG decode: unsupported color type {other:?}")));
            }
        };
        Ok(Self {
            width,
            height,
            channels,
            dtype: ImageDtype::U8,
            layout: ImageLayout::Hwc,
            color_space,
            alpha_mode,
            data: buf,
        })
    }

    /// Convert to NCHW f32 tensor (vision default). U8 may be normalized to `[0,1]`.
    pub fn to_tensor_nchw(&self, normalize: bool) -> Result<Tensor, TensorError> {
        if self.dtype != ImageDtype::U8 {
            return Err(TensorError::invalid_shape(format!(
                "ImageBuffer.toTensor: v0 supports u8 only, got {}",
                self.dtype.as_str()
            )));
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let c = self.channels as usize;
        let hw = h * w;
        let mut out = vec![0.0f32; c * hw];
        match self.layout {
            ImageLayout::Hwc => {
                for y in 0..h {
                    for x in 0..w {
                        let base = (y * w + x) * c;
                        for ch in 0..c {
                            let v = self.data[base + ch] as f32;
                            out[ch * hw + y * w + x] = if normalize { v / 255.0 } else { v };
                        }
                    }
                }
            }
            ImageLayout::Chw => {
                for ch in 0..c {
                    let plane = &self.data[ch * hw..(ch + 1) * hw];
                    for (i, &b) in plane.iter().enumerate() {
                        let v = b as f32;
                        out[ch * hw + i] = if normalize { v / 255.0 } else { v };
                    }
                }
            }
        }
        Tensor::from_vec(out, vec![1, c, h, w])
    }

    /// Decode encoded image bytes (v0: PNG only).
    pub fn decode(bytes: &[u8], format: Option<&str>) -> Result<Self, TensorError> {
        let hint = format.map(|s| s.trim().to_ascii_lowercase());
        match hint.as_deref() {
            Some("png") | None => Self::decode_png(bytes),
            Some("jpeg") | Some("jpg") | Some("webp") => Err(TensorError::from_diagnostic(
                Diagnostic::error(
                    "DXO_IMAGE_DECODE_UNSUPPORTED",
                    format!("ImageBuffer decode: format '{hint:?}' not wired yet (use PNG or raw pixels)"),
                )
                .with_arg("format", hint.as_deref().unwrap_or(""))
                .with_detail("hint", "png"),
            )),
            Some(other) => Err(TensorError::from_diagnostic(
                Diagnostic::error("DXO_IMAGE_DECODE_UNKNOWN", format!("ImageBuffer decode: unknown format '{other}'"))
                    .with_arg("format", other),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hwc_u8_to_nchw_normalize() {
        // 2x2 RGB white
        let data = vec![255u8; 2 * 2 * 3];
        let buf = HostImageBuffer::from_pixels(
            2,
            2,
            3,
            ImageDtype::U8,
            ImageLayout::Hwc,
            ColorSpace::Rgb,
            AlphaMode::Opaque,
            data,
        )
        .unwrap();
        let t = buf.to_tensor_nchw(true).unwrap();
        assert_eq!(t.shape(), &[1, 3, 2, 2]);
        assert!((t.to_vec()[0] - 1.0).abs() < 1e-6);
    }
}
