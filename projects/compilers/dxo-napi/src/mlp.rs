//! Boost `dl.*` 训练图在 Node 侧的会话封装。
//!
//! 图、反向与 Adam 全部在 Boost 的 `dl-graph` / `dl-train` 内完成；这里只做 Buffer 与
//! [`Tensor`] 的边界转换，不重建任何第二套张量或计算图。

use dl_nn::seeded_random;
use dl_tensor::Tensor;
use dl_train::{GraphAdam, GraphMlp as MlpGraph};
use napi::bindgen_prelude::*;
use napi_derive::napi;

fn dl_err(err: dl_types::DlError) -> Error {
    Error::from_reason(format!("DXO_DL: {err}"))
}

fn f32_values(buffer: &Buffer, kind: &str) -> Result<Vec<f32>> {
    if buffer.len() % 4 != 0 {
        return Err(Error::from_reason(format!("{kind} buffer byte length must be a multiple of 4")));
    }
    // SAFETY: 长度已校验为 4 的倍数，`Buffer` 在本调用期间存活，指针可按 f32 读取。
    let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const f32, buffer.len() / 4) };
    Ok(slice.to_vec())
}

fn f32_buffer(values: &[f32]) -> Buffer {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Buffer::from(bytes)
}

fn rows_of(buffer: &Buffer, columns: usize, kind: &str) -> Result<usize> {
    let values = f32_values(buffer, kind)?;
    if columns == 0 || values.len() % columns != 0 {
        return Err(Error::from_reason(format!("{kind} length must be a multiple of {columns}")));
    }
    Ok(values.len() / columns)
}

fn argmax(slice: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in slice.iter().enumerate() {
        if *value > slice[best] {
            best = index;
        }
    }
    best
}

/// 一轮训练的分类评估。
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    /// 平均交叉熵。
    pub mean_loss: f64,
    /// 逐行 argmax 命中率。
    pub accuracy: f64,
    /// 参与评估的批次数。
    pub batches: u32,
}

/// 参数摘要。
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// 参数名（与 `dl-nn` 的层下标一致，例如 `0.weight`）。
    pub name: String,
    /// 行主序形状。
    pub shape: Vec<u32>,
}

/// 固定批大小的图版 MLP 训练会话；数据按 `[rows, features]` / `[rows, classes]` 的 f32 字节传入。
#[napi]
pub struct GraphMlp {
    model: MlpGraph,
    optimizer: GraphAdam,
}

impl std::fmt::Debug for GraphMlp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GraphMlp")
            .field("sizes", &self.model.sizes())
            .field("batch_size", &self.model.batch_size())
            .finish()
    }
}

#[napi]
impl GraphMlp {
    /// 构造训练图与 Adam；`sizes` 为层宽，最后一维是类别数。
    #[napi(constructor)]
    pub fn new(sizes: Vec<u32>, batch_size: Option<u32>, seed: Option<f64>, learning_rate: Option<f64>) -> Result<Self> {
        let sizes: Vec<usize> = sizes.iter().map(|&value| value as usize).collect();
        if sizes.len() < 2 || sizes.contains(&0) {
            return Err(Error::from_reason("sizes must contain at least two positive layer widths"));
        }
        let batch = batch_size.unwrap_or(64);
        if batch == 0 {
            return Err(Error::from_reason("batch_size must be positive"));
        }
        let seed = seed.unwrap_or(0.0);
        if !seed.is_finite() || seed < 0.0 || seed > u64::MAX as f64 {
            return Err(Error::from_reason("seed must be a finite non-negative integer"));
        }
        let mut random = seeded_random(seed as u64);
        let model = MlpGraph::new(&sizes, batch as usize, &mut random).map_err(dl_err)?;
        let optimizer = GraphAdam::new(model.training(), learning_rate.unwrap_or(0.01) as f32);
        Ok(Self { model, optimizer })
    }

    /// 层宽。
    #[napi(getter)]
    pub fn sizes(&self) -> Vec<u32> {
        self.model.sizes().iter().map(|&value| u32::try_from(value).unwrap_or(u32::MAX)).collect()
    }

    /// 每批固定行数。
    #[napi(getter)]
    pub fn batch_size(&self) -> u32 {
        u32::try_from(self.model.batch_size()).unwrap_or(u32::MAX)
    }

    /// 跑一轮：只使用能凑满整批的行，返回平均交叉熵。
    #[napi]
    pub fn fit_epoch(&mut self, inputs: Buffer, targets: Buffer) -> Result<f64> {
        let (features, classes) = self.dimensions();
        let batch = self.model.batch_size();
        let rows = rows_of(&inputs, features, "inputs")?;
        if rows_of(&targets, classes, "targets")? != rows {
            return Err(Error::from_reason("inputs and targets must have the same row count"));
        }
        let batches = rows / batch;
        if batches == 0 {
            return Err(Error::from_reason("rows must cover at least one full batch"));
        }
        let input_values = f32_values(&inputs, "inputs")?;
        let target_values = f32_values(&targets, "targets")?;
        let mut total = 0.0f32;
        for index in 0..batches {
            let start = index * batch;
            let end = start + batch;
            let x =
                Tensor::from_vec(input_values[start * features..end * features].to_vec(), [batch, features]).map_err(dl_err)?;
            let y =
                Tensor::from_vec(target_values[start * classes..end * classes].to_vec(), [batch, classes]).map_err(dl_err)?;
            total += self.model.step_batch(&mut self.optimizer, x, y).map_err(dl_err)?;
        }
        Ok(f64::from(total / batches as f32))
    }

    /// 不更新参数，返回交叉熵与命中率；同样只统计整批。
    #[napi]
    pub fn evaluate(&self, inputs: Buffer, targets: Buffer) -> Result<ClassificationReport> {
        let (features, classes) = self.dimensions();
        let batch = self.model.batch_size();
        let rows = rows_of(&inputs, features, "inputs")?;
        if rows_of(&targets, classes, "targets")? != rows {
            return Err(Error::from_reason("inputs and targets must have the same row count"));
        }
        let batches = rows / batch;
        if batches == 0 {
            return Err(Error::from_reason("rows must cover at least one full batch"));
        }
        let input_values = f32_values(&inputs, "inputs")?;
        let target_values = f32_values(&targets, "targets")?;
        let mut total = 0.0f32;
        let mut correct = 0usize;
        for index in 0..batches {
            let start = index * batch;
            let end = start + batch;
            let x =
                Tensor::from_vec(input_values[start * features..end * features].to_vec(), [batch, features]).map_err(dl_err)?;
            let y =
                Tensor::from_vec(target_values[start * classes..end * classes].to_vec(), [batch, classes]).map_err(dl_err)?;
            total += self.model.loss_batch(x.clone(), y.clone()).map_err(dl_err)?;
            let logits = self.model.predict_batch(x).map_err(dl_err)?;
            let logits = logits.to_vec();
            let labels = y.to_vec();
            for row in 0..batch {
                let predicted = argmax(&logits[row * classes..(row + 1) * classes]);
                let expected = argmax(&labels[row * classes..(row + 1) * classes]);
                if predicted == expected {
                    correct += 1;
                }
            }
        }
        let samples = (batches * batch) as f64;
        Ok(ClassificationReport {
            mean_loss: f64::from(total / batches as f32),
            accuracy: correct as f64 / samples,
            batches: u32::try_from(batches).unwrap_or(u32::MAX),
        })
    }

    /// 返回整批的 logits（f32 字节，行主序 `[rows, classes]`）；行数必须是批大小的整数倍。
    #[napi]
    pub fn predict(&self, inputs: Buffer) -> Result<Buffer> {
        let (features, classes) = self.dimensions();
        let batch = self.model.batch_size();
        let rows = rows_of(&inputs, features, "inputs")?;
        let batches = rows / batch;
        if batches == 0 {
            return Err(Error::from_reason("rows must cover at least one full batch"));
        }
        let input_values = f32_values(&inputs, "inputs")?;
        let mut logits = Vec::with_capacity(batches * batch * classes);
        for index in 0..batches {
            let start = index * batch;
            let end = start + batch;
            let x =
                Tensor::from_vec(input_values[start * features..end * features].to_vec(), [batch, features]).map_err(dl_err)?;
            logits.extend(self.model.predict_batch(x).map_err(dl_err)?.to_vec());
        }
        Ok(f32_buffer(&logits))
    }

    /// 参数名与形状。
    #[napi]
    pub fn parameters(&self) -> Vec<ParameterInfo> {
        self.model
            .parameters()
            .into_iter()
            .map(|(name, shape)| ParameterInfo {
                name,
                shape: shape.iter().map(|&value| u32::try_from(value).unwrap_or(u32::MAX)).collect(),
            })
            .collect()
    }

    /// 读取参数值（f32 字节）。
    #[napi]
    pub fn parameter(&self, name: String) -> Result<Buffer> {
        let tensor = self.model.parameter(&name).ok_or_else(|| Error::from_reason(format!("unknown parameter '{name}'")))?;
        Ok(f32_buffer(&tensor.to_vec()))
    }

    /// 覆盖参数值；形状必须与建图时一致。
    #[napi]
    pub fn set_parameter(&mut self, name: String, data: Buffer) -> Result<()> {
        let expected = self
            .model
            .parameters()
            .into_iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, shape)| shape)
            .ok_or_else(|| Error::from_reason(format!("unknown parameter '{name}'")))?;
        let values = f32_values(&data, "parameter")?;
        if values.len() != expected.iter().product::<usize>() {
            return Err(Error::from_reason(format!("parameter '{name}' must have shape {expected:?}")));
        }
        let tensor = Tensor::from_vec(values, expected).map_err(dl_err)?;
        self.model.set_parameter(name, tensor).map_err(dl_err)
    }
}

impl GraphMlp {
    fn dimensions(&self) -> (usize, usize) {
        let sizes = self.model.sizes();
        (sizes[0], sizes[sizes.len() - 1])
    }
}
