# @dxo/lite

Browser / Worker / Service Worker 的 WebGPU/WASM runtime（M0 为 TS stub；M6 起接 wgpu lite-engine）。

- 独立 runtime，不是 `@dxo/core` 的去 napi 移植版
- GPU 路径：TypeScript facade → wasm-bindgen → Rust + wgpu → WebGPU
- 降级：`WebGPU → 显式允许的 CPU/WASM fallback → error`；**不支持 WebGL tensor 后端**
- 异步边界为一等公民（device 初始化、queue 提交、readback、跨设备迁移）

不依赖 `@dxo/dxo-*` 原生平台包与 Node.js API。
