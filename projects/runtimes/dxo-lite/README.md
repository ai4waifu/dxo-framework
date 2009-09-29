# @dxo/lite

**Developer preview — API unstable (`0.0.x`).**

Browser / Worker WebGPU/WASM runtime facade. Through **0.0.7** this package remains a **stub/probe**; wgpu lite-engine lands as the **0.0.8** thin gate.

- Not a napi/`@dxo/core` port
- Target path: TS facade → wasm-bindgen → Rust + wgpu → WebGPU
- Fallback: WebGPU → explicit CPU/WASM → error; **no WebGL tensor backend**
