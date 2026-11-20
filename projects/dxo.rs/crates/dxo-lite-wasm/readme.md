# dxo-lite-wasm

`wasm-bindgen` ABI for `@dxo/lite` → `@dxo/lite-unknown-wasm32`.

**Interim:** host f32 kernels in WASM (no Titan wgpu yet; does not depend on `dxo-core` until wasm32 feature-gates land).

```bash
node scripts/build/lite-wasm.mjs
```
