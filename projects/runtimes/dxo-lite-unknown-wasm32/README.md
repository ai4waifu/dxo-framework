# @dxo/lite-unknown-wasm32

**Internal** WASM artifact slot for `@dxo/lite`. Not a product API — do not import from apps.

Built from `projects/compilers/dxo-lite-wasm` via:

```bash
pnpm --filter @dxo/lite-unknown-wasm32 run build
# or: node scripts/build/lite-wasm.mjs
```

Outputs under `dist/`: `dxo_lite.js`, `dxo_lite_bg.wasm`, `dxo_lite.d.ts`.
