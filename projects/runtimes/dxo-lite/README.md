# @dxo/lite

**Developer preview — API unstable (`0.0.x`).**

Browser / Worker WebGPU runtime facade. **Not** a napi / `@dxo/core` port.

## Contract (0.0.8)

| API | Behavior |
|-----|----------|
| `createRuntime(opts?)` | **Async** init; acquires WebGPU when available |
| `fallback: 'cpu' \| 'error'` | Default `'error'`; explicit CPU host path only — **never WebGL** |
| `runtime.capabilities` | `backend`, `webgpu`, `webglTensorBackend: false`, dtype, features |
| `Tensor` ops | `matmul` / `add` / `toCpu` return **Promises** |
| Device compute | Host f32 subset in this slice; Rust `lite-engine` + wgpu kernels follow |

```typescript
import { createRuntime } from '@dxo/lite';

const rt = await createRuntime({ fallback: 'cpu' });
const a = await rt.tensor([1, 2, 3, 4], [2, 2]);
const b = await rt.ones([2, 2]);
const c = await a.matmul(b);
console.log(rt.capabilities.backend, await c.toArray());
rt.destroy();
```

## Non-goals (this gate)

- WebGL tensor backend (forbidden)
- Full wasm-bindgen / lite-engine ship
- Training / autograd parity with `@dxo/core`
