# @dxo/lite

**Developer preview — API unstable (`0.0.x`).**

Browser / Worker runtime facade. **Not** a napi / `@dxo/core` port.

## Contract (0.0.8+)

| API | Behavior |
|-----|----------|
| `createRuntime(opts?)` | **Async** init; probes WebGPU adapter only (no `GPUDevice` in TS) |
| `TITAN_WGPU_READY` | `false` until WASM → dxo-core → Titan wgpu is linked |
| `fallback: 'cpu' \| 'error'` | Default `'error'`; explicit host f32 path — **never WebGL** |
| `runtime.capabilities` | `backend`, `webgpu`, `titanWgpuReady`, `webglTensorBackend: false` |
| `Tensor` ops | `matmul` / `add` / `toCpu` return **Promises** (host f32 until Titan ready) |

```typescript
import { createRuntime } from '@dxo/lite';

const rt = await createRuntime({ fallback: 'cpu' });
const a = await rt.tensor([1, 2, 3, 4], [2, 2]);
const b = await rt.ones([2, 2]);
const c = await a.matmul(b);
console.log(rt.capabilities.backend, rt.capabilities.titanWgpuReady, await c.toArray());
rt.destroy();
```

## Non-goals (this gate)

- TS retention of `GPUAdapter` / `GPUDevice`
- WebGL tensor backend (forbidden)
- DXO-owned wgpu kernels (Titan owns GPU execution)
- Training / autograd parity with `@dxo/core`
