# @dxo/lite

Browser / Worker runtime facade for DXO tensors (WebGPU path + explicit CPU fallback).

| Symbol                 | Notes                                                               |
|------------------------|---------------------------------------------------------------------|
| `GPU_BACKEND_READY`    | `false` until WASM → dxo-core → GPU backend is linked               |
| `createRuntime`        | Promise barrier; never keeps `GPUDevice`                            |
| `runtime.capabilities` | `backend`, `webgpu`, `gpuBackendReady`, `webglTensorBackend: false` |

```ts
import {createRuntime} from '@dxo/lite';

const rt = await createRuntime({fallback: 'cpu'});
const a = rt.tensor([1, 2, 3, 4], [2, 2]);
const b = rt.tensor([5, 6, 7, 8], [2, 2]);
const c = a.matmul(b);
console.log(rt.capabilities.backend, rt.capabilities.gpuBackendReady, await c.toArray());
```

Constraints:

- WebGL is never a tensor backend
- DXO does not ship a second in-tree wgpu runtime; GPU execution is owned by the DXO-linked engine facade
