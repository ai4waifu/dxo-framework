# DXO Framework

**Developer preview** — API unstable (`0.0.x`). TypeScript-first deep learning for Node.js: Rust engine, napi-rs bindings.

## Quick start

```bash
pnpm install
pnpm build
pnpm verify -- smoke
```

```typescript
import { backend, tensor, version } from '@dxo/core';
import { Linear } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';

console.log(version(), backend()); // e.g. 0.1.0 titan-cpu

const model = new Linear(2, 1);
const x = tensor([1, 0, 0, 1], [2, 2]);
const y = model.forward(x);
const loss = y.mean(); // scalar shape [1]
```

## Verify gates

```bash
pnpm verify -- smoke
pnpm verify -- tensor-cpu
pnpm verify -- autograd-fd
pnpm verify -- mnist-linear
pnpm verify -- g3-contract
pnpm verify -- data-iter
pnpm verify -- serialize-roundtrip
pnpm verify -- trainer-loop
pnpm verify -- lite-webgpu-smoke
```

## Layout

| Path | npm package |
|------|-------------|
| `projects/compilers/dxo-core` | *(Rust crate)* |
| `projects/compilers/dxo-napi` | *(Rust napi cdylib)* |
| `projects/runtimes/dxo-core` | `@dxo/core` |
| `projects/runtimes/dxo-nn` | `@dxo/nn` |
| `projects/runtimes/dxo-optimizer` | `@dxo/optimizer` |
| `projects/runtimes/dxo-data` | `@dxo/data` |
| `projects/runtimes/dxo-serialize` | `@dxo/serialize` |
| `projects/runtimes/dxo-train` | `@dxo/train` |
| `projects/runtimes/dxo-lite` | `@dxo/lite` |
| `projects/runtimes/dxo` | `@dxo/dxo` (CLI) |
| `projects/runtimes/dxo-<platform>` | `@dxo/dxo-<platform>` |

## License

MIT OR Apache-2.0
