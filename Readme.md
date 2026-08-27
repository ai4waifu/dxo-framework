# DXO Framework

**Deep Learning for the JavaScript Era** — TypeScript-first API, Rust core engine, napi-rs native bindings.

## Quick start

```bash
pnpm install
pnpm build:native
pnpm smoke
```

```typescript
import {version, Tensor} from '@dxo/core';

console.log(version()); // "0.1.0"
new Tensor();
```

## Layout

| Path                               | npm package           |
|------------------------------------|-----------------------|
| `projects/compilers/dxo-core`      | *(Rust crate)*        |
| `projects/compilers/dxo-napi`      | *(Rust napi cdylib)*  |
| `projects/runtimes/dxo-core`       | `@dxo/core`           |
| `projects/runtimes/dxo-lite`       | `@dxo/lite`           |
| `projects/runtimes/dxo`            | `@dxo/dxo` (CLI)      |
| `projects/runtimes/dxo-<platform>` | `@dxo/dxo-<platform>` |

pnpm workspace：`projects/runtimes/*`（见 `pnpm-workspace.yaml`）。

## License

MIT OR Apache-2.0
