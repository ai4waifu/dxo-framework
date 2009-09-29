# @dxo/core

**Developer preview — API unstable (`0.0.x`).**

TypeScript entry for DXO. Loads the platform native addon via `optionalDependencies` (`@dxo/dxo-<platform>`).

## Install

```bash
npm install @dxo/core@0.0.4
```

## Contract (G3 preview)

| Surface | Behavior |
|---------|----------|
| `tensor(data, shape, { requiresGrad? })` | CPU leaf; `shape` is required (product must match flat length) |
| `zeros` / `ones` / `randn` | Same options; only `device: 'cpu'` in this slice |
| `backend()` | `"titan-cpu"` when Titan CPU facade is wired |
| `t.add/mul/matmul/relu/reshape/transpose/sum/mean` | Eager ops; record Tape when `requiresGrad` and grad enabled |
| `t.backward()` | **Scalar only** (`numel === 1`, typically shape `[1]`) |
| `t.grad` | Row-major `number[]` or `undefined` |
| `t.zeroGrad()` | Clears this leaf's accumulated grad |
| `t.detach()` | Values only — no tape / requiresGrad |
| `withoutGrad(fn)` | Disables tape for `fn`, then restores prior flag |

```typescript
import { tensor, withoutGrad } from '@dxo/core';

const x = tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
const y = x.matmul(tensor([0.5, 0, 0, 0.5], [2, 2])).sum();
y.backward();
console.log(x.grad);

withoutGrad(() => {
  // no tape
});
```

## Develop (monorepo)

```bash
pnpm build:native
pnpm verify -- g3-contract
```
