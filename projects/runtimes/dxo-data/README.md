# @dxo/data

**Developer preview — API unstable (`0.0.x`).**

Datasets and batching for DXO. Pure TypeScript over `@dxo/core`.

## Contract (0.0.5)

| API | Behavior |
|-----|----------|
| `Sample` | Flat `x` (+ optional `y`) with explicit `xShape` / `yShape` |
| `dataset(samples)` | Sync `Iterable<Sample>` |
| `batch(samples, { batchSize, dropLast? })` | Sync batches; stacks on a new leading dim |
| `batchAsync(samples, options)` | Same for `AsyncIterable` |
| `DataLoader` | Thin alias helper around `batch` / `batchAsync` |

Not included: shuffle sampler, multi-worker, Prefetch, real MNIST download.

```typescript
import { batch, dataset } from '@dxo/data';

const ds = dataset([
  { x: [1, 0], xShape: [2], y: [1], yShape: [1] },
  { x: [0, 1], xShape: [2], y: [0], yShape: [1] },
]);

for (const { x, y } of batch(ds, { batchSize: 2 })) {
  // x.shape === [2, 2]
}
```
