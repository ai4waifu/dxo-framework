# 🏋️ @dxo/train

**Developer preview — API unstable (`0.0.x`).**

Async training loop over `@dxo/core` / `@dxo/nn` / `@dxo/optimizer` / `@dxo/data` / `@dxo/serialize`.

## Contract (0.0.7 / G5)

| API | Behavior |
|-----|----------|
| `Trainer` | Owns model + optimizer + batch factory + epoch count |
| `fitIter` / `run` | `AsyncGenerator<TrainEvent>`; respects `AbortSignal` |
| `fit` | Consumes the event stream; returns a short summary |
| Checkpoint | Emits `encodeLinearState(model.state())` documents (no FS I/O) |
| Device | **CPU-only** in this slice (GPU spike not required for this gate) |

```typescript
import { batch, dataset } from '@dxo/data';
import { Linear } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';
import { Trainer } from '@dxo/train';

const model = new Linear(2, 1);
const trainer = new Trainer({
  model,
  optimizer: new SGD(0.1),
  epochs: 5,
  batches: () => batch(dataset(samples), { batchSize: 8 }),
});

for await (const event of trainer.fitIter({ signal })) {
  if (event.type === 'batch') console.log(event.loss);
  if (event.type === 'checkpoint') /* persist event.document */;
}
```

Not included: multi-GPU, distributed, early-stopping policies, `@dxo/inspect` adapters, real MNIST download.
