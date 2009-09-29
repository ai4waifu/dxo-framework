# @dxo/optimizer

**Developer preview — API unstable (`0.0.x`).**

Pure TypeScript optimizers over `@dxo/core` tensors.

## Contract (G3 preview)

| API | Behavior |
|-----|----------|
| `SGD(lr)` / `Adam(lr, …)` | `lr` must be `> 0` |
| `step(params)` | Returns **new** `requiresGrad: true` leaves; does **not** mutate inputs in place |
| Missing `grad` | Parameter returned unchanged |
| Caller duty | Reassign onto the module (`Linear.loadParameters` or field write) |

```typescript
import { SGD } from '@dxo/optimizer';
import { Linear } from '@dxo/nn';

const model = new Linear(2, 1);
const opt = new SGD(0.1);
// after backward:
model.loadParameters(opt.step(model.parameters()));
```

Adam keeps moment buffers by **parameter index** in the `step(params)` array; keep a stable parameter order across steps.
