# 🧱 @dxo/nn

**Developer preview — API unstable (`0.0.x`).**

Pure TypeScript modules over `@dxo/core`.

## Contract (G3 preview)

| Type | Behavior |
|------|----------|
| `Linear(in, out)` | **`y = x @ W + b` only** — no implicit ReLU |
| `Relu` / `relu(x)` | Explicit nonlinearity |
| `Sequential(layers)` | Chains `forward` |
| `parameters()` | Collects `Tensor` fields (and nested modules) in enumeration order |
| `zeroGrad()` | Calls `zeroGrad` on each parameter leaf |
| `Linear.loadParameters([W, b])` | Reassign leaves after `optimizer.step` |
| `Linear.state` / `loadState` | Plain `{ shape, data }` snapshots |

```typescript
import { Linear, Relu, Sequential } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';

const model = new Sequential([new Linear(2, 4), new Relu(), new Linear(4, 1)]);
const opt = new SGD(0.05);
// ... loss.backward()
model.layers[0] // update Linear leaves via loadParameters after step
```

Training step pattern:

```typescript
model.zeroGrad();
const loss = /* scalar Tensor */;
loss.backward();
const linear = model.layers[0] as Linear;
linear.loadParameters(opt.step(linear.parameters()));
```
