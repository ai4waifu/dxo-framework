# 💾 @dxo/serialize

**Developer preview — API unstable (`0.0.x`).**

Versioned module / tensor state codec for DXO. No Node FS dependency — encode to JSON string or plain objects; callers own persistence.

## Contract (0.0.5)

| API | Behavior |
|-----|----------|
| `STATE_FORMAT` / `STATE_VERSION` | `"dxo-state"` / `1` |
| `packTensors` / `unpackTensors` | Named `{ shape, data }` blobs |
| `encodeLinearState` / `decodeLinearState` | Interop with `@dxo/nn` `Linear.state()` |
| `encodeJson` / `decodeJson` | UTF-8 JSON document |

```typescript
import { encodeJson, encodeLinearState, decodeJson, decodeLinearState } from '@dxo/serialize';
import { Linear } from '@dxo/nn';

const model = new Linear(2, 1);
const doc = encodeLinearState(model.state());
const text = encodeJson(doc);
model.loadState(decodeLinearState(decodeJson(text)));
```
