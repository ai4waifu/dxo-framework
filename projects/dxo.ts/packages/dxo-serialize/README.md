# @dxo/serialize

Canonical DXO model state serialization. `State` is the sole model/checkpoint state type and safetensors is the only first-class wire format.

```ts
import { encodeState, decodeState } from '@dxo/serialize';
const bytes = encodeState(state);
const restored = decodeState(bytes);
```

`encodeSafetensors` and `decodeSafetensors` are available for low-level interop. JSON documents and model-specific checkpoint codecs are intentionally not part of the runtime API.
