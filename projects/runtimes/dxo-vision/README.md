# @dxo/vision

**Developer preview — API unstable.** Not a closed vision-classify gate.

Composable vision surface: `Image` / `ImageBatch`, `ResNet` (backbone / features), `LinearHead` / `Classifier` / `compose` (logits), `defineLabelSpace` / `decodeClassification` (labels).

```bash
pnpm add @dxo/vision
```

```ts
import { ResNet, LinearHead, compose } from '@dxo/vision';

const backbone = new ResNet({ depth: 18 });
const model = compose(backbone, new LinearHead({ output: 10 }));
```

Pretrained weight assets ship from the separate `@dxo/resnet` package (external repo). This package only defines networks and task adapters.
