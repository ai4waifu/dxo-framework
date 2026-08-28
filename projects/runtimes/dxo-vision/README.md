# @dxo/vision

**Developer preview — API unstable.** Not a closed `vision-classify` gate.

Composable vision surface: `ResNet` (backbone / features), `LinearHead` / `Classifier` / `compose` (logits), `defineLabelSpace` / `decodeClassification` (labels).

ResNet-18 uses **DXO-native** state keys (`stem.*`, `stage{n}.block{i}.*`). Preview `forward` supports `32×32` NCHW → `[N,512]` features.

```bash
pnpm add @dxo/vision
```

```ts
import { ResNet, LinearHead, compose } from '@dxo/vision';

const backbone = new ResNet({ depth: 18 });
const model = compose(backbone, new LinearHead({ output: 10 }));
```

Pretrained weight assets / torch→DXO conversion live in the separate `@dxo/resnet` package.
