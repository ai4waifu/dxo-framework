# @dxo/core

DXO 主入口包：TypeScript API + 通过 `optionalDependencies` 加载平台原生运行时。

```typescript
import { version, Tensor } from '@dxo/dxo-core';

console.log(version());
new Tensor();
```

开发：在 monorepo 根目录执行 `pnpm build:native` 编译当前平台的 `@dxo/dxo-<platform>` 包。
