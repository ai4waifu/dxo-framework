# 🏠 @dxo/homepage

DXO 产品首页（VMZ 静态站），含 `@dxo/lite` probe 与 **Playground**。

## 命令

仓库根目录：

```bash
pnpm build:lite-wasm       # 编译 @dxo/lite-unknown-wasm32（本地测 playground 必做）
pnpm homepage:dev          # --force 重建 wasm → public/，再 vmz dev
pnpm homepage              # install + wasm + TS + 静态构建
pnpm homepage:test:lite    # Node 侧 @dxo/lite 合同（需已 build:lite-wasm）
```

`projects/homepage` 内：

```bash
pnpm test:lite
pnpm build
```

- `/` — 首页 + lite probe 卡片
- `/lite` — `@dxo/lite` 详细测试面
- `/playground` — 浏览器 matmul/add 交互（读 `/dxo_lite_bg.wasm`）

本地测 playground 前必须：

```bash
pnpm build:lite-wasm
# 或直接 pnpm homepage:dev（会自动 stage）
```

构建产物：

- `dist/` — VMZ 完整构建树（开发/诊断）
- `dist/cdn/` — **静态托管根**（仅 HTML、assets、robots、sitemap、404）
- `public/dxo_lite_bg.wasm` — 由 `stage-homepage-wasm` 从 lite-unknown-wasm32 复制

## Cloudflare Pages

与控制台配置对齐：

| 项 | 值 |
|----|-----|
| Git 存储库 | `ai4waifu/dxo-framework` |
| 根目录 | `/`（仓库根） |
| 构建命令 | `pnpm homepage` |
| 构建输出目录 | `projects/homepage/dist/cdn` |
| Node.js | 22 |

可选构建环境变量：

| 变量 | 用途 |
|------|------|
| `VMZ_SITE_ORIGIN` | canonical / sitemap / Open Graph 的站点 origin（例：`https://dxo.example.com`） |

生产分支若在控制台设为 `master`，需将 `dev` 合并进 `master` 或把生产分支改为 `dev`。

## 依赖

| 包 | 来源 |
|----|------|
| `@dxo/lite` | workspace（依赖 `@dxo/lite-unknown-wasm32`） |
| `@vmz/core` / `@vmz/vmz` | npm |
