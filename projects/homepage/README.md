# 🏠 @dxo/homepage

DXO 产品首页（VMZ 静态站），含钉死版本的 `@dxo/lite` probe。首页**不跟** monorepo 最新 lite 特性；Cloudflare Pages **不能**编 Rust/wasm。

## 命令

仓库根目录：

```bash
pnpm homepage:cf           # CF / 无 Rust：只装 homepage 子图 + 静态构建
pnpm homepage:dev          # 本地：可选 force 编 wasm 再 vmz dev
pnpm homepage              # 本地全量：wasm + TS + homepage（需 wasm-pack）
pnpm homepage:test:lite    # Node 侧钉死版 @dxo/lite 合同
```

## Cloudflare Pages

| 项 | 值 |
|----|-----|
| 根目录 | `/`（仓库根） |
| 构建命令 | `pnpm homepage:cf` |
| 构建输出目录 | `projects/homepage/dist/cdn` |
| Node.js | 22 |

`CF_PAGES=1` 时 `stage-homepage-wasm` **跳过** wasm-pack（ENOENT）。

可选：`VMZ_SITE_ORIGIN`（canonical / OG）。

## 依赖

| 包 | 来源 |
|----|------|
| `@dxo/lite` | **npm `0.0.9`**（固定；升到 `0.0.10`+wasm 后再改钉） |
| `@vmz/core` / `@vmz/vmz` | npm |

发版后若要 playground 真 wasm：钉 `@dxo/lite@0.0.10`（含 `@dxo/lite-unknown-wasm32`），仍**不要**在 CF 上跑 `build:lite-wasm`。
