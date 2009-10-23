# 🏠 @dxo/homepage

DXO 产品首页（VMZ 静态站），页面与 Node 脚本中探测 `@dxo/lite`。

## 命令

仓库根目录：

```bash
pnpm homepage              # install + build + stage dist/cdn（CF Pages 同款）
pnpm homepage:test:lite    # Node 侧 @dxo/lite 合同
pnpm homepage:dev          # 本地开发
pnpm homepage:preview      # 预览 static 产物
```

`projects/homepage` 内：

```bash
pnpm test:lite
pnpm build
```

- `/` — 首页 + lite probe 卡片
- `/lite` — `@dxo/lite` 详细测试面

构建产物：

- `dist/` — VMZ 完整构建树（开发/诊断）
- `dist/cdn/` — **静态托管根**（仅 HTML、assets、robots、sitemap、404）

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
| `@dxo/lite` | workspace |
| `@vmz/core` / `@vmz/vmz` | npm |
