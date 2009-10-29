# @dxo/studio

DXO training workbench (developer preview). **WebUI is VMZ** — not a separate Vue app.

## Two entry points

| Entry | What it does |
|-------|----------------|
| **`dxo studio`** | VMZ **watch serve** + Rust inspect API → **browser** |
| **`dxo-studio.exe`** | Tauri desktop GUI (embedded VMZ static build + native API) |

| Layer | Path |
|-------|------|
| Rust API | `projects/compilers/dxo-studio` |
| Rust GUI | `projects/compilers/dxo-studio-tauri` → **`dxo-studio.exe`** |
| TS launcher | `launcher/` → `dist/` |
| VMZ WebUI | package root (`src/*.vmz`, `vmz.config.ts`) |

```bash
pnpm exec dxo studio
pnpm --filter @dxo/studio dev          # VMZ only
pnpm --filter @dxo/studio build:ui     # static profile for Tauri embed

cargo build -p dxo-studio-tauri --release   # → target/release/dxo-studio.exe
```
