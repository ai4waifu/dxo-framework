# dxo-studio-tauri

Tauri shell → **`dxo-studio.exe`**. Embeds VMZ static build from `@dxo/studio` (`dist/`).

Separate from `dxo studio` CLI (VMZ watch + browser).

```bash
pnpm --filter @dxo/studio build:ui
cargo build -p dxo-studio-tauri --release
```
