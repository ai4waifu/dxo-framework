/**
 * Ensure lite WASM is built and copy `dxo_lite_bg.wasm` into homepage `public/`
 * so browser playground can fetch `/dxo_lite_bg.wasm`.
 *
 * Cloudflare Pages sets `CF_PAGES=1` and cannot run wasm-pack — skip there.
 * Pass `--force` or `DXO_FORCE_WASM_BUILD=1` to rebuild even when lib wasm exists.
 */
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const wasmLib = path.join(root, 'projects/runtimes/dxo-lite-unknown-wasm32/lib/dxo_lite_bg.wasm');
const publicDir = path.join(root, 'projects/homepage/public');
const publicWasm = path.join(publicDir, 'dxo_lite_bg.wasm');
const force =
    process.argv.includes('--force') ||
    process.env.DXO_FORCE_WASM_BUILD === '1' ||
    process.env.DXO_FORCE_WASM_BUILD === 'true';
const skipRust =
    process.env.CF_PAGES === '1' ||
    process.env.DXO_HOMEPAGE_SKIP_WASM === '1' ||
    process.env.DXO_HOMEPAGE_SKIP_WASM === 'true';

if (skipRust) {
    if (existsSync(publicWasm)) {
        console.log(`stage-homepage-wasm: skip rust build (CF/skip); keep ${publicWasm}`);
    } else {
        console.log('stage-homepage-wasm: skip rust build (CF/skip); no public wasm (CPU fallback only)');
    }
    process.exit(0);
}

function buildLiteWasm() {
    console.log('stage-homepage-wasm: building @dxo/lite-unknown-wasm32…');
    const r = spawnSync(process.execPath, [path.join(root, 'scripts/build/lite-wasm.mjs')], {
        cwd: root,
        stdio: 'inherit',
        shell: false,
    });
    if (r.status !== 0) process.exit(r.status ?? 1);
}

if (force || !existsSync(wasmLib)) {
    buildLiteWasm();
}

if (!existsSync(wasmLib)) {
    console.error(`stage-homepage-wasm: missing ${wasmLib}`);
    process.exit(1);
}

mkdirSync(publicDir, { recursive: true });
copyFileSync(wasmLib, publicWasm);
console.log(`stage-homepage-wasm: ${publicWasm}`);
