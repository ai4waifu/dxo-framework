import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

/**
 * Ensure lite WASM is built and copy `dxo_lite_bg.wasm` into homepage `public/`
 * so browser playground can fetch `/dxo_lite_bg.wasm`.
 *
 * Pass `--force` or `DXO_FORCE_WASM_BUILD=1` to rebuild even when dist exists.
 */

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const wasmDist = path.join(root, 'projects/runtimes/dxo-lite-unknown-wasm32/dist/dxo_lite_bg.wasm');
const publicDir = path.join(root, 'projects/homepage/public');
const publicWasm = path.join(publicDir, 'dxo_lite_bg.wasm');
const force =
    process.argv.includes('--force') ||
    process.env.DXO_FORCE_WASM_BUILD === '1' ||
    process.env.DXO_FORCE_WASM_BUILD === 'true';

function buildLiteWasm() {
    console.log('stage-homepage-wasm: building @dxo/lite-unknown-wasm32…');
    const r = spawnSync(process.execPath, [path.join(root, 'scripts/build/lite-wasm.mjs')], {
        cwd: root,
        stdio: 'inherit',
        shell: false,
    });
    if (r.status !== 0) process.exit(r.status ?? 1);
}

if (force || !existsSync(wasmDist)) {
    buildLiteWasm();
}

if (!existsSync(wasmDist)) {
    console.error(`stage-homepage-wasm: missing ${wasmDist}`);
    process.exit(1);
}

mkdirSync(publicDir, { recursive: true });
copyFileSync(wasmDist, publicWasm);
console.log(`stage-homepage-wasm: ${publicWasm}`);
