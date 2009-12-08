/**
 * Ensure lite WASM is available as homepage `public/dxo_lite_bg.wasm`.
 *
 * Local: may build via wasm-pack (`build:lite-wasm`).
 * Cloudflare Pages (`CF_PAGES=1`): no Rust — copy from installed `@dxo/lite-unknown-wasm32`
 * (pulled by pinned `@dxo/lite`), never spawn wasm-pack.
 */

import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const wasmLib = path.join(root, 'projects/runtimes/dxo-lite-unknown-wasm32/lib/dxo_lite_bg.wasm');
const publicDir = path.join(root, 'projects/homepage/public');
const publicWasm = path.join(publicDir, 'dxo_lite_bg.wasm');
const force = process.argv.includes('--force') || process.env.DXO_FORCE_WASM_BUILD === '1' || process.env.DXO_FORCE_WASM_BUILD === 'true';
const skipRust = process.env.CF_PAGES === '1' || process.env.DXO_HOMEPAGE_SKIP_WASM === '1' || process.env.DXO_HOMEPAGE_SKIP_WASM === 'true';

function publishedWasmCandidates() {
    return [
        path.join(root, 'projects/homepage/node_modules/@dxo/lite-unknown-wasm32/lib/dxo_lite_bg.wasm'),
        path.join(root, 'projects/homepage/node_modules/@dxo/lite/node_modules/@dxo/lite-unknown-wasm32/lib/dxo_lite_bg.wasm'),
        path.join(root, 'node_modules/@dxo/lite-unknown-wasm32/lib/dxo_lite_bg.wasm'),
        path.join(root, 'node_modules/@dxo/lite/node_modules/@dxo/lite-unknown-wasm32/lib/dxo_lite_bg.wasm'),
    ];
}

function stageWasm(from) {
    mkdirSync(publicDir, { recursive: true });
    copyFileSync(from, publicWasm);
    console.log(`stage-homepage-wasm: ${publicWasm} ← ${from}`);
}

if (skipRust) {
    const fromPub = publishedWasmCandidates().find((p) => existsSync(p));
    if (fromPub) {
        stageWasm(fromPub);
        process.exit(0);
    }
    if (existsSync(publicWasm)) {
        console.log(`stage-homepage-wasm: skip rust; keep existing ${publicWasm}`);
        process.exit(0);
    }
    console.warn('stage-homepage-wasm: CF/skip and no published wasm in node_modules — CPU fallback only (pin @dxo/lite >= 0.0.10)');
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

stageWasm(wasmLib);
