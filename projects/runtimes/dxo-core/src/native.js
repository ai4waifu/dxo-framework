import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const pkgRoot = path.dirname(fileURLToPath(new URL('../package.json', import.meta.url)));

function platformTriple() {
    const { platform, arch } = process;
    if (platform === 'win32' && arch === 'x64') return 'win32-x64-msvc';
    if (platform === 'darwin' && arch === 'arm64') return 'darwin-arm64';
    if (platform === 'darwin' && arch === 'x64') return 'darwin-x64';
    if (platform === 'linux' && arch === 'x64') return 'linux-x64-gnu';
    return `${platform}-${arch}`;
}

function platformShort(triple = platformTriple()) {
    if (triple === 'win32-x64-msvc') return 'win32-x64';
    if (triple === 'linux-x64-gnu') return 'linux-x64';
    return triple;
}

/**
 * Resolve native `.node` via the platform optionalDependency (`@dxo/dxo-<short>`).
 * @returns {string}
 */
export function resolveNativePath() {
    const triple = platformTriple();
    const short = platformShort(triple);
    const name = `@dxo/dxo-${short}`;
    /** @type {string[]} */
    const candidates = [];
    try {
        const resolved = require.resolve(`${name}/package.json`);
        const dir = path.dirname(resolved);
        candidates.push(path.join(dir, `dxo.${triple}.node`), path.join(dir, 'dxo.node'));
    } catch {
        /* optional dep not installed */
    }
    candidates.push(path.join(pkgRoot, 'node_modules', name, `dxo.${triple}.node`), path.join(pkgRoot, 'node_modules', name, 'dxo.node'));
    for (const p of candidates) {
        if (existsSync(p)) return p;
    }
    throw new Error(
        `DXO native addon not found for ${name}. Run: pnpm build:native\n` + `Looked in:\n${candidates.map((c) => `  - ${c}`).join('\n')}`,
    );
}

let _native;

/** @returns {Record<string, unknown>} */
export function loadNative() {
    if (_native) return _native;
    const addonPath = resolveNativePath();
    _native = require(addonPath);
    return _native;
}
