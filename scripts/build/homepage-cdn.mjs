/**
 * Stage VMZ static-cdn deploy root into projects/homepage/dist/cdn.
 * Cloudflare Pages output directory: projects/homepage/dist/cdn
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const distDir = path.join(ROOT, 'projects/homepage/dist');
const cdnDir = path.join(distDir, 'cdn');

function fail(msg) {
    console.error(`homepage-cdn: ${msg}`);
    process.exit(1);
}

function readJson(rel) {
    const p = path.join(distDir, rel);
    if (!fs.existsSync(p)) fail(`missing ${p} — run vmz build first`);
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function copyRel(rel) {
    const from = path.join(distDir, rel);
    if (!fs.existsSync(from)) fail(`missing deploy file ${rel}`);
    const to = path.join(cdnDir, rel);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
}

if (!fs.existsSync(distDir)) {
    fail(`missing ${distDir}`);
}

if (fs.existsSync(cdnDir)) {
    fs.rmSync(cdnDir, { recursive: true, force: true });
}
fs.mkdirSync(cdnDir, { recursive: true });

const staticManifest = readJson('_vmz/static-delivery-manifest.json');
const assetsManifest = readJson('_vmz/content-addressed-assets.json');

/** @type {Set<string>} */
const relPaths = new Set();

for (const route of staticManifest.routes ?? []) {
    if (route.htmlPath) relPaths.add(String(route.htmlPath));
}

for (const doc of staticManifest.errorDocuments ?? []) {
    if (doc.path) relPaths.add(String(doc.path));
}

const seo = staticManifest.seoArtifacts ?? {};
if (seo.robots) relPaths.add(String(seo.robots));
if (seo.sitemap) relPaths.add(String(seo.sitemap));

for (const obj of assetsManifest.objects ?? []) {
    if (obj.assetPath) relPaths.add(String(obj.assetPath));
}

for (const rel of relPaths) {
    copyRel(rel);
}

/** Browser playground fetches `/dxo_lite_bg.wasm` from site root. */
const wasmName = 'dxo_lite_bg.wasm';
const wasmFromPublic = path.join(ROOT, 'projects/homepage/public', wasmName);
const wasmFromDist = path.join(distDir, wasmName);
const wasmSrc = fs.existsSync(wasmFromDist) ? wasmFromDist : fs.existsSync(wasmFromPublic) ? wasmFromPublic : null;
if (wasmSrc) {
    fs.copyFileSync(wasmSrc, path.join(cdnDir, wasmName));
    relPaths.add(wasmName);
} else {
    console.warn(`homepage-cdn: missing ${wasmName} (run stage-homepage-wasm / build:lite-wasm)`);
}

console.log(`homepage-cdn: staged ${relPaths.size} path(s) → ${cdnDir}`);
