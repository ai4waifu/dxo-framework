/**
 * Stage VMZ static-cdn deploy root into projects/homepage/dist/cdn.
 * Cloudflare Pages output directory: projects/homepage/dist/cdn
 *
 * Workaround for `@vmz/vmz@0.1.12`: hashed `entry-client` / `entry-event` still
 * use `./vmz-dom.js` relative to `/assets/`, which 404s. Mirror upstream Fix A
 * (`rewriteJsEntryRelativeImports`) until a published VMZ release includes it.
 * Also copy logical client modules + `lib/` so dynamic entry imports resolve.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const distDir = path.join(ROOT, 'projects/homepage/dist');
const cdnDir = path.join(distDir, 'cdn');

const JS_ENTRY_LOGICAL = new Set(['entry-client.js', 'entry-event.js']);

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

function copyTree(relDir) {
    const fromRoot = path.join(distDir, relDir);
    if (!fs.existsSync(fromRoot)) return 0;
    let n = 0;
    const walk = (dir, relBase) => {
        for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
            const rel = path.posix.join(relBase, ent.name);
            const from = path.join(dir, ent.name);
            if (ent.isDirectory()) {
                walk(from, rel);
                continue;
            }
            const to = path.join(cdnDir, rel);
            fs.mkdirSync(path.dirname(to), { recursive: true });
            fs.copyFileSync(from, to);
            n += 1;
        }
    };
    walk(fromRoot, relDir.replace(/\\/g, '/'));
    return n;
}

/**
 * Same contract as VMZ `rewriteJsEntryRelativeImports` (unreleased on npm 0.1.12).
 * @param {string} jsText
 * @param {Record<string, string>} rewrites logical → `assets/<hash>.ext`
 */
function rewriteJsEntryRelativeImports(jsText, rewrites = {}) {
    let out = String(jsText || '');
    out = out.replace(/import\(\s*"\.\/"\s*\+/g, 'import("../"+');
    out = out.replace(/import\(\s*'\.\/'\s*\+/g, "import('../'+");

    const rewriteSpec = (spec) => {
        const logical = String(spec || '').replace(/^\.\//, '');
        if (!logical || logical.startsWith('../') || logical.startsWith('/')) return spec;
        const pathPart = logical.split('?')[0];
        const hashed = rewrites[pathPart] || rewrites[`/${pathPart}`] || rewrites[`assets/${pathPart}`];
        if (hashed) {
            const rel = String(hashed).replace(/^\//, '');
            return rel.startsWith('assets/') ? `./${path.basename(rel)}` : `./${rel}`;
        }
        return `../${logical}`;
    };

    out = out.replace(
        /\b((?:import|export)\s+[^'"\n]*?\s+from\s+|import\s*\(\s*)(['"])(\.\/[^'"]+)\2/g,
        (_m, prefix, quote, spec) => `${prefix}${quote}${rewriteSpec(spec)}${quote}`,
    );
    out = out.replace(
        /(^|[;\s])(import\s*)(['"])(\.\/[^'"]+)\3/gm,
        (_m, lead, kw, quote, spec) => `${lead}${kw}${quote}${rewriteSpec(spec)}${quote}`,
    );
    return out;
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
/** @type {Record<string, string>} */
const rewrites = {};

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
    if (obj.logicalPath) {
        const logical = String(obj.logicalPath).replace(/^\//, '');
        // Dynamic entry imports resolve against site root (`../` + id + `.client.js`).
        relPaths.add(logical);
        if (obj.assetPath) {
            rewrites[logical] = String(obj.assetPath).replace(/^\//, '');
            rewrites[`/${logical}`] = rewrites[logical];
        }
    }
}

for (const rel of relPaths) {
    copyRel(rel);
}

const libCopied = copyTree('lib');

/** Rewrite hashed entry shells in place under cdn/assets. */
let entryRewrites = 0;
for (const obj of assetsManifest.objects ?? []) {
    const logical = String(obj.logicalPath || '').replace(/^\//, '');
    if (!JS_ENTRY_LOGICAL.has(logical) || !obj.assetPath) continue;
    const assetRel = String(obj.assetPath).replace(/^\//, '');
    const abs = path.join(cdnDir, assetRel);
    if (!fs.existsSync(abs)) fail(`missing entry asset ${assetRel}`);
    const before = fs.readFileSync(abs, 'utf8');
    const after = rewriteJsEntryRelativeImports(before, rewrites);
    if (after !== before) {
        fs.writeFileSync(abs, after);
        entryRewrites += 1;
    }
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

console.log(
    `homepage-cdn: staged ${relPaths.size} path(s) + lib=${libCopied} entryRewrites=${entryRewrites} → ${cdnDir}`,
);
