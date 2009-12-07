/**
 * Homepage entry for local vs Cloudflare Pages.
 * CF has no wasm-pack / Rust — use pinned registry `@dxo/lite` only (`homepage:cf`).
 */
import { spawnSync } from 'node:child_process';

function run(cmd, args) {
    const r = spawnSync(cmd, args, { stdio: 'inherit', shell: false });
    if (r.error) {
        console.error(r.error);
        process.exit(1);
    }
    if ((r.status ?? 1) !== 0) process.exit(r.status ?? 1);
}

if (process.env.CF_PAGES === '1') {
    console.log('homepage: CF_PAGES=1 → homepage:cf (registry @dxo/lite, no wasm-pack)');
    run('pnpm', ['run', 'homepage:cf']);
} else {
    run('pnpm', ['install']);
    run('pnpm', ['run', 'build:lite-wasm']);
    run('pnpm', ['run', 'build:ts']);
    run(process.execPath, ['scripts/build/stage-homepage-wasm.mjs']);
    run('pnpm', ['--filter', '@dxo/homepage', 'run', 'build']);
}
