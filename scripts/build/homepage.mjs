/**
 * Homepage entry for local vs Cloudflare Pages.
 *
 * CF (`CF_PAGES=1`) has no Rust/wasm-pack — install + build with pinned registry
 * `@dxo/lite` only; `stage-homepage-wasm` copies published wasm into `public/`.
 * Local: full lite-wasm + TS build, then homepage.
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
    console.log('homepage: CF_PAGES=1 → registry `@dxo/lite` only (no wasm-pack)');
    run('pnpm', ['--filter', '@dxo/homepage...', 'install']);
    run('pnpm', ['--filter', '@dxo/homepage', 'run', 'build']);
} else {
    run('pnpm', ['install']);
    run('pnpm', ['run', 'build:lite-wasm']);
    run('pnpm', ['run', 'build:ts']);
    run(process.execPath, ['scripts/build/stage-homepage-wasm.mjs']);
    run('pnpm', ['--filter', '@dxo/homepage', 'run', 'build']);
}
