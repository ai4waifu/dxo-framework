import { defineConfig } from '@vmz/vmz';

/** Static CDN profile — deploy root staged to dist/cdn for Cloudflare Pages. */
export default defineConfig({
    delivery: {
        default: 'static',
        profiles: {
            static: { host: 'browser', assembly: 'static-cdn' },
        },
    },
});
