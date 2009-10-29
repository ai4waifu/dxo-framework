import { defineConfig } from '@vmz/vmz';

/** Studio WebUI — VMZ dev watch (`dxo studio`) + static build for Tauri embed. */
export default defineConfig({
    delivery: {
        default: 'node',
        profiles: {
            node: { host: 'node', assembly: 'node-ssr' },
            static: { host: 'browser', assembly: 'static-cdn' },
        },
    },
});
