import { defineConfig } from '@vmz/vmz';

/** Studio WebUI — VMZ dev watch (`dxo studio`) + static build for Tauri embed. */
export default defineConfig({
    delivery: {
        default: 'node',
        profiles: {
            node: { host: 'browser', assembly: 'server-host', serverRuntime: 'node' },
            static: { host: 'browser', assembly: 'web-static' },
        },
    },
});
