import { defineConfig } from '@vmz/vmz';

/** Local Studio workbench — loopback only via `dxo studio`. */
export default defineConfig({
    delivery: {
        default: 'node',
        profiles: {
            node: { host: 'node', assembly: 'node-ssr' },
        },
    },
});
