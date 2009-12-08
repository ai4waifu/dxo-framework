import { createInspectApiServer as createNativeInspectApiServer, type NativeInspectApiServerOptions } from '@dxo/core';

export type InspectApiServerOptions = NativeInspectApiServerOptions;

export type InspectApiServer = {
    host: string;
    port: number;
    url: string;
    runsRoot: string;
    close: () => Promise<void>;
};

/**
 * Loopback inspect HTTP serve — thin TS wrapper over napi `create_inspect_api_server`.
 * Do not reimplement the HTTP server or run-store reader in TypeScript.
 */
export async function createInspectApiServer(options: InspectApiServerOptions = {}): Promise<InspectApiServer> {
    const handle = createNativeInspectApiServer({
        host: options.host,
        port: options.port,
        runsRoot: options.runsRoot,
    });
    return {
        host: handle.host,
        port: handle.port,
        url: handle.url,
        runsRoot: handle.runsRoot,
        close: async () => {
            handle.close();
        },
    };
}
