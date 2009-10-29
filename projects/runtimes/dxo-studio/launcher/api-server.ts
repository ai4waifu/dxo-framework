import { loadNative } from '@dxo/core/native';

export type InspectApiServerOptions = {
    host?: string;
    port?: number;
    runsRoot?: string;
};

export type InspectApiServer = {
    host: string;
    port: number;
    url: string;
    runsRoot: string;
    close: () => Promise<void>;
};

type NativeInspectHandle = {
    host: string;
    port: number;
    url: string;
    runsRoot: string;
    close: () => void;
};

type NativeWithInspect = ReturnType<typeof loadNative> & {
    createInspectApiServer(options: InspectApiServerOptions): NativeInspectHandle;
};

/** Loopback HTTP API over the append-only inspect run store (Rust `dxo-studio`). */
export async function createInspectApiServer(options: InspectApiServerOptions = {}): Promise<InspectApiServer> {
    const native = loadNative() as NativeWithInspect;
    if (typeof native.createInspectApiServer !== 'function') {
        throw new Error('createInspectApiServer requires `pnpm build:native`');
    }
    const handle = native.createInspectApiServer({
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
