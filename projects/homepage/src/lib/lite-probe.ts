import { createRuntime, version, type LiteRuntime } from '@dxo/lite';

export { version };

export type LiteProbe = {
    version: string;
    backend: string;
    webgpu: boolean;
    titanWgpuReady: boolean;
    webglTensorBackend: false;
    tensorClass: string;
    matmulOk: boolean;
    wasmVersion: string;
    wasmInterim: boolean;
};

function browserWasmUrl(): URL | undefined {
    if (typeof location === 'undefined') return undefined;
    return new URL('/dxo_lite_bg.wasm', location.origin);
}

/** Options shared by homepage probe / playground (browser serves public wasm). */
export function homepageLiteOptions() {
    const url = browserWasmUrl();
    return {
        fallback: 'cpu' as const,
        wasm: url ? { url } : undefined,
    };
}

/** Runtime probe for @dxo/lite (async createRuntime + CPU fallback + interim WASM). */
export async function probeLite(): Promise<LiteProbe> {
    const rt = await createRuntime(homepageLiteOptions());
    try {
        const a = rt.tensor([1, 2, 3, 4], [2, 2]);
        const b = rt.tensor([1, 0, 0, 1], [2, 2]);
        const c = a.matmul(b);
        const out = await c.toArray();
        return {
            version: version(),
            backend: rt.capabilities.backend,
            webgpu: rt.capabilities.webgpu,
            titanWgpuReady: rt.capabilities.titanWgpuReady,
            webglTensorBackend: rt.capabilities.webglTensorBackend,
            tensorClass: a.constructor.name,
            matmulOk: out[0] === 1 && out[1] === 2 && out[2] === 3 && out[3] === 4,
            wasmVersion: rt.capabilities.wasm?.version ?? '(not loaded)',
            wasmInterim: rt.capabilities.wasm?.interimHostF32 ?? false,
        };
    } finally {
        rt.destroy();
    }
}

export async function openHomepageRuntime(): Promise<LiteRuntime> {
    return createRuntime(homepageLiteOptions());
}
