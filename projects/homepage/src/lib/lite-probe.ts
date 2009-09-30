import { createRuntime, version } from '@dxo/lite';

export type LiteProbe = {
    version: string;
    backend: string;
    webgpu: boolean;
    webglTensorBackend: false;
    tensorClass: string;
    matmulOk: boolean;
};

/** Runtime probe for @dxo/lite (async createRuntime + CPU fallback). */
export async function probeLite(): Promise<LiteProbe> {
    const rt = await createRuntime({ fallback: 'cpu' });
    try {
        const a = await rt.tensor([1, 2, 3, 4], [2, 2]);
        const b = await rt.tensor([1, 0, 0, 1], [2, 2]);
        const c = await a.matmul(b);
        const out = await c.toArray();
        return {
            version: version(),
            backend: rt.capabilities.backend,
            webgpu: rt.capabilities.webgpu,
            webglTensorBackend: rt.capabilities.webglTensorBackend,
            tensorClass: a.constructor.name,
            matmulOk: out[0] === 1 && out[1] === 2 && out[2] === 3 && out[3] === 4,
        };
    } finally {
        rt.destroy();
    }
}
