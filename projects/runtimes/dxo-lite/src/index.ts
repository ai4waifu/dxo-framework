/**
 * @dxo/lite — browser/Worker WebGPU runtime facade (0.0.8 thin gate).
 *
 * Target stack: TS facade → (future) wasm-bindgen → lite-engine (Rust + wgpu) → WebGPU.
 * This slice freezes async init, capabilities, explicit CPU fallback, and a host f32 op subset.
 * WebGL is never a tensor backend.
 */

export type FallbackMode = 'cpu' | 'error';
export type LiteBackend = 'webgpu' | 'cpu';
export type PowerPreference = 'low-power' | 'high-performance';

export interface CreateRuntimeOptions {
    /** Passed to `navigator.gpu.requestAdapter` when WebGPU is available. */
    powerPreference?: PowerPreference;
    /** Reserved for future adapter feature negotiation (ignored if empty). */
    requiredFeatures?: readonly string[];
    /**
     * When WebGPU is unavailable:
     * - `'error'` (default): throw a diagnostic Error
     * - `'cpu'`: use host f32 Promise tensors
     */
    fallback?: FallbackMode;
}

export interface LiteCapabilities {
    backend: LiteBackend;
    /** True only when a WebGPU adapter+device were acquired. */
    webgpu: boolean;
    /** Always false — WebGL is not a DXO tensor backend. */
    webglTensorBackend: false;
    features: readonly string[];
    limits: Readonly<Record<string, number>>;
    dtype: { readonly f32: true; readonly f16: boolean };
    adapterInfo?: { vendor?: string; architecture?: string; device?: string; description?: string };
}

export interface LiteRuntime {
    readonly capabilities: LiteCapabilities;
    /** Create a dense f32 tensor (row-major). */
    tensor(data: ArrayLike<number>, shape: readonly number[]): Promise<Tensor>;
    zeros(shape: readonly number[]): Promise<Tensor>;
    ones(shape: readonly number[]): Promise<Tensor>;
    /** Release GPU device (no-op for CPU). Safe to call more than once. */
    destroy(): void;
}

/** Minimal GPU* shapes so Node builds do not need `@webgpu/types`. */
interface GpuAdapterInfo {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
}

interface GpuAdapter {
    features: { has(name: string): boolean; values(): IterableIterator<string> };
    limits: Record<string, number>;
    requestDevice(): Promise<GpuDevice>;
    requestAdapterInfo?: () => Promise<GpuAdapterInfo>;
    info?: GpuAdapterInfo;
}

interface GpuDevice {
    destroy(): void;
    createBuffer(desc: { size: number; usage: number; mappedAtCreation?: boolean }): { destroy(): void };
}

interface Gpu {
    requestAdapter(opts?: { powerPreference?: PowerPreference }): Promise<GpuAdapter | null>;
}

function getGpu(): Gpu | undefined {
    const nav = globalThis as typeof globalThis & { navigator?: { gpu?: Gpu } };
    return nav.navigator?.gpu;
}

function numel(shape: readonly number[]): number {
    return shape.reduce((n, d) => n * d, 1);
}

function assertShape(dataLen: number, shape: readonly number[]): void {
    const n = numel(shape);
    if (dataLen !== n) {
        throw new Error(`data length ${dataLen} does not match shape product ${n} (shape=[${shape.join(',')}])`);
    }
}

function matmulHost(
    a: Float32Array,
    aShape: readonly number[],
    b: Float32Array,
    bShape: readonly number[],
): {
    data: Float32Array;
    shape: number[];
} {
    if (aShape.length !== 2 || bShape.length !== 2) {
        throw new Error('lite matmul requires rank-2 tensors');
    }
    const [m, k] = aShape as [number, number];
    const [k2, n] = bShape as [number, number];
    if (k !== k2) throw new Error(`matmul inner dim mismatch: ${k} vs ${k2}`);
    const out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let t = 0; t < k; t++) sum += a[i * k + t]! * b[t * n + j]!;
            out[i * n + j] = sum;
        }
    }
    return { data: out, shape: [m, n] };
}

function addHost(a: Float32Array, b: Float32Array): Float32Array {
    if (a.length !== b.length) throw new Error(`add length mismatch: ${a.length} vs ${b.length}`);
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
    return out;
}

/** Dense float32 tensor. Ops return Promises (never pretend sync GPU). */
export class Tensor {
    readonly #data: Float32Array;
    readonly shape: readonly number[];
    readonly device: LiteBackend;

    /** @internal */
    constructor(data: Float32Array, shape: readonly number[], device: LiteBackend) {
        this.#data = data;
        this.shape = Object.freeze([...shape]);
        this.device = device;
    }

    numel(): number {
        return this.#data.length;
    }

    async matmul(other: Tensor): Promise<Tensor> {
        const { data, shape } = matmulHost(this.#data, this.shape, other.#data, other.shape);
        return new Tensor(data, shape, this.device);
    }

    async add(other: Tensor): Promise<Tensor> {
        if (this.shape.length !== other.shape.length || this.shape.some((d, i) => d !== other.shape[i])) {
            throw new Error(`add requires identical shapes: [${this.shape}] vs [${other.shape}]`);
        }
        return new Tensor(addHost(this.#data, other.#data), this.shape, this.device);
    }

    /** Explicit host readback (copy). */
    async toCpu(): Promise<number[]> {
        return Array.from(this.#data);
    }

    /** Alias for {@link toCpu}. */
    async toArray(): Promise<number[]> {
        return this.toCpu();
    }
}

class RuntimeImpl implements LiteRuntime {
    readonly capabilities: LiteCapabilities;
    #device: GpuDevice | null;
    #destroyed = false;

    constructor(capabilities: LiteCapabilities, device: GpuDevice | null) {
        this.capabilities = capabilities;
        this.#device = device;
    }

    async tensor(data: ArrayLike<number>, shape: readonly number[]): Promise<Tensor> {
        this.#assertLive();
        assertShape(data.length, shape);
        return new Tensor(Float32Array.from(data), shape, this.capabilities.backend);
    }

    async zeros(shape: readonly number[]): Promise<Tensor> {
        this.#assertLive();
        return new Tensor(new Float32Array(numel(shape)), shape, this.capabilities.backend);
    }

    async ones(shape: readonly number[]): Promise<Tensor> {
        this.#assertLive();
        const n = numel(shape);
        const data = new Float32Array(n);
        data.fill(1);
        return new Tensor(data, shape, this.capabilities.backend);
    }

    destroy(): void {
        if (this.#destroyed) return;
        this.#destroyed = true;
        try {
            this.#device?.destroy();
        } catch {
            /* ignore */
        }
        this.#device = null;
    }

    #assertLive(): void {
        if (this.#destroyed) throw new Error('LiteRuntime has been destroyed');
    }
}

function cpuCapabilities(): LiteCapabilities {
    return {
        backend: 'cpu',
        webgpu: false,
        webglTensorBackend: false,
        features: [],
        limits: {},
        dtype: { f32: true, f16: false },
    };
}

async function tryWebGpu(options: CreateRuntimeOptions): Promise<{
    capabilities: LiteCapabilities;
    device: GpuDevice;
} | null> {
    const gpu = getGpu();
    if (!gpu) return null;

    const adapter = await gpu.requestAdapter({
        powerPreference: options.powerPreference ?? 'high-performance',
    });
    if (!adapter) return null;

    const required = options.requiredFeatures ?? [];
    for (const feat of required) {
        if (!adapter.features.has(feat)) {
            throw new Error(`WebGPU adapter missing required feature '${feat}'`);
        }
    }

    const device = await adapter.requestDevice();
    const features = [...adapter.features.values()];
    let adapterInfo: LiteCapabilities['adapterInfo'];
    try {
        const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
        if (info) {
            adapterInfo = {
                vendor: info.vendor,
                architecture: info.architecture,
                device: info.device,
                description: info.description,
            };
        }
    } catch {
        /* optional */
    }

    const limits: Record<string, number> = {};
    for (const [k, v] of Object.entries(adapter.limits ?? {})) {
        if (typeof v === 'number') limits[k] = v;
    }

    return {
        device,
        capabilities: {
            backend: 'webgpu',
            webgpu: true,
            webglTensorBackend: false,
            features,
            limits,
            dtype: { f32: true, f16: adapter.features.has('shader-f16') },
            adapterInfo,
        },
    };
}

/**
 * Explicit async runtime bootstrap.
 * Never silently falls back to WebGL.
 */
export async function createRuntime(options: CreateRuntimeOptions = {}): Promise<LiteRuntime> {
    const fallback: FallbackMode = options.fallback ?? 'error';

    try {
        const web = await tryWebGpu(options);
        if (web) return new RuntimeImpl(web.capabilities, web.device);
    } catch (err) {
        if (fallback === 'error') throw err;
        // fall through to CPU when fallback === 'cpu'
    }

    if (fallback === 'cpu') {
        return new RuntimeImpl(cpuCapabilities(), null);
    }

    const gpu = getGpu();
    if (!gpu) {
        throw new Error(
            "WebGPU unavailable (no navigator.gpu). Pass { fallback: 'cpu' } for host tensors, or use a WebGPU-capable browser. WebGL is not a DXO tensor backend.",
        );
    }
    throw new Error("WebGPU adapter/device request failed. Pass { fallback: 'cpu' } for host tensors. WebGL is not a DXO tensor backend.");
}

/** Package identity string (not npm semver). */
export function version(): string {
    return '0.1.0-dxo-lite';
}
