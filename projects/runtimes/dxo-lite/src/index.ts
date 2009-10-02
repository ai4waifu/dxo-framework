/**
 * @dxo/lite — browser/Worker runtime facade (0.0.8+ thin gate).
 *
 * TS owns async init, capabilities, explicit CPU fallback, and synchronous Tensor composition.
 * GPU compute will load a WASM facade → dxo-core → Titan `titan-backend-wgpu` when ready.
 * This layer never retains `GPUAdapter` / `GPUDevice` handles.
 * WebGL is never a tensor backend.
 */

export type FallbackMode = 'cpu' | 'error';
export type LiteBackend = 'webgpu' | 'cpu';
export type PowerPreference = 'low-power' | 'high-performance';

/** Set true when WASM + Titan wgpu facade is wired; until then compute stays host f32. */
export const TITAN_WGPU_READY = false;

export interface CreateRuntimeOptions {
    /** Passed to `navigator.gpu.requestAdapter` for capability probe only (no device acquire). */
    powerPreference?: PowerPreference;
    /** Adapter must expose these features during probe (ignored if empty). */
    requiredFeatures?: readonly string[];
    /**
     * When GPU compute is unavailable (no WebGPU, or Titan WASM not ready):
     * - `'error'` (default): throw a diagnostic Error
     * - `'cpu'`: host f32 tensors with async observation barriers
     */
    fallback?: FallbackMode;
}

export interface LiteCapabilities {
    /** Active compute backend. `'webgpu'` only when {@link TITAN_WGPU_READY} and WASM facade loaded. */
    backend: LiteBackend;
    /** True when a WebGPU adapter was probed (adapter not retained). */
    webgpu: boolean;
    /** True when Titan wgpu WASM facade is linked and may dispatch GPU kernels. */
    titanWgpuReady: boolean;
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
    tensor(data: ArrayLike<number>, shape: readonly number[]): Tensor;
    zeros(shape: readonly number[]): Tensor;
    ones(shape: readonly number[]): Tensor;
    /** Release runtime (no GPU handles in TS; safe to call more than once). */
    destroy(): void;
}

/** Minimal GPU* shapes for adapter probe — not stored on {@link LiteRuntime}. */
interface GpuAdapterInfo {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
}

interface GpuAdapter {
    features: { has(name: string): boolean; values(): IterableIterator<string> };
    limits: Record<string, number>;
    requestAdapterInfo?: () => Promise<GpuAdapterInfo>;
    info?: GpuAdapterInfo;
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

/** Dense float32 tensor. Ops synchronously return composable handles; observation is async. */
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

    matmul(other: Tensor): Tensor {
        const { data, shape } = matmulHost(this.#data, this.shape, other.#data, other.shape);
        return new Tensor(data, shape, this.device);
    }

    add(other: Tensor): Tensor {
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

    /** Wait until pending device work for this handle is complete. CPU resolves immediately. */
    async ready(): Promise<void> {}
}

class RuntimeImpl implements LiteRuntime {
    readonly capabilities: LiteCapabilities;
    #destroyed = false;

    constructor(capabilities: LiteCapabilities) {
        this.capabilities = capabilities;
    }

    tensor(data: ArrayLike<number>, shape: readonly number[]): Tensor {
        this.#assertLive();
        assertShape(data.length, shape);
        return new Tensor(Float32Array.from(data), shape, this.capabilities.backend);
    }

    zeros(shape: readonly number[]): Tensor {
        this.#assertLive();
        return new Tensor(new Float32Array(numel(shape)), shape, this.capabilities.backend);
    }

    ones(shape: readonly number[]): Tensor {
        this.#assertLive();
        const n = numel(shape);
        const data = new Float32Array(n);
        data.fill(1);
        return new Tensor(data, shape, this.capabilities.backend);
    }

    destroy(): void {
        this.#destroyed = true;
    }

    #assertLive(): void {
        if (this.#destroyed) throw new Error('LiteRuntime has been destroyed');
    }
}

function cpuCapabilities(): LiteCapabilities {
    return {
        backend: 'cpu',
        webgpu: false,
        titanWgpuReady: TITAN_WGPU_READY,
        webglTensorBackend: false,
        features: [],
        limits: {},
        dtype: { f32: true, f16: false },
    };
}

/**
 * Probe WebGPU adapter metadata without acquiring or retaining a device.
 * The adapter handle is dropped before returning.
 */
async function probeWebGpuCapabilities(options: CreateRuntimeOptions): Promise<LiteCapabilities | null> {
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
        backend: TITAN_WGPU_READY ? 'webgpu' : 'cpu',
        webgpu: true,
        titanWgpuReady: TITAN_WGPU_READY,
        webglTensorBackend: false,
        features,
        limits,
        dtype: { f32: true, f16: adapter.features.has('shader-f16') },
        adapterInfo,
    };
}

function titanNotReadyError(): Error {
    return new Error(
        "Titan wgpu WASM facade is not ready; pass { fallback: 'cpu' } for host tensors until WASM is linked. WebGL is not a DXO tensor backend.",
    );
}

function webGpuUnavailableError(): Error {
    return new Error(
        "WebGPU unavailable (no navigator.gpu). Pass { fallback: 'cpu' } for host tensors, or use a WebGPU-capable browser. WebGL is not a DXO tensor backend.",
    );
}

/**
 * Explicit async runtime bootstrap.
 * Never silently falls back to WebGL; never retains wgpu adapter/device in TS.
 */
export async function createRuntime(options: CreateRuntimeOptions = {}): Promise<LiteRuntime> {
    const fallback: FallbackMode = options.fallback ?? 'error';

    if (TITAN_WGPU_READY) {
        // Future: dynamic import WASM facade → dxo-core → Titan wgpu session.
        throw new Error('Titan wgpu WASM facade hook is not implemented yet');
    }

    try {
        const probed = await probeWebGpuCapabilities(options);
        if (probed) {
            if (fallback === 'error') {
                throw titanNotReadyError();
            }
            return new RuntimeImpl(probed);
        }
    } catch (err) {
        if (fallback === 'error') throw err;
    }

    if (fallback === 'cpu') {
        return new RuntimeImpl(cpuCapabilities());
    }

    if (!getGpu()) {
        throw webGpuUnavailableError();
    }
    throw new Error("WebGPU adapter request failed. Pass { fallback: 'cpu' } for host tensors. WebGL is not a DXO tensor backend.");
}

/** Package identity string (not npm semver). */
export function version(): string {
    return '0.1.0-dxo-lite';
}
