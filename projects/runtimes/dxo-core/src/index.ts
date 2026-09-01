import { Buffer } from 'node:buffer';
import { DxoError, wrapNative } from './errors.js';
import { formatDiagnostic } from './locale.js';
import { loadNative } from './native.js';
import type {
    DecodeSafetensorsResult,
    DecodedSafetensorEntry,
    NativeAdamState,
    NativeDoctorReport,
    NativeInspectApiServerHandle,
    NativeInspectApiServerOptions,
    NativeInspectRunMeta,
    NativeInspectRunSummary,
    NativeTensor,
    SafetensorBufferEntry,
} from './native-types.js';

export type Device = 'cpu' | 'cuda' | 'metal';

export type DType = 'f32' | 'f16' | 'bf16' | 'i64' | 'bool';

export type TensorData = number[] | Float32Array;

export interface TensorOptions {
    device?: Device;
    requiresGrad?: boolean;
}

export type {
    NativeAddon,
    NativeDoctorReport,
    NativeInspectApiServerHandle,
    NativeInspectApiServerOptions,
    NativeInspectRunMeta,
    NativeInspectRunSummary,
    NativeTensor,
    SafetensorBufferEntry,
    DecodedSafetensorEntry,
    DecodeSafetensorsResult,
} from './native-types.js';

export {
    DxoError,
    wrapNative,
    rethrowAsDxoError,
    type DxoDiagnostic,
    type DxoErrorInit,
    type DxoSeverity,
} from './errors.js';
export { formatDiagnostic, resolveLocale, type DxoLocale } from './locale.js';

/**
 * Dense float32 tensor (CPU preview + optional CUDA matmul spike).
 *
 * Contract (G3 / 0.0.4+, G4 / 0.0.6 CUDA):
 * - Ops are eager; Tape records when `requiresGrad` and grad mode is enabled.
 * - `backward()` requires a scalar (`numel === 1`, usually shape `[1]`).
 * - `grad` is a row-major copy or `undefined`.
 * - Factory helpers create CPU tensors; use `.to('cuda')` on detached tensors when CUDA is available.
 */
export class Tensor {
    readonly #handle: NativeTensor;

    constructor(handle: NativeTensor) {
        this.#handle = handle;
    }

    get shape(): readonly number[] {
        return this.#handle.shape;
    }

    get device(): Device {
        const d = this.#handle.device;
        if (d === 'cpu' || d === 'cuda') return d;
        throw new DxoError({
            code: 'DXO_BACKEND_UNAVAILABLE',
            message: `unexpected native device tag: ${d}`,
            args: { requested: d, available: 'cpu,cuda' },
            operation: 'device',
        });
    }

    get requiresGrad(): boolean {
        return this.#handle.requiresGrad;
    }

    get dtype(): DType {
        const d = this.#handle.dtype;
        if (d === 'f32' || d === 'f16' || d === 'bf16' || d === 'i64' || d === 'bool') return d;
        return 'f32';
    }

    /** Native handle for inter-op calls within `@dxo/core`. */
    get nativeHandle(): NativeTensor {
        return this.#handle;
    }

    /** Accumulated gradient (row-major), or `undefined` if absent. */
    get grad(): number[] | undefined {
        return this.#handle.grad ?? undefined;
    }

    /** Number of elements (shape product). */
    numel(): number {
        return this.shape.reduce((n, d) => n * d, 1);
    }

    add(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.add(other.#handle)));
    }

    /** `this - other` (broadcast-aware). */
    sub(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.sub(other.#handle)));
    }

    neg(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.neg()));
    }

    div(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.div(other.#handle)));
    }

    mul(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.mul(other.#handle)));
    }

    matmul(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.matmul(other.#handle)));
    }

    reshape(shape: readonly number[]): Tensor {
        return new Tensor(wrapNative(() => this.#handle.reshape([...shape])));
    }

    transpose(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.transpose()));
    }

    relu(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.relu()));
    }

    /** Sum all elements → scalar tensor of shape `[1]`. */
    sum(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.sum()));
    }

    /** Mean of all elements → scalar tensor of shape `[1]`. */
    mean(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.mean()));
    }

    max(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.maxAll()));
    }

    softmax(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.softmax()));
    }

    logSoftmax(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.logSoftmax()));
    }

    narrow(dim: number, start: number, len: number): Tensor {
        return new Tensor(wrapNative(() => this.#handle.narrow(dim, start, len)));
    }

    conv2d(weight: Tensor, bias: Tensor | null, stride: number, padding: number): Tensor {
        return new Tensor(
            wrapNative(() => this.#handle.conv2d(weight.nativeHandle, bias ? bias.nativeHandle : null, stride, padding)),
        );
    }

    maxPool2d(kernel: number, stride: number, padding: number): Tensor {
        return new Tensor(wrapNative(() => this.#handle.maxPool2d(kernel, stride, padding)));
    }

    batchNorm2d(gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
        return new Tensor(wrapNative(() => this.#handle.batchNorm2d(gamma.nativeHandle, beta.nativeHandle, eps)));
    }

    /** LayerNorm over the last dimension. */
    layerNorm(weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
        return new Tensor(wrapNative(() => this.#handle.layerNorm(weight.nativeHandle, bias.nativeHandle, eps)));
    }

    /** Batch matmul `[B,M,K] @ [B,K,N]`. */
    bmm(other: Tensor): Tensor {
        return new Tensor(wrapNative(() => this.#handle.bmm(other.nativeHandle)));
    }

    /** Swap the last two axes. */
    transposeLast(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.transposeLast()));
    }

    /** Swap two axes. */
    transposeDims(dim0: number, dim1: number): Tensor {
        return new Tensor(wrapNative(() => this.#handle.transposeDims(dim0, dim1)));
    }

    /** Scaled dot-product attention; `this`/`k`/`v` are `[B,H,T,D]`. */
    scaledDotProductAttention(k: Tensor, v: Tensor, causal = false): Tensor {
        return new Tensor(wrapNative(() => this.#handle.scaledDotProductAttention(k.nativeHandle, v.nativeHandle, causal)));
    }

    /** Retag logical dtype without changing host f32 payload. */
    castDtype(dtype: DType): Tensor {
        return new Tensor(wrapNative(() => this.#handle.castDtype(dtype)));
    }

    /** Values only — drops requiresGrad and tape edges. */
    detach(): Tensor {
        return new Tensor(wrapNative(() => this.#handle.detach()));
    }

    /** Move to `cpu` or `cuda` (CUDA requires detached tensor in this preview). */
    to(device: 'cpu' | 'cuda'): Tensor {
        return new Tensor(wrapNative(() => this.#handle.to(device)));
    }

    /** Clear accumulated gradient on this leaf/intermediate slot. */
    zeroGrad(): void {
        wrapNative(() => this.#handle.zeroGrad());
    }

    /** Reverse-mode from a scalar output. */
    backward(): void {
        wrapNative(() => this.#handle.backward());
    }

    /** CPU preview: resolves immediately once host values are available. */
    async ready(): Promise<void> {}

    toFloat32Array(): Promise<Float32Array> {
        return Promise.resolve(Float32Array.from(wrapNative(() => this.#handle.toArray())));
    }

    async toArray(): Promise<number[]> {
        return wrapNative(() => this.#handle.toArray());
    }

    /** Little-endian f32 bytes for Rust safetensors encode (no JS `number[]`). */
    toF32Buffer(): Buffer {
        return wrapNative(() => this.#handle.toF32Buffer());
    }

    async item(): Promise<number> {
        if (this.numel() !== 1) {
            throw new DxoError({
                code: 'DXO_TENSOR_NON_SCALAR',
                message: `item() requires a scalar tensor, got shape [${this.shape.join(',')}]`,
                args: { shape: `[${this.shape.join(',')}]`, operation: 'item' },
                details: { shape: [...this.shape] },
                operation: 'item',
            });
        }
        const values = await this.toArray();
        return values[0]!;
    }

    /** Localize this error-like diagnostic from a thrown `DxoError` helper path. */
    static formatError(err: DxoError, locale?: string | null): string {
        return formatDiagnostic(err.toDiagnostic(), locale);
    }
}

function assertCpu(options: Pick<TensorOptions, 'device'>): void {
    if (options.device && options.device !== 'cpu') {
        throw new DxoError({
            code: 'DXO_DEVICE_UNAVAILABLE',
            message: `device '${options.device}' is not available in this preview (cpu only)`,
            args: { device: options.device },
            details: { requestedDevice: options.device },
            operation: 'tensor',
        });
    }
}

/** Create a CPU tensor from flat data + shape. */
export function tensor(data: TensorData, shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    const rg = options.requiresGrad ?? false;
    if (data instanceof Float32Array) {
        const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
        return new Tensor(wrapNative(() => native.tensorF32(buf, [...shape], rg)));
    }
    const flat = flattenData(data);
    return new Tensor(wrapNative(() => native.tensor(flat, [...shape], rg)));
}

export function zeros(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(wrapNative(() => native.zeros([...shape], options.requiresGrad ?? false)));
}

export function ones(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(wrapNative(() => native.ones([...shape], options.requiresGrad ?? false)));
}

export function randn(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(wrapNative(() => native.randn([...shape], options.requiresGrad ?? false)));
}

/** Gather embedding rows: `weight` `[vocab, dim]`, `indices` integer ids. */
export function embedding(weight: Tensor, indices: Tensor): Tensor {
    return new Tensor(wrapNative(() => loadNative().embedding(weight.nativeHandle, indices.nativeHandle)));
}

/** Clear gradients on every leaf in one napi call (train-batch path). */
export function zeroGrads(params: readonly Tensor[]): void {
    wrapNative(() => loadNative().zeroGrads(params.map((p) => p.nativeHandle)));
}

/** Batch SGD in Rust; returns new `requiresGrad` leaves. */
export function sgdStep(params: readonly Tensor[], lr: number): Tensor[] {
    return wrapNative(() =>
        loadNative()
            .sgdStep(
                params.map((p) => p.nativeHandle),
                lr,
            )
            .map((h) => new Tensor(h)),
    );
}

/** `loss.backward()` then batch SGD in one engine call. */
export function backwardSgdStep(loss: Tensor, params: readonly Tensor[], lr: number): Tensor[] {
    return wrapNative(() =>
        loadNative()
            .backwardSgdStep(
                loss.nativeHandle,
                params.map((p) => p.nativeHandle),
                lr,
            )
            .map((h) => new Tensor(h)),
    );
}

/** Native Adam moment state (opaque across steps). */
export type AdamStateHandle = NativeAdamState;

export function createAdamState(): AdamStateHandle {
    return wrapNative(() => new (loadNative().AdamState)());
}

/** Batch Adam in Rust; mutates `state` and returns new `requiresGrad` leaves. */
export function adamStep(
    params: readonly Tensor[],
    state: AdamStateHandle,
    lr: number,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
): Tensor[] {
    return wrapNative(() =>
        loadNative()
            .adamStep(
                params.map((p) => p.nativeHandle),
                state,
                lr,
                beta1,
                beta2,
                eps,
            )
            .map((h) => new Tensor(h)),
    );
}

/** Create a CPU tensor from little-endian f32 bytes + shape. */
export function tensorFromF32Buffer(data: Buffer, shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const rg = options.requiresGrad ?? false;
    return new Tensor(wrapNative(() => loadNative().tensorF32(data, [...shape], rg)));
}

/** Encode named f32 tensor buffers to safetensors bytes (Rust codec). */
export function encodeSafetensors(entries: SafetensorBufferEntry[], metadataJson?: string): Uint8Array {
    return wrapNative(() => {
        const buf = loadNative().encodeSafetensors(entries, metadataJson ?? undefined);
        return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    });
}

/** Decode safetensors bytes via Rust codec. */
export function decodeSafetensors(bytes: Uint8Array): DecodeSafetensorsResult {
    return wrapNative(() => loadNative().decodeSafetensors(Buffer.from(bytes)));
}

/** Host draw for sync init paths (e.g. Linear weight init). */
export function randnValues(shape: readonly number[]): number[] {
    return loadNative()
        .randn([...shape], false)
        .toArray();
}

export function version(): string {
    return loadNative().version();
}

/** Active engine backend for diagnostics (`cpu` / `cuda` / … — DXO product labels only). */
export function backend(): string {
    return loadNative().backend();
}

/** Whether CUDA Driver API is available on this machine. */
export function cudaAvailable(): boolean {
    return loadNative().cudaAvailable();
}

/** Engine/backend diagnosis from napi (CLI must not reimplement this). */
export function doctorReport(): NativeDoctorReport {
    return loadNative().doctorReport();
}

/** Probe HAL `wait_event` via the CPU session (does not use JS await as a substitute). */
export function probeEventDep(): void {
    loadNative().probeEventDep();
}

/** Probe HAL `wait_event` on CUDA (throws when CUDA unavailable). */
export function probeEventDepCuda(): void {
    loadNative().probeEventDepCuda();
}

/** Host↔device transfer count since process start / last reset. */
export function hostTransferCount(): number {
    return loadNative().hostTransferCount();
}

/** Reset residency transfer counter. */
export function resetHostTransferCount(): void {
    loadNative().resetHostTransferCount();
}

/** CUDA capability fingerprint for GPU CI manifests. */
export function cudaCapabilityFingerprint(): string {
    return loadNative().cudaCapabilityFingerprint();
}

/** Default inspect runs root (same rule as napi inspect HTTP serve). */
export function defaultInspectRunsRoot(): string {
    return loadNative().defaultInspectRunsRoot();
}

/** List inspect runs via Rust store. */
export function listInspectRuns(runsRoot?: string): NativeInspectRunSummary[] {
    return loadNative().listInspectRuns(runsRoot ?? null);
}

/** Read one run meta via Rust store. */
export function readInspectRunMeta(runId: string, runsRoot?: string): NativeInspectRunMeta | null {
    return loadNative().readInspectRunMeta(runId, runsRoot ?? null);
}

/** Read inspect events.jsonl as parsed JSON array via Rust store. */
export function readInspectEvents(runId: string, runsRoot?: string): unknown[] {
    const raw = loadNative().readInspectEventsJson(runId, runsRoot ?? null);
    return JSON.parse(raw) as unknown[];
}

/** Start loopback inspect HTTP serve (napi). Prefer `@dxo/studio` for WebUI orchestration. */
export function createInspectApiServer(options: NativeInspectApiServerOptions = {}): NativeInspectApiServerHandle {
    return loadNative().createInspectApiServer(options);
}

/** Whether the current thread records autograd ops. */
export function isGradEnabled(): boolean {
    return loadNative().isGradEnabled();
}

/**
 * Run `fn` with tape recording disabled, then restore the previous flag.
 * Uses a thread-local scope stack (not a permanent process global).
 */
export function withoutGrad<T>(run: () => T): T {
    const native = loadNative();
    const prev = native.setGradEnabled(false);
    try {
        return run();
    } finally {
        native.setGradEnabled(prev);
    }
}

export type {
    AudioBuffer,
    BatchBuffer,
    IndexBuffer,
    OutputBuffer,
    SparseBuffer,
    StateBuffer,
    TokenBuffer,
    TokenBufferDtype,
    VideoFrameBuffer,
} from './domain-buffers.js';
export type {
    CreateImageBufferFromPixelsOptions,
    DecodeImageBridgeOptions,
    ImageBuffer,
    ImageBufferAlphaMode,
    ImageBufferColorSpace,
    ImageBufferDtype,
    ImageBufferLayout,
    ImageBufferToTensorOptions,
} from './image-buffer.js';
export { decodeImageBuffer, createImageBufferFromPixels, unsupportedImageBufferApi } from './image-buffer.js';
export type { PrecisionCapabilities, PrecisionPolicy } from './precision.js';
export { precisionCapabilities, resolvePrecisionPolicy, resolveWeightDtype } from './precision.js';

export type {
    BufferToTensorOptions,
    CodecHandle,
    DeviceBuffer,
    HostDtype,
    MappedFile,
    StreamHandle,
    TensorView,
    TypedBuffer,
} from './typed-buffer.js';
export { unsupportedTypedBufferApi } from './typed-buffer.js';

function flattenData(data: number[]): number[] {
    return data.flat(Infinity) as number[];
}
