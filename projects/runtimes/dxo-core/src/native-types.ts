export type DType = 'f32' | 'f16' | 'bf16' | 'i64' | 'bool';

export interface NativeTensor {
    shape: number[];
    requiresGrad: boolean;
    grad: number[] | null;
    device: string;
    dtype: string;
    add(other: NativeTensor): NativeTensor;
    sub(other: NativeTensor): NativeTensor;
    neg(): NativeTensor;
    mul(other: NativeTensor): NativeTensor;
    div(other: NativeTensor): NativeTensor;
    matmul(other: NativeTensor): NativeTensor;
    reshape(shape: number[]): NativeTensor;
    transpose(): NativeTensor;
    relu(): NativeTensor;
    sum(): NativeTensor;
    mean(): NativeTensor;
    maxAll(): NativeTensor;
    softmax(): NativeTensor;
    logSoftmax(): NativeTensor;
    narrow(dim: number, start: number, len: number): NativeTensor;
    conv2d(weight: NativeTensor, bias: NativeTensor | null, stride: number, padding: number): NativeTensor;
    maxPool2d(kernel: number, stride: number, padding: number): NativeTensor;
    batchNorm2d(gamma: NativeTensor, beta: NativeTensor, eps?: number): NativeTensor;
    layerNorm(weight: NativeTensor, bias: NativeTensor, eps?: number): NativeTensor;
    bmm(other: NativeTensor): NativeTensor;
    transposeLast(): NativeTensor;
    transposeDims(dim0: number, dim1: number): NativeTensor;
    scaledDotProductAttention(k: NativeTensor, v: NativeTensor, causal?: boolean): NativeTensor;
    castDtype(dtype: string): NativeTensor;
    detach(): NativeTensor;
    to(device: string): NativeTensor;
    zeroGrad(): void;
    backward(): void;
    toArray(): number[];
}

/** Engine diagnosis from napi `doctorReport` (single source of truth for CLI doctor). */
export interface NativeDoctorReport {
    ok: boolean;
    version: string;
    backend: string;
    cudaAvailable: boolean;
    abi: string;
}

export interface NativeInspectRunMeta {
    format: string;
    version: number;
    runId: string;
    startedAtMs: number;
    endedAtMs?: number | null;
    label?: string | null;
    status: string;
    hyperparamsJson?: string | null;
}

export interface NativeInspectRunSummary {
    runId: string;
    meta: NativeInspectRunMeta;
}

export type NativeInspectApiServerOptions = {
    host?: string;
    port?: number;
    runsRoot?: string;
};

export type NativeInspectApiServerHandle = {
    host: string;
    port: number;
    url: string;
    runsRoot: string;
    close(): void;
};

export interface NativeAddon {
    backend(): string;
    version(): string;
    cudaAvailable(): boolean;
    doctorReport(): NativeDoctorReport;
    probeEventDep(): void;
    probeEventDepCuda(): void;
    hostTransferCount(): number;
    resetHostTransferCount(): void;
    cudaCapabilityFingerprint(): string;
    setGradEnabled(enabled: boolean): boolean;
    isGradEnabled(): boolean;
    tensor(data: number[], shape: number[], requiresGrad?: boolean): NativeTensor;
    tensorF32(data: Buffer, shape: number[], requiresGrad?: boolean): NativeTensor;
    zeros(shape: number[], requiresGrad?: boolean): NativeTensor;
    ones(shape: number[], requiresGrad?: boolean): NativeTensor;
    randn(shape: number[], requiresGrad?: boolean): NativeTensor;
    cat(tensors: NativeTensor[], dim: number): NativeTensor;
    stack(tensors: NativeTensor[], dim: number): NativeTensor;
    embedding(weight: NativeTensor, indices: NativeTensor): NativeTensor;
    /** Loopback inspect HTTP serve (Rust `dxo-studio`). */
    createInspectApiServer(options?: NativeInspectApiServerOptions): NativeInspectApiServerHandle;
    listInspectRuns(runsRoot?: string | null): NativeInspectRunSummary[];
    readInspectRunMeta(runId: string, runsRoot?: string | null): NativeInspectRunMeta | null;
    readInspectEventsJson(runId: string, runsRoot?: string | null): string;
    defaultInspectRunsRoot(): string;
}
