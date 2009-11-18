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

export interface NativeAddon {
    backend(): string;
    version(): string;
    cudaAvailable(): boolean;
    probeTitanEventDep(): void;
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
}
