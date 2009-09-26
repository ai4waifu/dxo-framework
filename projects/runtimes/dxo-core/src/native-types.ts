export interface NativeTensor {
    shape: number[];
    requiresGrad: boolean;
    grad: number[] | null;
    add(other: NativeTensor): NativeTensor;
    mul(other: NativeTensor): NativeTensor;
    matmul(other: NativeTensor): NativeTensor;
    reshape(shape: number[]): NativeTensor;
    transpose(): NativeTensor;
    relu(): NativeTensor;
    sum(): NativeTensor;
    detach(): NativeTensor;
    zeroGrad(): void;
    backward(): void;
    toArray(): number[];
}

export interface NativeAddon {
    backend(): string;
    version(): string;
    setGradEnabled(enabled: boolean): boolean;
    isGradEnabled(): boolean;
    tensor(data: number[], shape: number[], requiresGrad?: boolean): NativeTensor;
    tensorF32(data: Buffer, shape: number[], requiresGrad?: boolean): NativeTensor;
    zeros(shape: number[], requiresGrad?: boolean): NativeTensor;
    ones(shape: number[], requiresGrad?: boolean): NativeTensor;
    randn(shape: number[], requiresGrad?: boolean): NativeTensor;
}
