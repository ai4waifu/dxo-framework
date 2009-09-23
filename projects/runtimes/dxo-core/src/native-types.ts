export interface NativeTensor {
    shape: number[];
    add(other: NativeTensor): NativeTensor;
    mul(other: NativeTensor): NativeTensor;
    matmul(other: NativeTensor): NativeTensor;
    reshape(shape: number[]): NativeTensor;
    transpose(): NativeTensor;
    relu(): NativeTensor;
    toArray(): number[];
}

export interface NativeAddon {
    backend(): string;
    version(): string;
    tensor(data: number[], shape: number[]): NativeTensor;
    tensorF32(data: Buffer, shape: number[]): NativeTensor;
    zeros(shape: number[]): NativeTensor;
    ones(shape: number[]): NativeTensor;
    randn(shape: number[]): NativeTensor;
}
