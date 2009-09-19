export interface NativeTensor {
    shape: number[];
    add(other: NativeTensor): NativeTensor;
    matmul(other: NativeTensor): NativeTensor;
    reshape(shape: number[]): NativeTensor;
    relu(): NativeTensor;
    toArray(): number[];
}

export interface NativeAddon {
    version(): string;
    tensor(data: number[], shape: number[]): NativeTensor;
    zeros(shape: number[]): NativeTensor;
}
