/**
 * Vision product stub (Living `06` layer D — first domain package when A/B exit).
 * Workspace-only; excluded from placeholder / publish-npm lists.
 */

export type ImageTensorSpec = {
    layout: 'nchw' | 'nhwc';
    dtype: 'f32';
    channels: 1 | 3 | 4;
};

export function visionVersion(): string {
    return 'dxo-vision@placeholder';
}

export function unsupportedVisionApi(name: string): never {
    throw new Error(`@dxo/vision ${name} is a workspace stub; real transforms / zoo land after framework-core`);
}
