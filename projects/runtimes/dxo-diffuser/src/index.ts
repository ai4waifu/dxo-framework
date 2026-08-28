/**
 * Diffusion / multimodal pipeline product stub (Living `11`).
 * Workspace-only; excluded from placeholder / publish-npm until A/B + diffuser-pipeline.
 * Replaces the unfrozen `@dxo/generate` name.
 *
 * Generation pixels stay on OutputBuffer / device handle; preview uses explicit
 * low-res readback, thumbnail, or device-resident handle — never implicit full host copy.
 */

import type { OutputBuffer, TypedBuffer } from '@dxo/core';

export type DiffusionPipelineKind = 'text-to-image' | 'image-to-image';

export type ImageResult = {
    readonly width: number;
    readonly height: number;
    readonly output: OutputBuffer;
    readback(options?: { copy?: boolean; maxSide?: number }): Promise<TypedBuffer>;
};

export type DiffusionProgressEvent =
    | { type: 'progress'; step: number; total: number }
    | { type: 'preview'; step: number; thumbnail?: TypedBuffer; output?: OutputBuffer }
    | { type: 'result'; result: ImageResult }
    | { type: 'warning'; message: string };

export function diffuserVersion(): string {
    return 'dxo-diffuser@placeholder';
}

export function unsupportedDiffuserApi(name: string): never {
    throw new Error(
        `@dxo/diffuser ${name} is a workspace stub; DiffusionPipeline lands after vision-classify / llm-runtime`,
    );
}
