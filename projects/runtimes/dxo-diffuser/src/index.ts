/**
 * Diffusion / multimodal pipeline product surface (Living `11` · Wave 4).
 * Workspace-only until `diffuser-pipeline` closes (not on publish-npm).
 * Package name is `@dxo/diffuser` — never `@dxo/generate`.
 *
 * Generation pixels stay on `OutputBuffer` / device handle; preview uses explicit
 * low-res readback, thumbnail, or device-resident handle — never implicit full host copy.
 */

import type { OutputBuffer, TypedBuffer } from '@dxo/core';

export type DiffusionPipelineKind = 'text-to-image' | 'image-to-image';

export type SchedulerKind = 'ddim' | 'euler' | 'unipc' | string;

export type DiffusionCondition = {
    prompt: string;
    negativePrompt?: string;
    /** Optional image condition for image-to-image (vision `Image` / buffer later). */
    image?: unknown;
    strength?: number;
};

export type DiffusionGenerateConfig = {
    width?: number;
    height?: number;
    steps?: number;
    guidanceScale?: number;
    seed?: number;
    scheduler?: SchedulerKind;
    /** Emit preview events when the runtime can produce them without full host copy. */
    preview?: boolean;
};

export type ImageResult = {
    readonly width: number;
    readonly height: number;
    readonly seed: number;
    readonly output: OutputBuffer;
    readback(options?: { copy?: boolean; maxSide?: number }): Promise<TypedBuffer>;
};

export type DiffusionProgressEvent =
    | { type: 'progress'; step: number; total: number; requestId: string }
    | {
          type: 'preview';
          step: number;
          requestId: string;
          thumbnail?: TypedBuffer;
          output?: OutputBuffer;
      }
    | { type: 'result'; requestId: string; result: ImageResult }
    | { type: 'warning'; requestId: string; message: string; code?: string };

export type DiffusionGenerateContext = {
    signal?: AbortSignal;
    requestId?: string;
};

export type DiffusionPipelineManifest = {
    readonly id: string;
    readonly kind: DiffusionPipelineKind;
    readonly components: readonly string[];
    readonly capabilities?: {
        readonly backend?: string;
        readonly availability?: 'available' | 'unavailable';
        readonly reason?: string;
    };
};

export type DiffusionPipeline = {
    readonly manifest: DiffusionPipelineManifest;
    generate(
        condition: DiffusionCondition,
        config?: DiffusionGenerateConfig,
        ctx?: DiffusionGenerateContext,
    ): AsyncIterable<DiffusionProgressEvent>;
};

export function diffuserVersion(): string {
    return 'dxo-diffuser@placeholder';
}

/**
 * Reserve the pipeline factory. Real weights / schedulers land with `diffuser-pipeline`.
 */
export async function createDiffusionPipeline(_ref: string): Promise<DiffusionPipeline> {
    unsupportedDiffuserApi('createDiffusionPipeline');
}

export function unsupportedDiffuserApi(name: string): never {
    throw new Error(
        `@dxo/diffuser ${name} is a workspace stub; DiffusionPipeline lands with diffuser-pipeline after device-residency`,
    );
}
