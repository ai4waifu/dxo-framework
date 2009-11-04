export const INSPECT_PROTOCOL = 'dxo-inspect' as const;
export const INSPECT_PROTOCOL_VERSION = 0 as const;

export type RunStatus = 'running' | 'ok' | 'cancelled' | 'error';

export type RunMetaV0 = {
    format: typeof INSPECT_PROTOCOL;
    version: typeof INSPECT_PROTOCOL_VERSION;
    runId: string;
    startedAtMs: number;
    endedAtMs?: number;
    label?: string;
    status: RunStatus;
    hyperparams?: Record<string, unknown>;
};

export type MetricV0 = {
    name: string;
    value: number;
    step: number;
    wallTimeMs: number;
};

export type ArtifactKindV0 = 'checkpoint' | 'state' | 'image-samples' | 'confusion-matrix' | 'other';

export type ImageSampleV0 = {
    uri: string;
    label?: number | string;
    pred?: number | string;
};

export type ImageSamplesArtifactV0 = {
    samples: ImageSampleV0[];
};

export type ConfusionMatrixArtifactV0 = {
    labels: string[];
    matrix: number[][];
};

export type ArtifactV0 = {
    name: string;
    kind: ArtifactKindV0;
    uri: string;
    digest?: string;
    wallTimeMs: number;
};

export type InspectEventV0 =
    | { type: 'run/start'; runId: string; wallTimeMs: number; meta?: Record<string, unknown> }
    | { type: 'run/end'; runId: string; wallTimeMs: number; status: Exclude<RunStatus, 'running'>; error?: string }
    | { type: 'train/epoch_start'; runId: string; wallTimeMs: number; epoch: number; epochs: number }
    | { type: 'train/batch'; runId: string; wallTimeMs: number; epoch: number; step: number; loss: number }
    | { type: 'train/epoch_end'; runId: string; wallTimeMs: number; epoch: number; meanLoss: number; steps: number }
    | { type: 'metric/scalar'; runId: string; wallTimeMs: number; metric: MetricV0 }
    | { type: 'artifact/ref'; runId: string; wallTimeMs: number; artifact: ArtifactV0 }
    | { type: 'log'; runId: string; wallTimeMs: number; level: 'info' | 'warn' | 'error'; message: string };

export type ProfileSpan = {
    name: string;
    category: 'op' | 'kernel' | 'transfer' | 'readback' | 'other';
    startMs: number;
    endMs: number;
    meta?: Record<string, unknown>;
};

export type ProfileTraceV0 = {
    format: 'dxo-profile';
    version: 0;
    spans: ProfileSpan[];
};

export function inspectVersion(): string {
    return `${INSPECT_PROTOCOL}@${INSPECT_PROTOCOL_VERSION}`;
}
