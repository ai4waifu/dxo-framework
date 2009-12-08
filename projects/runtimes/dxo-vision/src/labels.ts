import type { Tensor } from '@dxo/core';
import { VisionError } from './errors.js';

export type LabelSpace = {
    id: string;
    labels: readonly string[];
    size: number;
};

export type LabelSpaceOptions = {
    id: string;
    labels: readonly string[];
};

export function defineLabelSpace(options: LabelSpaceOptions): LabelSpace {
    if (!options.id) {
        throw new VisionError('INVALID_LABEL_SPACE', 'defineLabelSpace requires a non-empty id');
    }
    if (!Array.isArray(options.labels) || options.labels.length === 0) {
        throw new VisionError('INVALID_LABEL_SPACE', 'defineLabelSpace requires a non-empty labels array');
    }
    return {
        id: options.id,
        labels: Object.freeze([...options.labels]),
        size: options.labels.length,
    };
}

export type ClassificationTopK = {
    index: number;
    score: number;
    label: string;
};

export type ClassificationDecode = {
    labelSpaceId: string;
    topK: ClassificationTopK[];
};

export type DecodeClassificationOptions = {
    labels: LabelSpace;
    topK?: number;
};

/**
 * Decode logits with an external label space.
 * Same logits may be decoded with different LabelSpaces (e.g. EN vs ZH).
 */
export async function decodeClassification(
    logits: Tensor,
    options: DecodeClassificationOptions,
): Promise<ClassificationDecode> {
    const { labels } = options;
    const k = Math.max(1, Math.min(options.topK ?? 5, labels.size));
    const shape = logits.shape;
    if (shape.length < 1) {
        throw new VisionError('INVALID_LOGITS', 'logits must be at least rank 1');
    }
    const classDim = shape[shape.length - 1]!;
    if (classDim !== labels.size) {
        throw new VisionError(
            'LABEL_SIZE_MISMATCH',
            `logits class dim=${classDim} does not match labelSpace size=${labels.size}`,
        );
    }

    const data = await logits.toArray();
    // Use last row for batched [N, C]; full vector for [C].
    const offset = shape.length === 1 ? 0 : (shape[0]! - 1) * classDim;
    const scores = data.slice(offset, offset + classDim);
    const order = scores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);

    return {
        labelSpaceId: labels.id,
        topK: order.map(({ score, index }) => ({
            index,
            score,
            label: labels.labels[index]!,
        })),
    };
}
