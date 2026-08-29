import type { Tensor } from '@dxo/core';
import { VisionError } from './errors.js';

export type LabelSpace = {
    id: string;
    labels: readonly string[];
    size: number;
    localized?: Readonly<Record<string, readonly string[]>>;
};

export type LabelSpaceOptions = {
    id: string;
    labels: readonly string[];
    localized?: Record<string, readonly string[]>;
};

export function defineLabelSpace(options: LabelSpaceOptions): LabelSpace {
    if (!options.id) {
        throw new VisionError('INVALID_LABEL_SPACE', 'defineLabelSpace requires a non-empty id');
    }
    if (!Array.isArray(options.labels) || options.labels.length === 0) {
        throw new VisionError('INVALID_LABEL_SPACE', 'defineLabelSpace requires a non-empty labels array');
    }
    const localized = options.localized
        ? Object.fromEntries(Object.entries(options.localized).map(([locale, labels]) => [locale, Object.freeze([...labels])]))
        : undefined;
    if (localized && Object.values(localized).some((labels) => labels.length !== options.labels.length)) {
        throw new VisionError('INVALID_LABEL_SPACE', 'localized label arrays must match the primary label count');
    }
    return {
        id: options.id,
        labels: Object.freeze([...options.labels]),
        size: options.labels.length,
        ...(localized ? { localized } : {}),
    };
}

export type BilingualLabelSpaceOptions = {
    id: string;
    english: readonly string[];
    chinese: readonly string[];
};

/** Define aligned English/Chinese labels for the same logits indices. */
export function defineBilingualLabelSpace(options: BilingualLabelSpaceOptions): LabelSpace {
    if (options.english.length !== options.chinese.length) {
        throw new VisionError('INVALID_LABEL_SPACE', 'English and Chinese labels must have the same length');
    }
    return defineLabelSpace({
        id: options.id,
        labels: options.english,
        localized: { en: options.english, zh: options.chinese },
    });
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
    locale?: string;
};

/**
 * Decode logits with an external label space.
 * Same logits may be decoded with different LabelSpaces (e.g. EN vs ZH).
 */
export async function decodeClassification(logits: Tensor, options: DecodeClassificationOptions): Promise<ClassificationDecode> {
    const { labels } = options;
    const k = Math.max(1, Math.min(options.topK ?? 5, labels.size));
    const shape = logits.shape;
    if (shape.length < 1) {
        throw new VisionError('INVALID_LOGITS', 'logits must be at least rank 1');
    }
    const classDim = shape[shape.length - 1]!;
    if (classDim !== labels.size) {
        throw new VisionError('LABEL_SIZE_MISMATCH', `logits class dim=${classDim} does not match labelSpace size=${labels.size}`);
    }

    const data = await logits.toArray();
    // Use last row for batched [N, C]; full vector for [C].
    const offset = shape.length === 1 ? 0 : (shape[0]! - 1) * classDim;
    const scores = data.slice(offset, offset + classDim);
    const localized = options.locale && labels.localized?.[options.locale];
    if (options.locale && !localized) {
        throw new VisionError('UNKNOWN_LABEL_LOCALE', `label space '${labels.id}' has no locale '${options.locale}'`);
    }
    const order = scores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);

    return {
        labelSpaceId: labels.id,
        topK: order.map(({ score, index }) => ({
            index,
            score,
            label: (localized ?? labels.labels)[index]!,
        })),
    };
}
