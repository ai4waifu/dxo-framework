import { readFile } from 'node:fs/promises';
import { DxoAsset } from '@dxo/core';
import type { TensorStateSlice } from '@dxo/nn';
import { decodeState } from '@dxo/serialize';
import { VisionError } from './errors.js';
import type { ResNet } from './resnet.js';

export type LoadWeightsScope = 'backbone' | 'all';

export type LoadWeightsOptions = {
    /** Local filesystem path to a DXO-key safetensors file. */
    path?: string;
    /** Pre-loaded safetensors bytes (DXO keys). */
    bytes?: Uint8Array;
    /** Local filesystem path to a `.dxo` weights container (kind=weights, f32 tensors). */
    assetPath?: string;
    /**
     * `backbone` (default): apply only keys in `model.parameterNames()`; ignore extras (e.g. `fc.*`).
     * `all`: require every tensor key to be a known parameter name (still no torch rename).
     */
    scope?: LoadWeightsScope;
    requiresGrad?: boolean;
};

type DxoWeightsMeta = {
    kind: string;
    tensors: Array<{ key: string; dtype: string; shape: number[] }>;
};

/** Read every tensor of a `weights` `.dxo` container through the napi reader. */
function decodeDxoWeights(assetPath: string): Record<string, TensorStateSlice> {
    const asset = new DxoAsset(assetPath);
    const meta = JSON.parse(asset.metadataJson()) as DxoWeightsMeta;
    if (meta.kind !== 'weights') {
        throw new VisionError('INVALID_LOAD_SOURCE', `loadWeights: ${assetPath} is kind=${meta.kind}, expected weights`);
    }
    const out: Record<string, TensorStateSlice> = {};
    for (const tensor of meta.tensors) {
        if (tensor.dtype !== 'f32') {
            throw new VisionError('UNSUPPORTED', `loadWeights: ${tensor.key} dtype ${tensor.dtype}, expected f32`);
        }
        const raw = asset.readTensor(tensor.key);
        out[tensor.key] = { shape: [...tensor.shape], data: Array.from(new Float32Array(Uint8Array.from(raw).buffer)) };
    }
    return out;
}

function isResNetLike(model: { parameterNames?: () => string[]; loadState?: unknown }): model is ResNet {
    return typeof model.parameterNames === 'function' && typeof model.loadState === 'function';
}

/**
 * Load **DXO-native** safetensors into a ResNet backbone.
 * Does **not** translate torchvision / `.pth` names — convert offline in external `@dxo/resnet` scripts.
 */
export async function loadWeights(model: ResNet, options: LoadWeightsOptions): Promise<void> {
    if (!isResNetLike(model)) {
        throw new VisionError('INVALID_LOAD_TARGET', 'loadWeights expects a ResNet with parameterNames/loadState');
    }
    const sources = [options.path, options.bytes, options.assetPath].filter((v) => v != null).length;
    if (sources !== 1) {
        throw new VisionError('INVALID_LOAD_SOURCE', 'loadWeights requires exactly one of path, bytes or assetPath');
    }

    let decoded: Record<string, TensorStateSlice>;
    if (options.assetPath) {
        decoded = decodeDxoWeights(options.assetPath);
    } else {
        const bytes = options.bytes ?? new Uint8Array(await readFile(options.path!));
        decoded = decodeState(bytes);
    }
    const scope = options.scope ?? 'backbone';
    const allowed = new Set(model.parameterNames());
    if (allowed.size === 0) {
        throw new VisionError('UNSUPPORTED', 'loadWeights: model has no wired parameterNames (depth=18 only today)');
    }

    const filtered: Record<string, TensorStateSlice> = {};
    for (const [key, slice] of Object.entries(decoded)) {
        if (allowed.has(key)) {
            filtered[key] = slice;
            continue;
        }
        if (scope === 'all') {
            throw new VisionError(
                'UNKNOWN_WEIGHT_KEY',
                `loadWeights scope=all: unexpected key '${key}' (not in DXO parameterNames; convert pth offline)`,
            );
        }
        // backbone: ignore foreign keys (fc.*, torch leftovers)
    }

    for (const name of allowed) {
        if (!filtered[name]) {
            throw new VisionError('MISSING_STATE_KEY', `loadWeights: missing DXO key '${name}'`);
        }
    }

    model.loadState(filtered, { requiresGrad: options.requiresGrad ?? false });
}
