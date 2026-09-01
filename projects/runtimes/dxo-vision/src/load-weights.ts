import { readFile } from 'node:fs/promises';
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
    /**
     * `backbone` (default): apply only keys in `model.parameterNames()`; ignore extras (e.g. `fc.*`).
     * `all`: require every tensor key to be a known parameter name (still no torch rename).
     */
    scope?: LoadWeightsScope;
    requiresGrad?: boolean;
};

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
    if ((options.path == null) === (options.bytes == null)) {
        throw new VisionError('INVALID_LOAD_SOURCE', 'loadWeights requires exactly one of path or bytes');
    }

    let bytes: Uint8Array;
    if (options.bytes) {
        bytes = options.bytes;
    } else {
        bytes = new Uint8Array(await readFile(options.path!));
    }

    const decoded = decodeState(bytes);
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
