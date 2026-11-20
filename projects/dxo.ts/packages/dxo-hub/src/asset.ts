/**
 * Shared Asset catalog types (Living `14`).
 * Kind taxonomy: neural | weights | dataset | tokenizer | label-space | config | artifact.
 * "pretrained" is metadata, never a kind.
 */

export type AssetKind = 'neural' | 'weights' | 'dataset' | 'tokenizer' | 'label-space' | 'config' | 'artifact';

/** Typed binary / definition locator — not HF-only. */
export type AssetSource =
    | {
          provider: 'hf';
          repo: string;
          revision?: string;
          path: string;
      }
    | {
          provider: 'local';
          path: string;
      }
    | {
          provider: 'http' | 'https';
          url: string;
      }
    | {
          provider: 's3' | 'r2';
          uri: string;
      }
    | {
          provider: 'modelscope';
          repo: string;
          revision?: string;
          path: string;
      };

export type AssetBase = {
    readonly kind: AssetKind;
    readonly id: string;
};

export type WeightAsset = AssetBase & {
    readonly kind: 'weights';
    readonly source: AssetSource;
    readonly format: 'safetensors' | 'onnx' | 'custom';
    readonly shards?: readonly string[];
    readonly storage?: {
        dtype?: string;
        quantization?: string;
        layout?: string;
    };
};

export type NeuralAssetTraining = {
    status?: 'pretrained' | 'scratch' | 'finetuned';
    source?: string;
};

export type NeuralAsset<TCreate extends (...args: never[]) => unknown = () => unknown> = AssetBase & {
    readonly kind: 'neural';
    readonly create: TCreate;
    /** Weight asset ids or inline WeightAsset entries. */
    readonly weights?: readonly (string | WeightAsset)[];
    readonly training?: NeuralAssetTraining;
    readonly metadata?: Record<string, unknown>;
};

export type DatasetAsset = AssetBase & {
    readonly kind: 'dataset';
    readonly source?: AssetSource;
    /** Open-ended split names (train/test/validation/…). */
    readonly splits?: readonly string[];
    readonly metadata?: Record<string, unknown>;
};

export type Asset = AssetBase | WeightAsset | NeuralAsset | DatasetAsset;

export type AssetCatalog<A extends readonly Asset[] = readonly Asset[]> = {
    readonly assets: A;
};

export function defineAsset<const A extends Asset>(asset: A): A {
    if (!asset.id) {
        throw new Error('@dxo/hub defineAsset: id is required');
    }
    if (!asset.kind) {
        throw new Error('@dxo/hub defineAsset: kind is required');
    }
    return asset;
}

export function defineAssets<const A extends readonly Asset[]>(input: { assets: A }): AssetCatalog<A> {
    return { assets: input.assets };
}
