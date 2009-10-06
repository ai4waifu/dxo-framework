/**
 * Artifact hub stub (Living `10` U4). Not wired to npm publish / placeholder OIDC yet.
 */

export type HubProviderId = 'hf' | 'modelscope' | 'local' | 'http' | 's3' | 'r2';

export type HubModelRef = {
    /** Explicit scheme, e.g. `hf:org/model` — no fuzzy guessing. */
    uri: string;
    revision?: string;
    files?: string[];
};

export type HubResolvedArtifact = {
    provider: HubProviderId;
    uri: string;
    revision: string;
    digest?: string;
    localPath?: string;
};

export type Hub = {
    readonly kind: 'placeholder';
    model(ref: string | HubModelRef, options?: { signal?: AbortSignal }): Promise<HubResolvedArtifact>;
};

export function createHub(): Hub {
    return {
        kind: 'placeholder',
        async model() {
            throw new Error('@dxo/hub createHub().model() is a workspace stub; providers land with hub-provider gate');
        },
    };
}

export function hubVersion(): string {
    return 'dxo-hub@placeholder';
}
