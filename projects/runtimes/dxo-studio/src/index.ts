import { graphVersion } from '@dxo/graph';
import { inspectVersion } from '@dxo/inspect';

/**
 * Placeholder entry for the future VMZ Studio app (Living `10` U1).
 * Does not start a server yet — protocol packages only.
 */

export type StudioOptions = {
    /** Default loopback only when a server exists. */
    host?: string;
    port?: number;
};

export type StudioHandle = {
    readonly kind: 'placeholder';
    readonly protocols: { inspect: string; graph: string };
};

export function createStudio(_options: StudioOptions = {}): StudioHandle {
    return {
        kind: 'placeholder',
        protocols: {
            inspect: inspectVersion(),
            graph: graphVersion(),
        },
    };
}

export function studioVersion(): string {
    return 'dxo-studio@placeholder';
}
