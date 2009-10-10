/**
 * UI-agnostic model graph IR placeholder (Living `10` / `@dxo/graph`).
 * Module graph ≠ execution trace ≠ kernel graph — do not collapse them.
 */

export const MODEL_GRAPH_FORMAT = 'dxo-model-graph' as const;
export const MODEL_GRAPH_VERSION = 0 as const;

export type GraphViewKind = 'module' | 'execution' | 'kernel';

export type GraphNode = {
    id: string;
    kind: string;
    label?: string;
    modulePath?: string;
    attrs?: Record<string, unknown>;
};

export type GraphEdge = {
    id: string;
    source: string;
    target: string;
    tensor?: { shape?: number[]; dtype?: string; device?: string };
};

export type ModelGraphV0 = {
    format: typeof MODEL_GRAPH_FORMAT;
    version: typeof MODEL_GRAPH_VERSION;
    view: GraphViewKind;
    nodes: GraphNode[];
    edges: GraphEdge[];
};

export function graphVersion(): string {
    return `${MODEL_GRAPH_FORMAT}@${MODEL_GRAPH_VERSION}`;
}

export {
    emptyModelGraph,
    moduleGraphFromLinear,
    moduleGraphFromModule,
    moduleGraphFromSequential,
    serializeModelGraph,
} from './module-graph.js';
