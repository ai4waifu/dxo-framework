/**
 * Model graph IR — module ≠ execution ≠ kernel (never collapse into one diagram).
 */

export const MODEL_GRAPH_FORMAT = 'dxo-model-graph' as const;
export const MODEL_GRAPH_VERSION = 0 as const;
export const MODEL_GRAPH_BUNDLE_FORMAT = 'dxo-model-graph-bundle' as const;

export type GraphViewKind = 'module' | 'execution' | 'kernel';

export type GraphAvailability = 'ready' | 'unavailable';

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
    /** Kernel (and optionally execution) may be unavailable without a profile trace / real forward. */
    availability?: GraphAvailability;
    unavailableReason?: string;
    nodes: GraphNode[];
    edges: GraphEdge[];
};

/** Three strictly separated views for Studio Models tabs. */
export type ModelGraphBundleV0 = {
    format: typeof MODEL_GRAPH_BUNDLE_FORMAT;
    version: 0;
    module: ModelGraphV0;
    execution: ModelGraphV0;
    kernel: ModelGraphV0;
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

export {
    executionGraphFromLinear,
    executionGraphFromModule,
    executionGraphFromSequential,
    executionGraphUnavailable,
} from './execution-graph.js';

export {
    bundleModelGraphs,
    kernelGraphFromProfile,
    kernelGraphUnavailable,
    parseModelGraphArtifact,
} from './kernel-graph.js';
