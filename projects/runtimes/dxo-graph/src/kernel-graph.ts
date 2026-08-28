import type { ProfileSpan, ProfileTraceV0 } from '@dxo/inspect';
import {
    type GraphEdge,
    type GraphNode,
    MODEL_GRAPH_BUNDLE_FORMAT,
    MODEL_GRAPH_FORMAT,
    MODEL_GRAPH_VERSION,
    type ModelGraphBundleV0,
    type ModelGraphV0,
} from './index.js';
import { emptyModelGraph } from './module-graph.js';

function node(id: string, kind: string, label: string, attrs?: Record<string, unknown>): GraphNode {
    return { id, kind, label, attrs };
}

function edge(id: string, source: string, target: string, tensor?: GraphEdge['tensor']): GraphEdge {
    return { id, source, target, tensor };
}

/** Empty kernel view with structured unavailable (never invent spans). */
export function kernelGraphUnavailable(reason: string): ModelGraphV0 {
    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view: 'kernel',
        availability: 'unavailable',
        unavailableReason: reason,
        nodes: [],
        edges: [],
    };
}

/**
 * Build kernel view **only** from real ProfileTrace spans.
 * Categories other than kernel/transfer/readback are ignored for this view.
 * If none exist, returns unavailable — does not fabricate timing.
 */
export function kernelGraphFromProfile(trace: ProfileTraceV0 | null | undefined): ModelGraphV0 {
    if (!trace || trace.format !== 'dxo-profile') {
        return kernelGraphUnavailable('no ProfileTraceV0 from runtime / inspect');
    }
    const spans = (trace.spans ?? []).filter(
        (s): s is ProfileSpan =>
            !!s &&
            (s.category === 'kernel' || s.category === 'transfer' || s.category === 'readback') &&
            typeof s.startMs === 'number' &&
            typeof s.endMs === 'number',
    );
    if (spans.length === 0) {
        return kernelGraphUnavailable('ProfileTrace has no kernel/transfer/readback spans');
    }

    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    let prev: string | null = null;
    spans.forEach((span, i) => {
        const id = `k${i}`;
        nodes.push(
            node(id, span.category, span.name, {
                startMs: span.startMs,
                endMs: span.endMs,
                durationMs: span.endMs - span.startMs,
                ...(span.meta ?? {}),
            }),
        );
        if (prev) {
            edges.push(edge(`e-${prev}-${id}`, prev, id));
        }
        prev = id;
    });

    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view: 'kernel',
        availability: 'ready',
        nodes,
        edges,
    };
}

export function bundleModelGraphs(parts: { module: ModelGraphV0; execution: ModelGraphV0; kernel: ModelGraphV0 }): ModelGraphBundleV0 {
    if (parts.module.view !== 'module') throw new Error('bundle: module.view must be module');
    if (parts.execution.view !== 'execution') throw new Error('bundle: execution.view must be execution');
    if (parts.kernel.view !== 'kernel') throw new Error('bundle: kernel.view must be kernel');
    return {
        format: MODEL_GRAPH_BUNDLE_FORMAT,
        version: 0,
        module: parts.module,
        execution: parts.execution,
        kernel: parts.kernel,
    };
}

/** Accept legacy single ModelGraph or a three-view bundle. */
export function parseModelGraphArtifact(raw: unknown): ModelGraphBundleV0 {
    if (!raw || typeof raw !== 'object') {
        const empty = emptyModelGraph('module');
        return bundleModelGraphs({
            module: empty,
            execution: {
                ...emptyModelGraph('execution'),
                availability: 'unavailable',
                unavailableReason: 'empty artifact',
            },
            kernel: kernelGraphUnavailable('empty artifact'),
        });
    }
    const obj = raw as Record<string, unknown>;
    if (obj.format === MODEL_GRAPH_BUNDLE_FORMAT) {
        return obj as unknown as ModelGraphBundleV0;
    }
    if (obj.format === MODEL_GRAPH_FORMAT && obj.view === 'module') {
        return bundleModelGraphs({
            module: obj as unknown as ModelGraphV0,
            execution: {
                ...emptyModelGraph('execution'),
                availability: 'unavailable',
                unavailableReason: 'legacy artifact had module view only',
            },
            kernel: kernelGraphUnavailable('legacy artifact had module view only'),
        });
    }
    if (obj.format === MODEL_GRAPH_FORMAT) {
        const g = obj as unknown as ModelGraphV0;
        return bundleModelGraphs({
            module: g.view === 'module' ? g : emptyModelGraph('module'),
            execution:
                g.view === 'execution'
                    ? g
                    : {
                          ...emptyModelGraph('execution'),
                          availability: 'unavailable',
                          unavailableReason: 'single-view artifact was not execution',
                      },
            kernel: g.view === 'kernel' ? g : kernelGraphUnavailable('single-view artifact was not kernel'),
        });
    }
    return bundleModelGraphs({
        module: emptyModelGraph('module'),
        execution: {
            ...emptyModelGraph('execution'),
            availability: 'unavailable',
            unavailableReason: 'unrecognized model-graph artifact',
        },
        kernel: kernelGraphUnavailable('unrecognized model-graph artifact'),
    });
}
