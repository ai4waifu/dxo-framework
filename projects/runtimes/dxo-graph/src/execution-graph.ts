import type { Module } from '@dxo/nn';
import { Linear, Relu, Sequential } from '@dxo/nn';
import { type GraphEdge, type GraphNode, MODEL_GRAPH_FORMAT, MODEL_GRAPH_VERSION, type ModelGraphV0 } from './index.js';

function node(id: string, kind: string, label: string, modulePath?: string, attrs?: Record<string, unknown>): GraphNode {
    return { id, kind, label, modulePath, attrs };
}

function edge(id: string, source: string, target: string, tensor?: GraphEdge['tensor']): GraphEdge {
    return { id, source, target, tensor };
}

/** Structured unavailable execution view (no invented ops). */
export function executionGraphUnavailable(reason: string): ModelGraphV0 {
    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view: 'execution',
        availability: 'unavailable',
        unavailableReason: reason,
        nodes: [],
        edges: [],
    };
}

/**
 * Execution view for Linear: one matmul+bias op with concrete input/output shapes.
 * This is an *execution* sketch for a given input shape — not the module hierarchy.
 */
export function executionGraphFromLinear(linear: Linear, inputShape: number[], path = 'linear'): ModelGraphV0 {
    if (inputShape.length < 1) throw new Error('executionGraphFromLinear: empty inputShape');
    const last = inputShape[inputShape.length - 1]!;
    if (last !== linear.inFeatures) {
        throw new Error(`executionGraphFromLinear: last dim ${last} != inFeatures ${linear.inFeatures}`);
    }
    const outShape = [...inputShape.slice(0, -1), linear.outFeatures];
    const nodes: GraphNode[] = [
        node('in', 'tensor', 'input', undefined, { shape: inputShape, dtype: 'f32', device: 'cpu' }),
        node('op_linear', 'op', 'linear', path, {
            op: 'addmm',
            inFeatures: linear.inFeatures,
            outFeatures: linear.outFeatures,
        }),
        node('out', 'tensor', 'output', undefined, { shape: outShape, dtype: 'f32', device: 'cpu' }),
    ];
    const edges: GraphEdge[] = [
        edge('e-in-op', 'in', 'op_linear', { shape: inputShape, dtype: 'f32', device: 'cpu' }),
        edge('e-op-out', 'op_linear', 'out', { shape: outShape, dtype: 'f32', device: 'cpu' }),
    ];
    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view: 'execution',
        availability: 'ready',
        nodes,
        edges,
    };
}

/** Execution view for Sequential: one op node per layer along the actual control path. */
export function executionGraphFromSequential(seq: Sequential, inputShape: number[], path = 'sequential'): ModelGraphV0 {
    const nodes: GraphNode[] = [node('in', 'tensor', 'input', undefined, { shape: [...inputShape], dtype: 'f32', device: 'cpu' })];
    const edges: GraphEdge[] = [];
    let prev = 'in';
    let shape = [...inputShape];

    seq.layers.forEach((layer, i) => {
        const id = `op_${i}`;
        if (layer instanceof Linear) {
            const last = shape[shape.length - 1]!;
            if (last !== layer.inFeatures) {
                throw new Error(`executionGraphFromSequential: layer ${i} expects ${layer.inFeatures}, got ${last}`);
            }
            const outShape = [...shape.slice(0, -1), layer.outFeatures];
            nodes.push(
                node(id, 'op', `linear`, `${path}/layer_${i}`, {
                    op: 'addmm',
                    inFeatures: layer.inFeatures,
                    outFeatures: layer.outFeatures,
                }),
            );
            edges.push(edge(`e-${prev}-${id}`, prev, id, { shape: [...shape], dtype: 'f32', device: 'cpu' }));
            shape = outShape;
        } else if (layer instanceof Relu) {
            nodes.push(node(id, 'op', 'relu', `${path}/layer_${i}`, { op: 'relu' }));
            edges.push(edge(`e-${prev}-${id}`, prev, id, { shape: [...shape], dtype: 'f32', device: 'cpu' }));
        } else {
            nodes.push(node(id, 'op', layer.constructor.name, `${path}/layer_${i}`, { op: layer.constructor.name }));
            edges.push(edge(`e-${prev}-${id}`, prev, id, { shape: [...shape], dtype: 'f32', device: 'cpu' }));
        }
        prev = id;
    });

    nodes.push(node('out', 'tensor', 'output', undefined, { shape: [...shape], dtype: 'f32', device: 'cpu' }));
    edges.push(edge(`e-${prev}-out`, prev, 'out', { shape: [...shape], dtype: 'f32', device: 'cpu' }));

    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view: 'execution',
        availability: 'ready',
        nodes,
        edges,
    };
}

export function executionGraphFromModule(mod: Module, inputShape: number[], path = 'model'): ModelGraphV0 {
    if (mod instanceof Sequential) return executionGraphFromSequential(mod, inputShape, path);
    if (mod instanceof Linear) return executionGraphFromLinear(mod, inputShape, path);
    return executionGraphUnavailable(`no execution lowering for ${mod.constructor.name}`);
}
