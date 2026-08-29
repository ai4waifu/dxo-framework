import type { NeuralNetwork } from '@dxo/nn';
import { Linear, Relu, Sequential } from '@dxo/nn';
import { type GraphEdge, type GraphNode, type GraphViewKind, MODEL_GRAPH_FORMAT, MODEL_GRAPH_VERSION, type ModelGraphV0 } from './index.js';

export function emptyModelGraph(view: GraphViewKind = 'module'): ModelGraphV0 {
    return {
        format: MODEL_GRAPH_FORMAT,
        version: MODEL_GRAPH_VERSION,
        view,
        availability: 'ready',
        nodes: [],
        edges: [],
    };
}

export function serializeModelGraph(graph: ModelGraphV0): string {
    return JSON.stringify(graph);
}

function node(id: string, kind: string, label: string, modulePath?: string, attrs?: Record<string, unknown>): GraphNode {
    return { id, kind, label, modulePath, attrs };
}

function edge(id: string, source: string, target: string, tensor?: GraphEdge['tensor']): GraphEdge {
    return { id, source, target, tensor };
}

export function moduleGraphFromLinear(linear: Linear, path = 'linear'): ModelGraphV0 {
    const nodes: GraphNode[] = [
        node('input', 'tensor', 'input'),
        node('linear', 'Linear', 'Linear', path, {
            inFeatures: linear.inFeatures,
            outFeatures: linear.outFeatures,
            params: ['weight', 'bias'],
        }),
        node('output', 'tensor', 'output'),
    ];
    const edges: GraphEdge[] = [
        edge('e-in-linear', 'input', 'linear', { dtype: 'f32' }),
        edge('e-linear-out', 'linear', 'output', { dtype: 'f32' }),
    ];
    return { format: MODEL_GRAPH_FORMAT, version: MODEL_GRAPH_VERSION, view: 'module', nodes, edges };
}

export function moduleGraphFromSequential(seq: Sequential, path = 'sequential'): ModelGraphV0 {
    const nodes: GraphNode[] = [node('input', 'tensor', 'input')];
    const edges: GraphEdge[] = [];
    let prev = 'input';

    seq.layers.forEach((layer, i) => {
        const id = `layer_${i}`;
        if (layer instanceof Linear) {
            nodes.push(
                node(id, 'Linear', `Linear(${layer.inFeatures}->${layer.outFeatures})`, `${path}/${id}`, {
                    inFeatures: layer.inFeatures,
                    outFeatures: layer.outFeatures,
                }),
            );
        } else if (layer instanceof Relu) {
            nodes.push(node(id, 'Relu', 'Relu', `${path}/${id}`));
        } else {
            nodes.push(node(id, layer.constructor.name, layer.constructor.name, `${path}/${id}`));
        }
        edges.push(edge(`e-${prev}-${id}`, prev, id, { dtype: 'f32' }));
        prev = id;
    });

    nodes.push(node('output', 'tensor', 'output'));
    edges.push(edge(`e-${prev}-output`, prev, 'output', { dtype: 'f32' }));

    return { format: MODEL_GRAPH_FORMAT, version: MODEL_GRAPH_VERSION, view: 'module', nodes, edges };
}

/** Build a model graph from any @dxo/nn Neural (Linear or Sequential supported). */
export function moduleGraphFromModule(mod: NeuralNetwork, path = 'model'): ModelGraphV0 {
    if (mod instanceof Sequential) return moduleGraphFromSequential(mod, path);
    if (mod instanceof Linear) return moduleGraphFromLinear(mod, path);
    throw new Error(`moduleGraphFromModule: unsupported module ${mod.constructor.name}`);
}
