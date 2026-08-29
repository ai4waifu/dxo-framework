import type { Tensor } from '@dxo/core';
import type { Batch } from '@dxo/data';
import type { FullyConnected } from '@dxo/nn';
import type { Optimizer } from '@dxo/optimizer';
import { encodeState, type State } from '@dxo/serialize';

/** Events yielded by {@link Trainer.fitIter} / {@link Trainer.run}. */
export type TrainEvent =
    | { type: 'epoch_start'; epoch: number; epochs: number }
    | { type: 'batch'; epoch: number; step: number; loss: number }
    | { type: 'epoch_end'; epoch: number; meanLoss: number; steps: number }
    | { type: 'checkpoint'; epoch: number; state: State; bytes: Uint8Array; format: 'safetensors' }
    | { type: 'aborted'; reason: 'signal'; epoch: number; step: number }
    | { type: 'done'; epochs: number; steps: number; finalMeanLoss?: number };

export interface FitSummary {
    epochs: number;
    steps: number;
    aborted: boolean;
    finalMeanLoss?: number;
    /** Last checkpoint document, if any was emitted. */
    lastCheckpoint?: Uint8Array;
    lastState?: State;
}

export type BatchSource = Iterable<Batch> | AsyncIterable<Batch>;

export interface TrainerOptions {
    model: FullyConnected;
    optimizer: Optimizer;
    /** Called each epoch so iterators restart (sync or async batches). */
    batches: () => BatchSource;
    epochs: number;
    /**
     * Scalar loss from prediction and target. Default: mean squared error.
     * Must return a tensor with `numel === 1`.
     */
    loss?: (pred: Tensor, y: Tensor) => Tensor;
    /** Emit a checkpoint every N epochs (and always after the last completed epoch). Default: every epoch. */
    checkpointEvery?: number;
}

function isAsyncIterable(source: BatchSource): source is AsyncIterable<Batch> {
    return typeof (source as AsyncIterable<Batch>)[Symbol.asyncIterator] === 'function';
}

async function* iterateBatches(source: BatchSource): AsyncGenerator<Batch, void, undefined> {
    if (isAsyncIterable(source)) {
        for await (const b of source) yield b;
        return;
    }
    for (const b of source) yield b;
}

/** Default MSE: `mean((pred - y)^2)`. */
export function mseLoss(pred: Tensor, y: Tensor): Tensor {
    const diff = pred.sub(y);
    return diff.mul(diff).mean();
}

/**
 * Minimal CPU training loop (G5 / 0.0.7).
 *
 * GPU / multi-device is out of scope; declare CPU-only when GPU is deferred.
 */
export class Trainer {
    readonly model: FullyConnected;
    readonly optimizer: Optimizer;
    readonly epochs: number;
    readonly checkpointEvery: number;
    private readonly batches: () => BatchSource;
    private readonly lossFn: (pred: Tensor, y: Tensor) => Tensor;

    constructor(options: TrainerOptions) {
        if (!(options.epochs > 0) || !Number.isInteger(options.epochs)) {
            throw new Error('epochs must be a positive integer');
        }
        const every = options.checkpointEvery ?? 1;
        if (!(every > 0) || !Number.isInteger(every)) {
            throw new Error('checkpointEvery must be a positive integer');
        }
        this.model = options.model;
        this.optimizer = options.optimizer;
        this.epochs = options.epochs;
        this.checkpointEvery = every;
        this.batches = options.batches;
        this.lossFn = options.loss ?? mseLoss;
    }

    /** Alias for {@link fitIter} (Living API name). */
    run(opts?: { signal?: AbortSignal }): AsyncGenerator<TrainEvent, void, undefined> {
        return this.fitIter(opts);
    }

    /**
     * Yield training events. Honors `signal.aborted` between batches.
     * Checkpoint payloads are in-memory safetensors bytes — callers persist them.
     */
    async *fitIter(opts?: { signal?: AbortSignal }): AsyncGenerator<TrainEvent, void, undefined> {
        const signal = opts?.signal;
        let globalStep = 0;
        let lastMean: number | undefined;

        for (let epoch = 1; epoch <= this.epochs; epoch++) {
            if (signal?.aborted) {
                yield { type: 'aborted', reason: 'signal', epoch, step: globalStep };
                return;
            }

            yield { type: 'epoch_start', epoch, epochs: this.epochs };

            let epochLossSum = 0;
            let epochSteps = 0;

            for await (const batch of iterateBatches(this.batches())) {
                if (signal?.aborted) {
                    yield { type: 'aborted', reason: 'signal', epoch, step: globalStep };
                    return;
                }
                if (!batch.y) {
                    throw new Error('Trainer requires batch.y targets');
                }

                this.model.zeroGrad();
                const pred = this.model.forward(batch.x);
                const loss = this.lossFn(pred, batch.y);
                const value = await loss.item();
                if (!Number.isFinite(value)) {
                    throw new Error(`non-finite loss at epoch ${epoch} step ${globalStep + 1}: ${value}`);
                }
                loss.backward();
                this.model.loadParameters(await this.optimizer.step(this.model.parameters()));

                globalStep += 1;
                epochSteps += 1;
                epochLossSum += value;
                yield { type: 'batch', epoch, step: globalStep, loss: value };
            }

            if (epochSteps === 0) {
                throw new Error(`epoch ${epoch} produced zero batches`);
            }

            lastMean = epochLossSum / epochSteps;
            yield { type: 'epoch_end', epoch, meanLoss: lastMean, steps: epochSteps };

            const shouldCheckpoint = epoch % this.checkpointEvery === 0 || epoch === this.epochs;
            if (shouldCheckpoint) {
                const state = (await this.model.state()) as unknown as State;
                yield {
                    type: 'checkpoint',
                    epoch,
                    state,
                    bytes: encodeState(state),
                    format: 'safetensors',
                };
            }
        }

        yield { type: 'done', epochs: this.epochs, steps: globalStep, finalMeanLoss: lastMean };
    }

    /** Consume {@link fitIter} and return a summary (including last checkpoint document). */
    async fit(opts?: { signal?: AbortSignal }): Promise<FitSummary> {
        let epochs = 0;
        let steps = 0;
        let aborted = false;
        let finalMeanLoss: number | undefined;
        let lastCheckpoint: Uint8Array | undefined;
        let lastState: State | undefined;

        for await (const event of this.fitIter(opts)) {
            switch (event.type) {
                case 'batch':
                    steps = event.step;
                    break;
                case 'epoch_end':
                    epochs = event.epoch;
                    finalMeanLoss = event.meanLoss;
                    break;
                case 'checkpoint':
                    lastCheckpoint = event.bytes;
                    lastState = event.state;
                    break;
                case 'aborted':
                    aborted = true;
                    epochs = event.epoch;
                    steps = event.step;
                    break;
                case 'done':
                    epochs = event.epochs;
                    steps = event.steps;
                    finalMeanLoss = event.finalMeanLoss;
                    break;
                default:
                    break;
            }
        }

        return { epochs, steps, aborted, finalMeanLoss, lastCheckpoint, lastState };
    }
}
