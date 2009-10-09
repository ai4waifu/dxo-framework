import type { InspectEventV0 } from './schemas.js';
import { RunRecorder, type RunRecorderOptions } from './store.js';

/** Minimal TrainEvent shape (matches @dxo/train without a hard dependency). */
export type TrainEventLike =
    | { type: 'epoch_start'; epoch: number; epochs: number }
    | { type: 'batch'; epoch: number; step: number; loss: number }
    | { type: 'epoch_end'; epoch: number; meanLoss: number; steps: number }
    | {
          type: 'checkpoint';
          epoch: number;
          document: unknown;
          state: unknown;
      }
    | { type: 'aborted'; reason: 'signal'; epoch: number; step: number }
    | { type: 'done'; epochs: number; steps: number; finalMeanLoss?: number };

function wallTimeMs(): number {
    return Date.now();
}

export function trainEventToInspectEvents(runId: string, event: TrainEventLike): InspectEventV0[] {
    const t = wallTimeMs();
    switch (event.type) {
        case 'epoch_start':
            return [{ type: 'train/epoch_start', runId, wallTimeMs: t, epoch: event.epoch, epochs: event.epochs }];
        case 'batch':
            return [
                { type: 'train/batch', runId, wallTimeMs: t, epoch: event.epoch, step: event.step, loss: event.loss },
                {
                    type: 'metric/scalar',
                    runId,
                    wallTimeMs: t,
                    metric: { name: 'loss', value: event.loss, step: event.step, wallTimeMs: t },
                },
            ];
        case 'epoch_end':
            return [
                {
                    type: 'train/epoch_end',
                    runId,
                    wallTimeMs: t,
                    epoch: event.epoch,
                    meanLoss: event.meanLoss,
                    steps: event.steps,
                },
                {
                    type: 'metric/scalar',
                    runId,
                    wallTimeMs: t,
                    metric: {
                        name: 'mean_loss',
                        value: event.meanLoss,
                        step: event.epoch,
                        wallTimeMs: t,
                    },
                },
            ];
        case 'aborted':
            return [
                {
                    type: 'log',
                    runId,
                    wallTimeMs: t,
                    level: 'warn',
                    message: `training aborted at epoch ${event.epoch} step ${event.step}`,
                },
            ];
        case 'done':
            return [
                {
                    type: 'log',
                    runId,
                    wallTimeMs: t,
                    level: 'info',
                    message: `training done: epochs=${event.epochs} steps=${event.steps}`,
                },
            ];
        case 'checkpoint':
            return [];
        default:
            return [];
    }
}

export type RecordTrainIterOptions = RunRecorderOptions & {
    checkpointEncoder?: (document: unknown) => string;
};

/**
 * Consume a Trainer fitIter stream and append inspect events to the local run store.
 */
export async function recordTrainIter(
    events: AsyncIterable<TrainEventLike>,
    options: RecordTrainIterOptions = {},
): Promise<{ runId: string; recorder: RunRecorder; status: 'ok' | 'cancelled' }> {
    const recorder = await RunRecorder.open(options);
    const encoder = options.checkpointEncoder ?? ((doc) => JSON.stringify(doc));
    let status: 'ok' | 'cancelled' = 'ok';

    try {
        for await (const event of events) {
            if (event.type === 'checkpoint') {
                const name = `checkpoint-epoch-${event.epoch}.json`;
                await recorder.writeArtifact(name, 'checkpoint', encoder(event.document));
            }
            if (event.type === 'aborted') {
                status = 'cancelled';
            }
            for (const inspectEvent of trainEventToInspectEvents(recorder.runId, event)) {
                await recorder.append(inspectEvent);
            }
        }
        await recorder.close(status);
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        await recorder.close('error', message);
        throw err;
    }

    return { runId: recorder.runId, recorder, status };
}
