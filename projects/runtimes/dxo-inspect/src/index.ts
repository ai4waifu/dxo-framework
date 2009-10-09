/**
 * @dxo/inspect — Run / metric / artifact / profile protocols (v0).
 */

export {
    INSPECT_PROTOCOL,
    INSPECT_PROTOCOL_VERSION,
    inspectVersion,
    type ArtifactKindV0,
    type ArtifactV0,
    type InspectEventV0,
    type MetricV0,
    type ProfileSpan,
    type ProfileTraceV0,
    type RunMetaV0,
    type RunStatus,
} from './schemas.js';

export { RunRecorder, defaultRunsRoot, listRuns, readEvents, readRunMeta } from './store.js';
export type { RunRecorderOptions, RunSummary } from './store.js';
export { recordTrainIter, trainEventToInspectEvents } from './train-adapter.js';
export type { RecordTrainIterOptions, TrainEventLike } from './train-adapter.js';
