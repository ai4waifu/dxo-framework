/**
 * @dxo/inspect — Run / metric / artifact / profile protocols (v0).
 */

export {
    type ArtifactKindV0,
    type ArtifactV0,
    type ConfusionMatrixArtifactV0,
    type ImageSamplesArtifactV0,
    type ImageSampleV0,
    INSPECT_PROTOCOL,
    INSPECT_PROTOCOL_VERSION,
    type InspectEventV0,
    inspectVersion,
    type MetricV0,
    type ProfileSpan,
    type ProfileTraceV0,
    profileTraceUnavailable,
    type RunMetaV0,
    type RunStatus,
} from './schemas.js';
export type { RunRecorderOptions, RunSummary } from './store.js';
export { defaultRunsRoot, listRuns, RunRecorder, readEvents, readRunMeta } from './store.js';
export type { RecordTrainIterOptions, TrainEventLike } from './train-adapter.js';
export { recordTrainIter, trainEventToInspectEvents } from './train-adapter.js';
