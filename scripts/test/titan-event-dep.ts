/**
 * titan-event-dep: HAL wait_event encodes upload→compute dependency.
 * CPU session always exercises the primitive; CUDA remains optional.
 */
import { backend, probeTitanEventDep, version } from '@dxo/core';

probeTitanEventDep();
console.log(`titan-event-dep ok: wait_event via ${backend()} (version=${version()})`);
