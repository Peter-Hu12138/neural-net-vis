/// <reference lib="webworker" />
import type { FromTrainer, ToTrainer } from './protocol';
import { Trainer } from './trainer';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const trainer = new Trainer((msg: FromTrainer, transfer?: Transferable[]) => scope.postMessage(msg, transfer ?? []));
scope.onmessage = (e: MessageEvent<ToTrainer>) => trainer.handle(e.data);
scope.postMessage({ type: 'ready' } satisfies FromTrainer);
