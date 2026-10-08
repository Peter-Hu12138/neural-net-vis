/// <reference lib="webworker" />
import { Analyzer } from './analyzer';
import type { FromAnalyzer, ToAnalyzer } from './protocol';
import { registry } from './registry';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const analyzer = new Analyzer((msg: FromAnalyzer) => scope.postMessage(msg), registry);
scope.onmessage = (e: MessageEvent<ToAnalyzer>) => analyzer.handle(e.data);
scope.postMessage({ type: 'ready' } satisfies FromAnalyzer);
