import { describe, expect, it } from 'vitest';
import { Analyzer } from '../src/analysis/analyzer';
import type { FromAnalyzer, Job } from '../src/analysis/protocol';
import { Network } from '../src/nn/network';

const spec = [{ kind: 'dense' as const, units: 4, act: 'relu' as const }];
const weights = new Network(spec, 1).getWeights();

/** Counts to `n`, one step per yield, and reports which test labels it saw. */
const count: Job<{ n: number }, { n: number; firstLabel: number; pixel: number }> = function* (ctx, p) {
  for (let i = 0; i < p.n; i++) yield { done: i, total: p.n };
  return { n: p.n, firstLabel: ctx.testY[0], pixel: ctx.image(0)[0] };
};
const boom: Job = function* () {
  yield { done: 0, total: 1 };
  throw new Error('kaput');
};

function harness() {
  const log: FromAnalyzer[] = [];
  const a = new Analyzer((m) => log.push(m), { count, boom }, 5);
  const run = (id: number, channel: string, kind: string, params: unknown) =>
    a.handle({ type: 'run', id, channel, kind, params, spec, weights });
  const until = async (pred: () => boolean) => {
    for (let t = 0; t < 2000 && !pred(); t++) await new Promise((r) => setTimeout(r, 2));
    expect(pred()).toBe(true);
  };
  return { a, log, run, until, results: () => log.filter((m) => m.type === 'result') as { id: number; result: any }[] };
}

const data = { testX: new Uint8Array(784 * 2).fill(255), testY: new Uint8Array([7, 3]) };

describe('Analyzer', () => {
  it('waits for data, then runs the job to completion', async () => {
    const h = harness();
    h.run(1, 'a', 'count', { n: 50 });
    await new Promise((r) => setTimeout(r, 20));
    expect(h.results()).toHaveLength(0);
    h.a.handle({ type: 'data', ...data });
    await h.until(() => h.results().length === 1);
    expect(h.results()[0]).toMatchObject({ id: 1, result: { n: 50, firstLabel: 7, pixel: 1 } });
  });

  it('a new request on a channel replaces the running one; other channels continue', async () => {
    const h = harness();
    h.a.handle({ type: 'data', ...data });
    h.run(1, 'a', 'count', { n: 100000 });
    h.run(2, 'b', 'count', { n: 300 });
    h.run(3, 'a', 'count', { n: 10 });
    await h.until(() => h.results().length === 2);
    expect(h.results().map((r) => r.id).sort()).toEqual([2, 3]);
  });

  it('cancel stops a job without a result', async () => {
    const h = harness();
    h.a.handle({ type: 'data', ...data });
    h.run(1, 'a', 'count', { n: 1e9 });
    await new Promise((r) => setTimeout(r, 20));
    h.a.handle({ type: 'cancel', channel: 'a' });
    h.run(2, 'b', 'count', { n: 5 });
    await h.until(() => h.results().length === 1);
    expect(h.results()[0].id).toBe(2);
    expect(h.log.some((m) => m.type === 'progress' && m.id === 1)).toBe(true);
  });

  it('reports errors and unknown jobs', async () => {
    const h = harness();
    h.a.handle({ type: 'data', ...data });
    h.run(1, 'a', 'boom', {});
    h.run(2, 'b', 'nope', {});
    await h.until(() => h.log.filter((m) => m.type === 'error').length === 2);
    const errs = h.log.filter((m) => m.type === 'error') as { id: number; message: string }[];
    expect(errs.find((e) => e.id === 1)?.message).toBe('kaput');
    expect(errs.find((e) => e.id === 2)?.message).toContain('Unknown analysis');
  });
});
