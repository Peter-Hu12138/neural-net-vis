import { beforeAll, describe, expect, it, vi } from 'vitest';
import { DATASETS, datasetInfo } from '../src/data/datasets';
import { generate, SYNTHETIC } from '../src/data/synthetic';

/**
 * The dataset strip's pure helpers. The module also builds UI, and the modules it imports touch a
 * few browser globals when they load, so those are stubbed before it is imported.
 */
type Picker = typeof import('../src/ui/datasetPicker');
let picker: Picker;

beforeAll(async () => {
  vi.stubGlobal('matchMedia', () => ({ matches: false, addEventListener() {} }));
  vi.stubGlobal(
    'MutationObserver',
    class {
      observe() {}
    },
  );
  vi.stubGlobal('document', { documentElement: {} });
  picker = await import('../src/ui/datasetPicker');
});

describe('dataset strip helpers', () => {
  it('predicts the training-set size that the point generator makes', () => {
    for (const s of SYNTHETIC) {
      for (const count of [200, 400, 600, 1_000, 2_000]) {
        for (const trainRatio of [0.1, 0.3, 0.5, 0.9]) {
          const syn = generate({ id: s.id, count, noise: 0.1, trainRatio, seed: 3 });
          expect(picker.pointTrainCount(s.classes, count, trainRatio), `${s.id} ${count} ${trainRatio}`).toBe(syn.train.labels.length);
        }
      }
    }
  });

  it('offers image-sized subsets for images and point-sized subsets for points', () => {
    expect(picker.limitsFor(datasetInfo('cifar10'))).toEqual([null, 5_000, 1_000, 200]);
    expect(picker.limitsFor(datasetInfo('helix'))).toEqual([null, 100, 50, 20]);
    // Every image subset is smaller than every image training set, so each choice changes something.
    for (const d of DATASETS.filter((x) => x.kind === 'image')) {
      for (const v of picker.limitsFor(d)) if (v !== null) expect(v).toBeLessThan(d.image!.train);
    }
  });

  it('projects 3-D points with the vertical axis up', () => {
    const [u0, v0] = picker.project3(0, 0, 0);
    expect(u0).toBeCloseTo(0);
    expect(v0).toBeCloseTo(0);
    const [, up] = picker.project3(0, 0, 1);
    const [, down] = picker.project3(0, 0, -1);
    expect(up).toBeGreaterThan(0.9);
    expect(down).toBeLessThan(-0.9);
    // Points along x₁ and x₂ spread sideways in opposite directions, so neither axis hides the other.
    expect(Math.sign(picker.project3(1, 0, 0)[0])).not.toBe(Math.sign(picker.project3(0, 1, 0)[0]));
  });
});
