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

  it('ships a 2 × 2 preview mosaic of real samples with every image dataset', () => {
    for (const d of DATASETS.filter((x) => x.kind === 'image')) {
      const m = /^data:image\/(png|jpeg);base64,(.+)$/.exec(d.preview ?? '');
      expect(m, d.id).not.toBeNull();
      const bytes = Buffer.from(m![2], 'base64');
      const { c, h, w } = d.image!.shape;
      let width = 0;
      let height = 0;
      let channels = 0;
      if (m![1] === 'png') {
        expect(bytes.subarray(1, 4).toString()).toBe('PNG');
        width = bytes.readUInt32BE(16);
        height = bytes.readUInt32BE(20);
        channels = bytes[25] === 0 ? 1 : 3; // colour type 0: grey
      } else {
        // The first start-of-frame marker holds the size and the number of components.
        let at = 2;
        while (bytes[at + 1] !== 0xc0 && bytes[at + 1] !== 0xc2) at += 2 + bytes.readUInt16BE(at + 2);
        height = bytes.readUInt16BE(at + 5);
        width = bytes.readUInt16BE(at + 7);
        channels = bytes[at + 9];
      }
      expect([width, height, channels], d.id).toEqual([2 * w, 2 * h, c]);
      // A few kilobytes at most: it is part of the page's script.
      expect(d.preview!.length, d.id).toBeLessThan(6_000);
    }
    // Point datasets draw theirs from the generator.
    for (const d of DATASETS.filter((x) => x.kind === 'points')) expect(d.preview).toBeUndefined();
  });

  it('names the groups with one spelling of 2-D and 3-D', () => {
    expect([...new Set(DATASETS.map((d) => d.group))]).toEqual(['Images', 'Points in 2-D', 'Points in 3-D']);
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
