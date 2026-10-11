import { readFileSync } from 'node:fs';
import { beforeAll, describe as suite, expect, it, vi } from 'vitest';
import { featureCatalog, featurize } from '../src/data/features';
import { inkFromRGBA, rgbaToMnist, toFashion } from '../src/data/preprocess';
import { decodeModel, encodeModel, type ModelFile } from '../src/models/format';
import { Network } from '../src/nn/network';
import type { Arch, LayerSpec } from '../src/nn/types';

/**
 * Regression tests for the third review: model files that are checked before they touch the page
 * (MOD-2, MOD-4), keep-weights edits that follow the layers rather than their positions (MOD-3),
 * feature labels that say what they compute (DATA-3) and uploads on a transparent background
 * (DATA-7).
 */

const json = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

const mnistArch = (layers: LayerSpec[]): Arch => ({ input: { c: 1, h: 28, w: 28 }, layers, classes: 10 });
const pointArch = (layers: LayerSpec[], features = 2): Arch => ({ input: { c: features, h: 1, w: 1 }, layers, classes: 2 });

/** A valid saved MNIST model (one dense layer) and a valid Circle model. */
const goodImage = (): ModelFile => json(encodeModel(new Network(mnistArch([{ kind: 'dense', units: 4, act: 'relu' }]), 1), { name: 'Image model', dataset: 'mnist', frozen: [false, false], meta: { testAccuracy: 0.5 } }));
const goodPoints = (): ModelFile =>
  json(encodeModel(new Network(pointArch([{ kind: 'dense', units: 4, act: 'tanh' }]), 1), { name: 'Point model', dataset: 'circle', features: ['x1', 'x2'], frozen: [false, false], meta: {} }));

suite('MOD-2 / MOD-4: a model file is checked before it touches the page', () => {
  it('accepts the files the page writes', () => {
    expect(decodeModel(goodImage()).file.name).toBe('Image model');
    expect(decodeModel(goodPoints()).file.features).toEqual(['x1', 'x2']);
    // Files without the optional fields (as older versions and the zoo write them).
    const bare = goodImage() as Partial<ModelFile>;
    delete bare.frozen;
    expect(() => decodeModel(bare)).not.toThrow();
  });

  it('refuses a dataset this page does not have, and names it', () => {
    expect(() => decodeModel({ ...goodImage(), dataset: 'imagenet' })).toThrow(/dataset "imagenet".*this page does not have/);
    expect(() => decodeModel({ ...goodImage(), dataset: 42 })).toThrow(/does not say which dataset/);
  });

  it('refuses an architecture that does not fit its dataset', () => {
    const f = goodImage();
    expect(() => decodeModel({ ...f, dataset: 'cifar10' })).toThrow('The model takes 28×28 grey inputs, but CIFAR-10 has 32×32 colour images.');
    expect(() => decodeModel({ ...f, dataset: 'circle' })).toThrow(/28×28 grey inputs, but Circle has/);
  });

  it('refuses a frozen list that is not one true or false per layer', () => {
    expect(() => decodeModel({ ...goodImage(), frozen: 'yes' })).toThrow('The model file’s frozen list should hold one true or false per layer (2 layers).');
    expect(() => decodeModel({ ...goodImage(), frozen: [true] })).toThrow(/one true or false per layer \(2 layers\)/);
    expect(() => decodeModel({ ...goodImage(), frozen: [1, 0] })).toThrow(/one true or false per layer/);
  });

  it('refuses damaged weight text in words, not with a browser error', () => {
    const f = goodImage();
    expect(() => decodeModel({ ...f, weights: [f.weights[0], '***not base64***', ...f.weights.slice(2)] })).toThrow('Weight array 2 of the model file is damaged: it is not base64 text.');
    expect(() => decodeModel({ ...f, weights: [f.weights[0], 12, ...f.weights.slice(2)] })).toThrow('Weight array 2 of the model file is damaged: it is not base64 text.');
    expect(() => decodeModel({ ...f, weights: [f.weights[0], 'AAA=', ...f.weights.slice(2)] })).toThrow(/Weight array 2 .* not a whole number of float32 values/);
  });

  it('refuses unknown, repeated or missing input features of a point model', () => {
    const f = goodPoints();
    expect(() => decodeModel({ ...f, features: ['x9', 'x10'] })).toThrow('The model file asks for input features this page does not have for 2-D points: x9, x10.');
    expect(() => decodeModel({ ...f, features: ['x1', 'x1'] })).toThrow(/lists the feature x1 twice/);
    expect(() => decodeModel({ ...f, features: ['x1'] })).toThrow('The model takes 2 input features, but the file lists 1 (x₁).');
    const none = { ...f } as Partial<ModelFile>;
    delete none.features;
    expect(() => decodeModel(none)).toThrow(/does not say which input features/);
    expect(() => decodeModel({ ...goodImage(), features: ['x1'] })).toThrow(/lists input features, but MNIST digits is an image dataset/);
  });

  it('refuses a name or meta data of the wrong kind', () => {
    expect(() => decodeModel({ ...goodImage(), name: 42 })).toThrow('The model file has no name.');
    expect(() => decodeModel({ ...goodImage(), name: '   ' })).toThrow('The model file has no name.');
    expect(() => decodeModel({ ...goodImage(), meta: { testAccuracy: 'high' } })).toThrow(/test accuracy .* a number between 0 and 1/);
    expect(() => decodeModel({ ...goodImage(), meta: { testAccuracy: 1.5 } })).toThrow(/test accuracy .* a number between 0 and 1/);
    expect(() => decodeModel({ ...goodImage(), meta: 'none' })).toThrow(/meta/);
  });

  suite('loading a damaged file leaves the page as it was', () => {
    type Actions = typeof import('../src/actions');
    type Store = typeof import('../src/store')['store'];
    let actions: Actions;
    let store: Store;
    beforeAll(async () => {
      vi.stubGlobal('requestAnimationFrame', (fn: () => void) => setTimeout(fn, 0));
      actions = await import('../src/actions');
      store = (await import('../src/store')).store;
    });

    it('unknown dataset, bad frozen list, unknown features: nothing changes', async () => {
      const before = { dataset: store.dataset, version: store.version, frozen: store.frozen.slice(), features: store.features.slice() };
      for (const bad of [{ ...goodImage(), dataset: 'imagenet' }, { ...goodImage(), frozen: 'yes' }, { ...goodPoints(), features: ['x9', 'x10'] }]) {
        await expect(actions.loadModel(bad)).rejects.toThrow();
        expect(() => actions.transferModel(bad)).toThrow();
      }
      expect({ dataset: store.dataset, version: store.version, frozen: store.frozen.slice(), features: store.features.slice() }).toEqual(before);
      expect(store.info.id).toBe(before.dataset);
    });
  });
});

suite('MOD-3: keep-weights follows the layers, not their positions', () => {
  const dense = (units: number, act: LayerSpec['act'] = 'tanh'): LayerSpec => ({ kind: 'dense', units, act });
  const same = (a: Float32Array, b: Float32Array) => a.length === b.length && a.every((v, i) => v === b[i]);

  it('removing the first hidden layer: the next one starts fresh (its input changed), the output keeps its weights', () => {
    const A = dense(8);
    const B = dense(8);
    const old = new Network(pointArch([A, B]), 1);
    // The builder's remove keeps the other layers' spec objects.
    const next = new Network(pointArch([B]), 2);
    const fresh = next.getWeights();
    expect(next.copyCompatible(old)).toEqual([false, true]);
    expect(same(next.blocks[0].W, fresh[0])).toBe(true); // not the removed layer's weights
    expect(same(next.blocks[0].W, old.blocks[0].W)).toBe(false);
    expect(same(next.output.W, old.output.W)).toBe(true);
    expect(same(next.output.b, old.output.b)).toBe(true);
  });

  it('removing the second hidden layer keeps the first and the output', () => {
    const A = dense(8);
    const B = dense(8);
    const old = new Network(pointArch([A, B]), 1);
    const next = new Network(pointArch([A]), 2);
    expect(next.copyCompatible(old)).toEqual([true, true]);
    expect(same(next.blocks[0].W, old.blocks[0].W)).toBe(true);
    expect(same(next.output.W, old.output.W)).toBe(true);
  });

  it('inserting a layer keeps the layers around it that still fit', () => {
    const A = dense(8);
    const B = dense(8);
    const old = new Network(pointArch([A, B]), 1);
    const next = new Network(pointArch([A, dense(8), B]), 2);
    expect(next.copyCompatible(old)).toEqual([true, false, true, true]);
    expect(same(next.blocks[2].W, old.blocks[1].W)).toBe(true);
    expect(same(next.output.W, old.output.W)).toBe(true);
  });

  it('a conv layer inserted in front: the dense layer above it and the output keep their weights', () => {
    const C = { kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: true } as const;
    const D = dense(16, 'relu');
    const old = new Network(mnistArch([C, D]), 1);
    // "+ Conv layer" inserts after the last conv: [C, new conv without pooling, D].
    const next = new Network(mnistArch([C, { kind: 'conv', filters: 4, kernel: 3, act: 'relu', pool: false }, D]), 2);
    expect(next.copyCompatible(old)).toEqual([true, false, true, true]);
  });

  it('a changed setting keeps the weights when the shapes stay; copies of a spec match by position', () => {
    const A = dense(8);
    const B = dense(8);
    const old = new Network(pointArch([A, B]), 1);
    expect(new Network(pointArch([{ ...A, act: 'relu' }, B]), 2).copyCompatible(old)).toEqual([true, true, true]);
    expect(new Network(pointArch(structuredClone([A, B])), 2).copyCompatible(old)).toEqual([true, true, true]);
    expect(new Network(pointArch([A, dense(4)]), 2).copyCompatible(old)).toEqual([true, false, false]);
  });
});

suite('DATA-3: feature labels say what they compute', () => {
  it('the sine features are labelled sin πxᵢ, and compute exactly that', () => {
    const cat = featureCatalog(3);
    const sines = cat.filter((f) => f.id.startsWith('sin'));
    expect(sines.map((f) => f.label)).toEqual(['sin πx₁', 'sin πx₂', 'sin πx₃']);
    const x = featurize(Float32Array.from([0.19, -0.5, 1]), 3, sines.map((f) => f.id));
    expect(x[0]).toBeCloseTo(Math.sin(Math.PI * 0.19), 6);
    expect(x[0]).toBeCloseTo(0.565, 3);
    expect(x[1]).toBeCloseTo(-1, 6);
  });
});

suite('DATA-7: uploads on a transparent background', () => {
  const W = 60;
  const H = 60;
  /** A bar-shaped "digit" (or item) of colour `fg` on `bg`; both are [r, g, b, a]. */
  const image = (fg: number[], bg: number[]) => {
    const d = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const on = (x >= 22 && x < 34 && y >= 8 && y < 52) || (x >= 14 && x < 42 && y >= 40 && y < 50);
        d.set(on ? fg : bg, 4 * (y * W + x));
      }
    }
    return d;
  };
  const CLEAR_BLACK = [0, 0, 0, 0];
  const CLEAR_WHITE = [255, 255, 255, 0];
  const close = (a: Float32Array | null, b: Float32Array | null) => {
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(Math.max(...a!.map((v, i) => Math.abs(v - b![i])))).toBeLessThan(1e-6);
  };

  it('MNIST: white ink on a transparent background reads like white ink on black', () => {
    const onBlack = rgbaToMnist(image([255, 255, 255, 255], [0, 0, 0, 255]), W, H);
    for (const clear of [CLEAR_BLACK, CLEAR_WHITE]) close(rgbaToMnist(image([255, 255, 255, 255], clear), W, H), onBlack);
    // Light grey ink too.
    close(rgbaToMnist(image([230, 230, 230, 255], CLEAR_BLACK), W, H), onBlack);
  });

  it('MNIST: dark ink on a transparent background still reads exactly like dark ink on white', () => {
    const onWhite = rgbaToMnist(image([20, 20, 20, 255], [255, 255, 255, 255]), W, H);
    close(rgbaToMnist(image([20, 20, 20, 255], CLEAR_BLACK), W, H), onWhite);
    expect(inkFromRGBA(image([0, 0, 0, 255], CLEAR_WHITE), W, H)).toEqual(inkFromRGBA(image([0, 0, 0, 255], [255, 255, 255, 255]), W, H));
  });

  it('Fashion-MNIST: a light item cut out on a transparent background is found and framed like one on black', () => {
    const onBlack = toFashion({ data: image([240, 240, 240, 255], [0, 0, 0, 255]), width: W, height: H });
    for (const clear of [CLEAR_BLACK, CLEAR_WHITE]) {
      const r = toFashion({ data: image([240, 240, 240, 255], clear), width: W, height: H });
      expect(r.box).toEqual(onBlack.box);
      close(r.x, onBlack.x);
    }
    // A white item as well.
    expect(toFashion({ data: image([255, 255, 255, 255], CLEAR_BLACK), width: W, height: H }).box).toEqual({ x0: 14, y0: 8, x1: 42, y1: 52 });
  });

  it('Fashion-MNIST: a dark item on a transparent background still reads like one on white', () => {
    const onWhite = toFashion({ data: image([40, 50, 60, 255], [255, 255, 255, 255]), width: W, height: H });
    const r = toFashion({ data: image([40, 50, 60, 255], CLEAR_BLACK), width: W, height: H });
    expect(r.box).toEqual(onWhite.box);
    close(r.x, onWhite.x);
  });

  it('a few transparent corners on an opaque photo change nothing', () => {
    const d = image([20, 20, 20, 255], [255, 255, 255, 255]);
    const rounded = d.slice();
    for (const [x0, y0] of [[0, 0], [W - 3, 0], [0, H - 3], [W - 3, H - 3]]) {
      for (let y = y0; y < y0 + 3; y++) for (let x = x0; x < x0 + 3; x++) rounded.set(CLEAR_BLACK, 4 * (y * W + x));
    }
    close(rgbaToMnist(rounded, W, H), rgbaToMnist(d, W, H));
    expect(toFashion({ data: rounded, width: W, height: H }).box).toEqual(toFashion({ data: d, width: W, height: H }).box);
  });
});

suite('MOD-9 / DATA-8 / ANA-7: the documentation says what the code does', () => {
  const doc = readFileSync('docs/models.md', 'utf8');
  it('describes the control run as it is run: nothing copied, conv and dense layers start random', () => {
    const pretrain = readFileSync('scripts/pretrain.ts', 'utf8');
    expect(pretrain).toMatch(/randomConv: \{ label: 'random conv frozen', copy: \(\) => null/);
    expect(doc).not.toContain('keep their random starting weights instead of the copied ones');
    expect(doc).toMatch(/nothing is copied/);
  });
});
