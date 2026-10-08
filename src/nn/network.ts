import { activate, activateBackward } from './activations';
import { Rng } from './rng';
import { CLASSES, INPUT_SHAPE, size, type Act, type ConvSpec, type DenseSpec, type LayerSpec, type Shape } from './types';

/**
 * A tiny feed-forward engine: conv blocks (conv → activation → optional 2×2 max-pool)
 * followed by dense blocks, then a 10-way softmax output. One sample at a time, so every
 * intermediate tensor and gradient stays inspectable for the visualizations.
 */

const initStd = (act: Act, fanIn: number, fanOut: number) =>
  act === 'relu' || act === 'leaky' ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));

// The three matrix products behind a convolution. Each handles four filters per pass so every
// load from the long operand is reused four times, which roughly doubles throughput in V8.

/** out[f,i] += Σ_r W[f,r]·col[r,i] */
function convForward(W: Float32Array, col: Float32Array, out: Float32Array, F: number, R: number, HW: number): void {
  let f = 0;
  for (; f + 4 <= F; f += 4) {
    const o0 = f * HW, o1 = o0 + HW, o2 = o1 + HW, o3 = o2 + HW;
    for (let r = 0; r < R; r++) {
      const w0 = W[f * R + r], w1 = W[(f + 1) * R + r], w2 = W[(f + 2) * R + r], w3 = W[(f + 3) * R + r];
      const co = r * HW;
      for (let i = 0; i < HW; i++) {
        const v = col[co + i];
        out[o0 + i] += w0 * v;
        out[o1 + i] += w1 * v;
        out[o2 + i] += w2 * v;
        out[o3 + i] += w3 * v;
      }
    }
  }
  for (; f < F; f++) {
    const o = f * HW;
    for (let r = 0; r < R; r++) {
      const w = W[f * R + r];
      const co = r * HW;
      for (let i = 0; i < HW; i++) out[o + i] += w * col[co + i];
    }
  }
}

/** gW[f,r] += Σ_i dZ[f,i]·col[r,i] */
function convWeightGrad(dZ: Float32Array, col: Float32Array, gW: Float32Array, F: number, R: number, HW: number): void {
  let f = 0;
  for (; f + 4 <= F; f += 4) {
    const o0 = f * HW, o1 = o0 + HW, o2 = o1 + HW, o3 = o2 + HW;
    for (let r = 0; r < R; r++) {
      const co = r * HW;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < HW; i++) {
        const v = col[co + i];
        s0 += dZ[o0 + i] * v;
        s1 += dZ[o1 + i] * v;
        s2 += dZ[o2 + i] * v;
        s3 += dZ[o3 + i] * v;
      }
      gW[f * R + r] += s0;
      gW[(f + 1) * R + r] += s1;
      gW[(f + 2) * R + r] += s2;
      gW[(f + 3) * R + r] += s3;
    }
  }
  for (; f < F; f++) {
    const o = f * HW;
    for (let r = 0; r < R; r++) {
      const co = r * HW;
      let s = 0;
      for (let i = 0; i < HW; i++) s += dZ[o + i] * col[co + i];
      gW[f * R + r] += s;
    }
  }
}

/** dCol[r,i] += Σ_f W[f,r]·dZ[f,i] */
function convColGrad(W: Float32Array, dZ: Float32Array, dCol: Float32Array, F: number, R: number, HW: number): void {
  let f = 0;
  for (; f + 4 <= F; f += 4) {
    const o0 = f * HW, o1 = o0 + HW, o2 = o1 + HW, o3 = o2 + HW;
    for (let r = 0; r < R; r++) {
      const w0 = W[f * R + r], w1 = W[(f + 1) * R + r], w2 = W[(f + 2) * R + r], w3 = W[(f + 3) * R + r];
      const co = r * HW;
      for (let i = 0; i < HW; i++) dCol[co + i] += w0 * dZ[o0 + i] + w1 * dZ[o1 + i] + w2 * dZ[o2 + i] + w3 * dZ[o3 + i];
    }
  }
  for (; f < F; f++) {
    const o = f * HW;
    for (let r = 0; r < R; r++) {
      const w = W[f * R + r];
      const co = r * HW;
      for (let i = 0; i < HW; i++) dCol[co + i] += w * dZ[o + i];
    }
  }
}

export class ConvBlock {
  readonly kind = 'conv' as const;
  readonly k: number;
  readonly pad: number;
  readonly zShape: Shape;
  readonly outShape: Shape;
  W: Float32Array;
  b: Float32Array;
  gW: Float32Array;
  gb: Float32Array;
  x: Float32Array;
  z: Float32Array;
  a: Float32Array;
  out: Float32Array;
  argmax: Int32Array | null;
  dOut: Float32Array;
  dA: Float32Array;
  dZ: Float32Array;
  dX: Float32Array;
  /** Rows of the unrolled patch matrix: in-channels × k × k. */
  readonly R: number;
  private col: Float32Array;
  private dCol: Float32Array;

  constructor(
    readonly spec: ConvSpec,
    readonly inShape: Shape,
    rng: Rng,
  ) {
    const k = (this.k = spec.kernel);
    this.pad = (k - 1) >> 1;
    const F = spec.filters;
    const C = inShape.c;
    this.zShape = { c: F, h: inShape.h, w: inShape.w };
    this.outShape = spec.pool ? { c: F, h: inShape.h >> 1, w: inShape.w >> 1 } : this.zShape;
    this.W = new Float32Array(F * C * k * k);
    const std = initStd(spec.act, C * k * k, F * k * k);
    for (let i = 0; i < this.W.length; i++) this.W[i] = rng.normal() * std;
    this.b = new Float32Array(F);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(F);
    this.x = new Float32Array(size(inShape));
    this.z = new Float32Array(size(this.zShape));
    this.a = new Float32Array(this.z.length);
    this.out = spec.pool ? new Float32Array(size(this.outShape)) : this.a;
    this.argmax = spec.pool ? new Int32Array(this.out.length) : null;
    this.dOut = new Float32Array(this.out.length);
    this.dA = spec.pool ? new Float32Array(this.a.length) : this.dOut;
    this.dZ = new Float32Array(this.z.length);
    this.dX = new Float32Array(this.x.length);
    this.R = C * k * k;
    this.col = new Float32Array(this.R * inShape.h * inShape.w);
    this.dCol = new Float32Array(this.col.length);
  }

  /** Unrolls every k×k input patch into a column (zero-padded), so the conv becomes a matrix product. */
  private im2col(x: Float32Array): void {
    const { c: C, h: H, w: Wd } = this.inShape;
    const k = this.k;
    const p = this.pad;
    const HW = H * Wd;
    const col = this.col;
    for (let c = 0; c < C; c++) {
      for (let ky = 0; ky < k; ky++) {
        const dy = ky - p;
        for (let kx = 0; kx < k; kx++) {
          const dx = kx - p;
          const r = ((c * k + ky) * k + kx) * HW;
          for (let y = 0; y < H; y++) {
            const sy = y + dy;
            const ro = r + y * Wd;
            if (sy < 0 || sy >= H) {
              col.fill(0, ro, ro + Wd);
              continue;
            }
            const so = c * HW + sy * Wd + dx;
            for (let xx = 0; xx < Wd; xx++) {
              const sx = xx + dx;
              col[ro + xx] = sx >= 0 && sx < Wd ? x[so + xx] : 0;
            }
          }
        }
      }
    }
  }

  forward(x: Float32Array): Float32Array {
    this.x = x;
    this.im2col(x);
    const HW = this.zShape.h * this.zShape.w;
    const F = this.spec.filters;
    const R = this.R;
    const z = this.z;
    for (let f = 0; f < F; f++) z.fill(this.b[f], f * HW, (f + 1) * HW);
    convForward(this.W, this.col, z, F, R, HW);
    activate(this.spec.act, z, this.a);
    if (this.spec.pool) this.poolForward();
    return this.out;
  }

  private poolForward(): void {
    const { c: F, h: H, w: Wd } = this.zShape;
    const { h: PH, w: PW } = this.outShape;
    const a = this.a;
    const out = this.out;
    const am = this.argmax!;
    for (let f = 0; f < F; f++) {
      for (let py = 0; py < PH; py++) {
        for (let px = 0; px < PW; px++) {
          const i0 = f * H * Wd + 2 * py * Wd + 2 * px;
          let best = i0;
          if (a[i0 + 1] > a[best]) best = i0 + 1;
          if (a[i0 + Wd] > a[best]) best = i0 + Wd;
          if (a[i0 + Wd + 1] > a[best]) best = i0 + Wd + 1;
          const o = (f * PH + py) * PW + px;
          out[o] = a[best];
          am[o] = best;
        }
      }
    }
  }

  /** Reads this.dOut (∂L/∂out); accumulates gW, gb; writes dX when needDx. */
  backward(needDx: boolean): void {
    if (this.spec.pool) {
      const dA = this.dA;
      dA.fill(0);
      const am = this.argmax!;
      for (let i = 0; i < am.length; i++) dA[am[i]] += this.dOut[i];
    }
    activateBackward(this.spec.act, this.z, this.a, this.dA, this.dZ);

    const HW = this.zShape.h * this.zShape.w;
    const F = this.spec.filters;
    const R = this.R;
    const dZ = this.dZ;
    for (let f = 0; f < F; f++) {
      let sb = 0;
      for (let i = f * HW; i < (f + 1) * HW; i++) sb += dZ[i];
      this.gb[f] += sb;
    }
    convWeightGrad(dZ, this.col, this.gW, F, R, HW);
    if (!needDx) return;

    // ∂L/∂col = Wᵀ·dZ, then fold the columns back onto the input (col2im).
    const dCol = this.dCol;
    dCol.fill(0);
    convColGrad(this.W, dZ, dCol, F, R, HW);
    const { c: C, h: H, w: Wd } = this.inShape;
    const k = this.k;
    const p = this.pad;
    const dX = this.dX;
    dX.fill(0);
    for (let c = 0; c < C; c++) {
      for (let ky = 0; ky < k; ky++) {
        const dy = ky - p;
        const y0 = Math.max(0, -dy);
        const y1 = Math.min(H, H - dy);
        for (let kx = 0; kx < k; kx++) {
          const dx = kx - p;
          const x0 = Math.max(0, -dx);
          const x1 = Math.min(Wd, Wd - dx);
          const r = ((c * k + ky) * k + kx) * HW;
          for (let y = y0; y < y1; y++) {
            const ro = r + y * Wd;
            const so = c * HW + (y + dy) * Wd + dx;
            for (let xx = x0; xx < x1; xx++) dX[so + xx] += dCol[ro + xx];
          }
        }
      }
    }
  }
}

export class DenseBlock {
  readonly kind = 'dense' as const;
  readonly inSize: number;
  readonly outShape: Shape;
  W: Float32Array;
  b: Float32Array;
  gW: Float32Array;
  gb: Float32Array;
  x: Float32Array;
  z: Float32Array;
  a: Float32Array;
  out: Float32Array;
  dOut: Float32Array;
  dA: Float32Array;
  dZ: Float32Array;
  dX: Float32Array;

  constructor(
    readonly spec: DenseSpec,
    readonly inShape: Shape,
    readonly isOutput: boolean,
    rng: Rng,
  ) {
    const n = (this.inSize = size(inShape));
    const m = spec.units;
    this.outShape = { c: m, h: 1, w: 1 };
    this.W = new Float32Array(m * n);
    const std = initStd(spec.act, n, m);
    for (let i = 0; i < this.W.length; i++) this.W[i] = rng.normal() * std;
    this.b = new Float32Array(m);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(m);
    this.x = new Float32Array(n);
    this.z = new Float32Array(m);
    this.a = new Float32Array(m);
    this.out = this.a;
    this.dOut = new Float32Array(m);
    this.dA = this.dOut;
    this.dZ = new Float32Array(m);
    this.dX = new Float32Array(n);
  }

  forward(x: Float32Array): Float32Array {
    this.x = x;
    const n = this.inSize;
    const m = this.spec.units;
    const W = this.W;
    for (let j = 0; j < m; j++) {
      let s = this.b[j];
      const o = j * n;
      for (let i = 0; i < n; i++) s += W[o + i] * x[i];
      this.z[j] = s;
    }
    activate(this.spec.act, this.z, this.a);
    return this.out;
  }

  backward(needDx: boolean): void {
    activateBackward(this.spec.act, this.z, this.a, this.dA, this.dZ);
    const n = this.inSize;
    const m = this.spec.units;
    const W = this.W;
    const gW = this.gW;
    const x = this.x;
    const dX = this.dX;
    if (needDx) dX.fill(0);
    for (let j = 0; j < m; j++) {
      const g = this.dZ[j];
      this.gb[j] += g;
      if (g === 0) continue;
      const o = j * n;
      for (let i = 0; i < n; i++) gW[o + i] += g * x[i];
      if (needDx) for (let i = 0; i < n; i++) dX[i] += W[o + i] * g;
    }
  }
}

export type Block = ConvBlock | DenseBlock;

export interface LayerInfo {
  spec: LayerSpec | null; // null = output layer
  inShape: Shape;
  outShape: Shape;
  params: number;
  error?: string;
}

/** Shape bookkeeping for the builder, without allocating any weights. */
export function describe(spec: LayerSpec[]): LayerInfo[] {
  const out: LayerInfo[] = [];
  let shape: Shape = INPUT_SHAPE;
  let sawDense = false;
  for (const l of spec) {
    if (l.kind === 'conv') {
      let error: string | undefined;
      if (sawDense) error = 'Convolutions must come before dense layers.';
      const z = { c: l.filters, h: shape.h, w: shape.w };
      let o = z;
      if (l.pool) {
        if (shape.h < 2) error = 'Too small to pool. Turn pooling off or remove a pooling layer.';
        o = { c: l.filters, h: shape.h >> 1, w: shape.w >> 1 };
      }
      out.push({ spec: l, inShape: shape, outShape: o, params: l.filters * shape.c * l.kernel * l.kernel + l.filters, error });
      shape = o;
    } else {
      sawDense = true;
      const n = size(shape);
      const o = { c: l.units, h: 1, w: 1 };
      out.push({ spec: l, inShape: shape, outShape: o, params: n * l.units + l.units });
      shape = o;
    }
  }
  out.push({ spec: null, inShape: shape, outShape: { c: CLASSES, h: 1, w: 1 }, params: size(shape) * CLASSES + CLASSES });
  return out;
}

export const OUTPUT_SPEC: DenseSpec = { kind: 'dense', units: CLASSES, act: 'linear' };

export class Network {
  readonly blocks: Block[] = [];
  readonly probs = new Float32Array(CLASSES);
  input: Float32Array = new Float32Array(size(INPUT_SHAPE));

  constructor(
    readonly spec: LayerSpec[],
    seed: number,
  ) {
    const rng = new Rng(seed);
    let shape: Shape = INPUT_SHAPE;
    for (const l of spec) {
      const b = l.kind === 'conv' ? new ConvBlock(l, shape, rng) : new DenseBlock(l, shape, false, rng);
      this.blocks.push(b);
      shape = b.outShape;
    }
    this.blocks.push(new DenseBlock(OUTPUT_SPEC, shape, true, rng));
  }

  get output(): DenseBlock {
    return this.blocks[this.blocks.length - 1] as DenseBlock;
  }

  /** Runs the forward pass and returns softmax probabilities. Intermediates stay on each block. */
  forward(x: Float32Array): Float32Array {
    this.input = x;
    let h = x;
    for (const b of this.blocks) h = b.forward(h);
    softmax(h, this.probs);
    return this.probs;
  }

  /** Cross-entropy loss of the last forward pass against `label`. */
  loss(label: number): number {
    return -Math.log(Math.max(this.probs[label], 1e-12));
  }

  /** Backpropagates the cross-entropy loss of the last forward pass; gradients accumulate. */
  backward(label: number, needInputGrad = false): number {
    const out = this.output;
    for (let j = 0; j < CLASSES; j++) out.dOut[j] = this.probs[j] - (j === label ? 1 : 0);
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const b = this.blocks[i];
      const needDx = i > 0 || needInputGrad;
      b.backward(needDx);
      if (i > 0) this.blocks[i - 1].dOut.set(b.dX);
    }
    return this.loss(label);
  }

  zeroGrad(): void {
    for (const b of this.blocks) {
      b.gW.fill(0);
      b.gb.fill(0);
    }
  }

  get paramCount(): number {
    let n = 0;
    for (const b of this.blocks) n += b.W.length + b.b.length;
    return n;
  }

  /** Flat list [W0, b0, W1, b1, …] of copies. */
  getWeights(): Float32Array[] {
    const out: Float32Array[] = [];
    for (const b of this.blocks) out.push(b.W.slice(), b.b.slice());
    return out;
  }

  setWeights(ws: Float32Array[]): void {
    this.blocks.forEach((b, i) => {
      b.W.set(ws[2 * i]);
      b.b.set(ws[2 * i + 1]);
    });
  }

  predict(x: Float32Array): number {
    const p = this.forward(x);
    let best = 0;
    for (let j = 1; j < CLASSES; j++) if (p[j] > p[best]) best = j;
    return best;
  }
}

export function softmax(z: Float32Array, out: Float32Array): void {
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
  let s = 0;
  for (let i = 0; i < z.length; i++) {
    out[i] = Math.exp(z[i] - m);
    s += out[i];
  }
  for (let i = 0; i < z.length; i++) out[i] /= s;
}

export const argmax = (a: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
};
