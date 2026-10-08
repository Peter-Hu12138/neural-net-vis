import { describe, expect, it } from 'vitest';
import {
  argmaxField,
  axisName,
  backToFront,
  boundarySegments,
  classSurfaces,
  confidence,
  marchingSquares,
  marginField,
  nearSide,
  orbit,
  paintRegions,
  project,
  resolutionFor,
  shadeAmount,
  surfaceNets,
  ticks,
  type RGB,
} from '../src/ui/boundaryMath';

/** Segment endpoints as [x, y] pairs. */
const points = (segs: number[]) => {
  const out: [number, number][] = [];
  for (let i = 0; i < segs.length; i += 2) out.push([segs[i], segs[i + 1]]);
  return out;
};

/** How often each endpoint occurs (rounded), to check that a contour closes. */
function endpointDegrees(segs: number[]): number[] {
  const m = new Map<string, number>();
  for (const [x, y] of points(segs)) {
    const k = `${x.toFixed(5)},${y.toFixed(5)}`;
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return [...m.values()];
}

describe('confidence shading', () => {
  it('maps 1/K to 0 and certainty to 1', () => {
    expect(confidence(0.5, 2)).toBe(0);
    expect(confidence(1, 2)).toBe(1);
    expect(confidence(1 / 3, 3)).toBeCloseTo(0, 12);
    expect(confidence(0.75, 2)).toBeCloseTo(0.5, 12);
    expect(confidence(0.2, 2)).toBe(0); // below chance clamps
  });

  it('paints ties pale, certain cells in the class colour, and flat tints when discrete', () => {
    const surface: RGB = [255, 255, 255];
    const cat: RGB[] = [[0, 0, 255], [255, 128, 0]];
    const probs = new Float32Array([0.5, 0.5, 1, 0, 0, 1, 0.2, 0.8]);
    const img = new Uint8ClampedArray(16);
    paintRegions(img, probs, 2, cat, surface, false);
    // A 50/50 cell is nearly the surface.
    expect(img[0]).toBeGreaterThan(240);
    expect(img[2]).toBe(255);
    // A certain class-0 cell moves toward blue: red and green drop, blue stays.
    const t = shadeAmount(1, false);
    expect(img[4]).toBeCloseTo(255 * (1 - t), -0.5);
    expect(img[6]).toBe(255);
    // A certain class-1 cell keeps red and loses blue.
    expect(img[8]).toBe(255);
    expect(img[10]).toBeCloseTo(255 * (1 - t), -0.5);
    // 0.8 is less certain than 1: paler.
    expect(img[14]).toBeGreaterThan(img[10]);
    expect(img.filter((_, i) => i % 4 === 3).every((a) => a === 255)).toBe(true);
    const flat = new Uint8ClampedArray(16);
    paintRegions(flat, probs, 2, cat, surface, true);
    expect(flat[10]).toBe(flat[14]); // discrete: confidence no longer matters
  });

  it('margins and argmax agree on the predicted class', () => {
    const probs = new Float32Array([0.6, 0.3, 0.1, 0.2, 0.3, 0.5]);
    expect(Array.from(argmaxField(probs, 3))).toEqual([0, 2]);
    const f0 = marginField(probs, 3, 0);
    expect(f0[0]).toBeCloseTo(0.3, 6);
    expect(f0[1]).toBeCloseTo(-0.3, 6);
  });
});

describe('marching squares', () => {
  it('finds a straight boundary where it is', () => {
    const cols = 6;
    const rows = 4;
    const f = new Float32Array(cols * rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) f[y * cols + x] = x - 2.5;
    const segs = marchingSquares(f, cols, rows);
    expect(segs.length / 4).toBe(rows - 1); // one segment per cell row
    for (const [x] of points(segs)) expect(x).toBeCloseTo(2.5, 12);
  });

  it('traces a circle as a closed curve at the right radius', () => {
    const n = 40;
    const R = 12;
    const f = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) f[y * n + x] = R - Math.hypot(x - 19.5, y - 19.3);
    const segs = marchingSquares(f, n, n);
    expect(segs.length).toBeGreaterThan(40);
    for (const [x, y] of points(segs)) expect(Math.abs(Math.hypot(x - 19.5, y - 19.3) - R)).toBeLessThan(0.05);
    // Closed: every endpoint is shared by exactly two segments.
    expect(endpointDegrees(segs).every((d) => d === 2)).toBe(true);
  });

  it('resolves saddles by the mean of the corners', () => {
    // tl and br above the level. Mean positive: they connect, so the other two corners are cut off.
    const joined = marchingSquares([2, -1, -1, 2], 2, 2);
    const cut = points(joined).map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`);
    // top-edge crossing joins the right-edge crossing (around the tr corner).
    expect(cut.slice(0, 2)).toEqual(['0.667,0.000', '1.000,0.333']);
    // Mean negative: each positive corner is its own island.
    const apart = points(marchingSquares([1, -2, -2, 1], 2, 2)).map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`);
    expect(apart.slice(0, 2)).toEqual(['0.000,0.333', '0.333,0.000']); // around the tl corner
    expect(marchingSquares([1, 1, 1, 1], 2, 2)).toEqual([]);
  });

  it('draws the edges between three class regions', () => {
    // Columns 0–3 class 0, 4–7 class 1, 8–11 class 2 (as probabilities).
    const cols = 12;
    const rows = 3;
    const probs = new Float32Array(cols * rows * 3);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const k = Math.floor(x / 4);
        for (let j = 0; j < 3; j++) probs[(y * cols + x) * 3 + j] = j === k ? 0.8 : 0.1;
      }
    }
    const xs = new Set(points(boundarySegments(probs, 3, cols, rows)).map(([x]) => x.toFixed(3)));
    expect([...xs].sort()).toEqual(['3.500', '7.500']);
    // Two classes use p₀ − p₁ directly.
    const two = new Float32Array(cols * 2);
    for (let x = 0; x < cols; x++) {
      two[2 * x] = x < 5 ? 0.9 : 0.1;
      two[2 * x + 1] = 1 - two[2 * x];
    }
    const segs = boundarySegments(two, 2, cols, 1);
    expect(segs).toEqual([]); // a single row has no cells
    const two2 = new Float32Array([...two, ...two]);
    const xs2 = points(boundarySegments(two2, 2, cols, 2)).map(([x]) => x);
    expect(xs2.every((x) => Math.abs(x - 4.5) < 1e-6)).toBe(true);
  });
});

describe('boundary surfaces in 3-D', () => {
  const sphere = (n: number, R: number) => {
    const f = new Float32Array(n * n * n);
    const c = (n - 1) / 2;
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) f[i + n * (j + n * k)] = R - Math.hypot(i - c, j - c, k - c);
    return { f, c };
  };

  it('surface nets put a sphere’s vertices near its radius and close it', () => {
    const n = 24;
    const R = 7.3;
    const { f, c } = sphere(n, R);
    const mesh = surfaceNets(f, n);
    const nv = mesh.verts.length / 3;
    expect(nv).toBeGreaterThan(200);
    expect(mesh.quads.length / 4).toBeGreaterThan(200);
    for (let v = 0; v < nv; v++) {
      const d = Math.hypot(mesh.verts[3 * v] - c, mesh.verts[3 * v + 1] - c, mesh.verts[3 * v + 2] - c);
      expect(Math.abs(d - R)).toBeLessThan(0.35);
    }
    // Closed and consistently oriented: every directed edge appears once, its reverse once.
    const edges = new Map<string, number>();
    for (let q = 0; q < mesh.quads.length; q += 4) {
      for (let t = 0; t < 4; t++) {
        const a = mesh.quads[q + t];
        const b = mesh.quads[q + ((t + 1) % 4)];
        edges.set(`${a}>${b}`, (edges.get(`${a}>${b}`) ?? 0) + 1);
      }
    }
    for (const [k, count] of edges) {
      expect(count).toBe(1);
      const [a, b] = k.split('>');
      expect(edges.get(`${b}>${a}`)).toBe(1);
    }
  });

  it('two-class probabilities give the p₀ = p₁ surface', () => {
    const n = 16;
    const R = 4.6;
    const { f, c } = sphere(n, R);
    const probs = new Float32Array(n * n * n * 2);
    for (let i = 0; i < f.length; i++) {
      const p0 = 1 / (1 + Math.exp(-3 * f[i]));
      probs[2 * i] = p0;
      probs[2 * i + 1] = 1 - p0;
    }
    const mesh = classSurfaces(probs, 2, n);
    expect(mesh.quads.length).toBeGreaterThan(0);
    for (let v = 0; v < mesh.verts.length / 3; v++) {
      const d = Math.hypot(mesh.verts[3 * v] - c, mesh.verts[3 * v + 1] - c, mesh.verts[3 * v + 2] - c);
      expect(Math.abs(d - R)).toBeLessThan(0.4);
    }
  });

  it('keeps each surface between two of several classes once', () => {
    // Slabs along x: class 0, 1, 2 → two planes, each n−2 × n−2 interior quads... found once.
    const n = 12;
    const probs = new Float32Array(n * n * n * 3);
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const cls = i < 4 ? 0 : i < 8 ? 1 : 2;
          for (let q = 0; q < 3; q++) probs[(i + n * (j + n * k)) * 3 + q] = q === cls ? 0.7 : 0.15;
        }
      }
    }
    const mesh = classSurfaces(probs, 3, n);
    // An x-edge crossing has four cells around it when j, k ∈ [1, n−2]: (n−2)² quads per plane.
    expect(mesh.quads.length / 4).toBe(2 * (n - 2) * (n - 2));
    const xs = new Set<string>();
    for (let v = 0; v < mesh.verts.length / 3; v++) xs.add(mesh.verts[3 * v].toFixed(3));
    expect([...xs].sort()).toEqual(['3.500', '7.500']);
  });
});

describe('orbit camera', () => {
  const dot = (m: number[], a: number, b: number) => m[3 * a] * m[3 * b] + m[3 * a + 1] * m[3 * b + 1] + m[3 * a + 2] * m[3 * b + 2];

  it('is a rotation for any angles', () => {
    for (const [yaw, pitch] of [[0, 0], [0.7, 0.3], [-2.1, -1.2], [3, 1.5]]) {
      const m = orbit(yaw, pitch);
      for (let a = 0; a < 3; a++) {
        expect(dot(m, a, a)).toBeCloseTo(1, 12);
        for (let b = a + 1; b < 3; b++) expect(dot(m, a, b)).toBeCloseTo(0, 12);
      }
      // Right-handed with depth into the screen: right × up points at the viewer (−depth).
      const r = m.slice(0, 3);
      const u = m.slice(3, 6);
      const cross = [r[1] * u[2] - r[2] * u[1], r[2] * u[0] - r[0] * u[2], r[0] * u[1] - r[1] * u[0]];
      cross.forEach((v, i) => expect(v).toBeCloseTo(-m[6 + i], 12));
    }
  });

  it('puts x₁ to the right and x₃ up when level, and x₃ toward the eye from above', () => {
    const level = orbit(0, 0);
    const p = project(level, 1, 0, 0, 100, 200, 150);
    expect(p[0]).toBeCloseTo(300, 9);
    expect(p[1]).toBeCloseTo(150, 9);
    const up = project(level, 0, 0, 1, 100, 200, 150);
    expect(up[1]).toBeCloseTo(50, 9); // screen y grows downward
    expect(project(level, 0, 1, 0, 100, 200, 150)[2]).toBeCloseTo(1, 9); // x₂ goes into the screen
    const top = orbit(0, Math.PI / 2);
    expect(project(top, 0, 0, 1, 1, 0, 0)[2]).toBeCloseTo(-1, 9); // looking down: higher is nearer
  });

  it('orders points back to front and knows which side of a slice faces the eye', () => {
    const m = orbit(0.4, 0.35);
    const pts = [[0, 0, 0], [0.5, 0.9, -0.2], [-0.4, -0.8, 0.6], [0.1, 0.3, 0.2]];
    const depth = pts.map((q) => project(m, q[0], q[1], q[2], 1, 0, 0)[2]);
    const order = Array.from(backToFront(depth));
    for (let i = 1; i < order.length; i++) expect(depth[order[i - 1]]).toBeGreaterThanOrEqual(depth[order[i]]);
    for (let axis = 0; axis < 3; axis++) {
      const s = nearSide(m, axis);
      const a = [0, 0, 0];
      const b = [0, 0, 0];
      a[axis] = 0.5 * s; // on the near side
      b[axis] = -0.5 * s;
      expect(project(m, a[0], a[1], a[2], 1, 0, 0)[2]).toBeLessThan(project(m, b[0], b[1], b[2], 1, 0, 0)[2]);
    }
  });
});

describe('budgets and labels', () => {
  it('picks the largest grid that fits the time budget', () => {
    expect(resolutionFor(0.001, 8, 2, 20, 100)).toBe(89); // √8000
    expect(resolutionFor(0.0001, 8, 2, 20, 100)).toBe(100);
    expect(resolutionFor(1, 8, 2, 20, 100)).toBe(20);
    expect(resolutionFor(0.001, 27, 3, 8, 40)).toBe(30);
    expect(resolutionFor(0, 8, 2, 20, 100)).toBe(100);
  });

  it('ticks and axis names', () => {
    expect(ticks(1.25)).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(ticks(2, 1)).toEqual([-2, -1, 0, 1, 2]);
    expect([0, 1, 2].map(axisName)).toEqual(['x₁', 'x₂', 'x₃']);
  });
});
