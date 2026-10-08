/**
 * Pure helpers behind the decision-boundary views: colouring class regions by confidence, contour
 * lines (marching squares), boundary surfaces in 3-D (surface nets), the orbit camera and
 * back-to-front ordering. No DOM, so the unit tests can run them directly.
 */

export type RGB = [number, number, number];

// ── Colour ───────────────────────────────────────────────────────────────

/**
 * How sure a prediction is, from 0 (every class equally likely, top probability 1/K) to 1 (one
 * class has all of it).
 */
export function confidence(pmax: number, classes: number): number {
  const floor = 1 / Math.max(2, classes);
  return Math.max(0, Math.min(1, (pmax - floor) / (1 - floor)));
}

/** How far a region's colour moves from the surface toward its class colour. */
export function shadeAmount(conf: number, discrete: boolean): number {
  return discrete ? 0.4 : 0.03 + 0.45 * Math.pow(conf, 0.9);
}

/**
 * Paints class regions into RGBA pixels: every cell takes the colour of its most likely class,
 * mixed from the surface by confidence (pale where the network hesitates), or a flat tint when
 * `discrete`. `probs` is n × classes; `img` has 4n bytes.
 */
export function paintRegions(img: Uint8ClampedArray, probs: ArrayLike<number>, classes: number, cat: RGB[], surface: RGB, discrete: boolean): void {
  const n = probs.length / classes;
  for (let i = 0; i < n; i++) {
    let best = 0;
    let pm = probs[i * classes];
    for (let k = 1; k < classes; k++) {
      const v = probs[i * classes + k];
      if (v > pm) {
        pm = v;
        best = k;
      }
    }
    const t = shadeAmount(confidence(pm, classes), discrete);
    const c = cat[best % cat.length];
    img[4 * i] = surface[0] + (c[0] - surface[0]) * t;
    img[4 * i + 1] = surface[1] + (c[1] - surface[1]) * t;
    img[4 * i + 2] = surface[2] + (c[2] - surface[2]) * t;
    img[4 * i + 3] = 255;
  }
}

/**
 * Class k's margin at every cell: p_k minus the largest other probability. Positive inside k's
 * region, zero on its boundary, so its zero contour is the edge of that region.
 */
export function marginField(probs: ArrayLike<number>, classes: number, k: number, out?: Float32Array): Float32Array {
  const n = probs.length / classes;
  const f = out ?? new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let other = -Infinity;
    for (let j = 0; j < classes; j++) if (j !== k && probs[i * classes + j] > other) other = probs[i * classes + j];
    f[i] = probs[i * classes + k] - other;
  }
  return f;
}

/** Most likely class at every cell. */
export function argmaxField(probs: ArrayLike<number>, classes: number): Uint8Array {
  const n = probs.length / classes;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let k = 1; k < classes; k++) if (probs[i * classes + k] > probs[i * classes + best]) best = k;
    out[i] = best;
  }
  return out;
}

// ── Contours in 2-D ──────────────────────────────────────────────────────

/**
 * Contour of a scalar field sampled on a cols × rows grid (row-major, row 0 first) at `level`,
 * as line segments [x0, y0, x1, y1, …] in grid units (x = column, y = row; sample (c, r) sits at
 * (c, r)). Crossings are placed by linear interpolation; ambiguous saddle cells are resolved by
 * the mean of their four corners.
 */
export function marchingSquares(f: ArrayLike<number>, cols: number, rows: number, level = 0): number[] {
  const out: number[] = [];
  for (let y = 0; y < rows - 1; y++) {
    for (let x = 0; x < cols - 1; x++) {
      const tl = f[y * cols + x] - level;
      const tr = f[y * cols + x + 1] - level;
      const br = f[(y + 1) * cols + x + 1] - level;
      const bl = f[(y + 1) * cols + x] - level;
      const code = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const T = () => out.push(x + tl / (tl - tr), y);
      const R = () => out.push(x + 1, y + tr / (tr - br));
      const B = () => out.push(x + bl / (bl - br), y + 1);
      const L = () => out.push(x, y + tl / (tl - bl));
      const centre = tl + tr + br + bl > 0;
      switch (code) {
        case 1: case 14: L(); B(); break;
        case 2: case 13: B(); R(); break;
        case 3: case 12: L(); R(); break;
        case 4: case 11: T(); R(); break;
        case 6: case 9: T(); B(); break;
        case 7: case 8: L(); T(); break;
        case 5: // tr and bl above the level
          if (centre) { L(); T(); B(); R(); } else { T(); R(); L(); B(); }
          break;
        case 10: // tl and br above the level
          if (centre) { T(); R(); L(); B(); } else { L(); T(); B(); R(); }
          break;
      }
    }
  }
  return out;
}

/**
 * Decision boundary lines on a grid of class probabilities: where the most likely class changes.
 * Two classes: the zero contour of p₀ − p₁. More: the edge of every class's region (each line
 * between two regions is found from both sides and drawn twice, which is invisible).
 */
export function boundarySegments(probs: ArrayLike<number>, classes: number, cols: number, rows: number): number[] {
  if (classes === 2) {
    const n = cols * rows;
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = probs[2 * i] - probs[2 * i + 1];
    return marchingSquares(f, cols, rows);
  }
  const out: number[] = [];
  const f = new Float32Array(cols * rows);
  for (let k = 0; k < classes; k++) {
    marginField(probs, classes, k, f);
    for (const v of marchingSquares(f, cols, rows)) out.push(v);
  }
  return out;
}

// ── Surfaces in 3-D ──────────────────────────────────────────────────────

export interface Mesh {
  /** Vertex positions in grid units (node (i, j, k) sits at (i, j, k)), 3 per vertex. */
  verts: Float32Array;
  /** Four vertex indices per quad, in order around the quad. */
  quads: Uint32Array;
}

const CORNERS = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
const EDGES = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/**
 * The surface where a field sampled on an n × n × n grid crosses `level`, by naive surface nets:
 * one vertex inside every grid cell the surface passes through (the mean of the crossings on its
 * edges), and one quad for every grid edge that crosses the level, joining the four cells around
 * it. Node (i, j, k) is f[i + n·(j + n·k)]. `keep(a, b)` may drop quads: `a` is the node above the
 * level, `b` the one below.
 */
export function surfaceNets(f: ArrayLike<number>, n: number, level = 0, keep?: (above: number, below: number) => boolean): Mesh {
  const m = n - 1;
  const cell = new Int32Array(m * m * m).fill(-1);
  const verts: number[] = [];
  const v = new Float64Array(8);
  const idx = (i: number, j: number, k: number) => i + n * (j + n * k);
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < m; j++) {
      for (let i = 0; i < m; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const o = CORNERS[c];
          v[c] = f[idx(i + o[0], j + o[1], k + o[2])] - level;
          if (v[c] > 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let cnt = 0;
        for (const [a, b] of EDGES) {
          if (v[a] > 0 === v[b] > 0) continue;
          const t = v[a] / (v[a] - v[b]);
          const A = CORNERS[a];
          const B = CORNERS[b];
          sx += A[0] + (B[0] - A[0]) * t;
          sy += A[1] + (B[1] - A[1]) * t;
          sz += A[2] + (B[2] - A[2]) * t;
          cnt++;
        }
        cell[i + m * (j + m * k)] = verts.length / 3;
        verts.push(i + sx / cnt, j + sy / cnt, k + sz / cnt);
      }
    }
  }
  const quads: number[] = [];
  const cellAt = (i: number, j: number, k: number) => cell[i + m * (j + m * k)];
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = idx(i, j, k);
        const fa = f[a] - level;
        for (let d = 0; d < 3; d++) {
          const p = [i, j, k];
          p[d]++;
          if (p[d] >= n) continue;
          const b = idx(p[0], p[1], p[2]);
          const fb = f[b] - level;
          if (fa > 0 === fb > 0) continue;
          // The four cells around this edge vary along the other two axes.
          const u = (d + 1) % 3;
          const w = (d + 2) % 3;
          const base = [i, j, k];
          if (base[u] < 1 || base[w] < 1 || base[u] > m - 1 || base[w] > m - 1 || base[d] > m - 1) continue;
          const at = (du: number, dw: number) => {
            const q = base.slice();
            q[u] -= du;
            q[w] -= dw;
            return cellAt(q[0], q[1], q[2]);
          };
          if (keep && !(fa > 0 ? keep(a, b) : keep(b, a))) continue;
          const c0 = at(1, 1);
          const c1 = at(0, 1);
          const c2 = at(0, 0);
          const c3 = at(1, 0);
          if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue;
          if (fa > 0) quads.push(c0, c1, c2, c3);
          else quads.push(c3, c2, c1, c0);
        }
      }
    }
  }
  return { verts: Float32Array.from(verts), quads: Uint32Array.from(quads) };
}

/**
 * Boundary surfaces between predicted classes on an n³ grid of probabilities (node order as in
 * surfaceNets). Two classes: the p₀ = p₁ surface. More: every class region's edge, each surface
 * between two regions kept once (from the lower class), so translucent layers do not double up.
 */
export function classSurfaces(probs: ArrayLike<number>, classes: number, n: number): Mesh {
  if (classes === 2) {
    const N = n * n * n;
    const f = new Float32Array(N);
    for (let i = 0; i < N; i++) f[i] = probs[2 * i] - probs[2 * i + 1];
    return surfaceNets(f, n);
  }
  const label = argmaxField(probs, classes);
  const parts: Mesh[] = [];
  const f = new Float32Array(n * n * n);
  for (let k = 0; k < classes; k++) {
    marginField(probs, classes, k, f);
    parts.push(surfaceNets(f, n, 0, (_above, below) => label[below] > k));
  }
  return mergeMeshes(parts);
}

export function mergeMeshes(parts: Mesh[]): Mesh {
  const nv = parts.reduce((s, p) => s + p.verts.length, 0);
  const nq = parts.reduce((s, p) => s + p.quads.length, 0);
  const verts = new Float32Array(nv);
  const quads = new Uint32Array(nq);
  let ov = 0;
  let oq = 0;
  for (const p of parts) {
    verts.set(p.verts, ov);
    const base = ov / 3;
    for (let i = 0; i < p.quads.length; i++) quads[oq + i] = p.quads[i] + base;
    ov += p.verts.length;
    oq += p.quads.length;
  }
  return { verts, quads };
}

// ── Orbit camera ─────────────────────────────────────────────────────────

/**
 * Rotation for an orbit view, as 9 numbers: the screen-right, screen-up and depth (into the
 * screen) directions in data coordinates. x₃ is up. `yaw` turns the data about x₃; `pitch` raises
 * the eye above the x₁x₂ plane (radians).
 */
export function orbit(yaw: number, pitch: number): number[] {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const ce = Math.cos(pitch);
  const se = Math.sin(pitch);
  return [cy, -sy, 0, sy * se, cy * se, ce, sy * ce, cy * ce, -se];
}

/** Orthographic projection of (x, y, z) to screen pixels (y down) and depth (larger is farther). */
export function project(m: number[], x: number, y: number, z: number, scale: number, cx: number, cy: number, out: number[] = [0, 0, 0]): number[] {
  out[0] = cx + scale * (m[0] * x + m[1] * y + m[2] * z);
  out[1] = cy - scale * (m[3] * x + m[4] * y + m[5] * z);
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

/** Indices sorted from farthest to nearest (painter's algorithm: draw in this order). */
export function backToFront(depth: ArrayLike<number>): Uint32Array {
  const order = new Uint32Array(depth.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  order.sort((a, b) => depth[b] - depth[a]);
  return order;
}

/**
 * Which side of a plane perpendicular to data axis `axis` faces the camera: +1 when points with a
 * larger coordinate on that axis are nearer to the viewer, −1 otherwise.
 */
export function nearSide(m: number[], axis: number): 1 | -1 {
  return m[6 + axis] > 0 ? -1 : 1;
}

// ── Colour scales ────────────────────────────────────────────────────────

/**
 * Colour scale for one layer's unit maps: the 98th percentile of |value| over every unit and
 * position, so one unbounded ReLU unit does not wash all the others out to blank tiles. Values
 * beyond it show at full colour.
 */
export function robustScale(a: ArrayLike<number>): number {
  const n = a.length;
  if (!n) return 1;
  const step = Math.max(1, Math.floor(n / 4096));
  const s: number[] = [];
  for (let i = 0; i < n; i += step) s.push(Math.abs(a[i]));
  s.sort((x, y) => x - y);
  const q = s[Math.min(s.length - 1, Math.floor(0.98 * s.length))];
  if (q > 1e-6) return q;
  let m = 0;
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i]));
  return m || 1;
}

// ── Budgets and labels ───────────────────────────────────────────────────

/**
 * Grid resolution (per side) whose res^dims evaluations fit in `budgetMs`, given the measured cost
 * of one evaluation, clamped to [min, max].
 */
export function resolutionFor(msPerPoint: number, budgetMs: number, dims: 2 | 3, min: number, max: number): number {
  if (!(msPerPoint > 0)) return max;
  const r = Math.floor(Math.pow(budgetMs / msPerPoint, 1 / dims) + 1e-9);
  return Math.max(min, Math.min(max, r));
}

/** Tick positions every `step` inside [−r, r]. */
export function ticks(r: number, step = 0.5): number[] {
  const out: number[] = [];
  const n = Math.floor(r / step + 1e-9);
  for (let i = -n; i <= n; i++) out.push(Math.round(i * step * 1e6) / 1e6);
  return out;
}

const SUB = ['₁', '₂', '₃'];
/** Axis name with a subscript: axisName(0) = "x₁". */
export const axisName = (axis: number): string => `x${SUB[axis] ?? axis + 1}`;
