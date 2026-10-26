import { fixed } from '../analysis/stats';
import type { Data } from '../data/datasets';
import { pointDomain, type PointEvaluator } from '../data/grid';
import { store } from '../store';
import { h } from './dom';
import { fitCanvas } from './draw';
import {
  Cost,
  Evaluator,
  GridJob,
  Live,
  PlaneCanvas,
  PlaneJob,
  boundaryOpts,
  coarseNote,
  netFits,
  onBoundaryOpts,
  pickAt,
  pointCoords,
  probeCross,
  probsText,
  regionCanvas,
  sliceMs,
  srHint,
  type Budget,
  type Marker,
  type PlaneField,
  type PlaneHost,
  type PointRef,
  type View,
} from './boundary2d';
import { axisName, backToFront, classSurfaces, nearSide, orbit, project, type Mesh } from './boundaryMath';
import { drawMarks, type MarkKind } from './marks';
import { onThemeChange, palette } from './theme';
import { hideTip, showTip } from './tip';

/**
 * Section 03 for 3-D point datasets: the data cube with the decision surface, turned by dragging,
 * plus a movable slice through it shown flat beside the cube.
 */

// ── The slice (shared with the network view, whose unit maps show this plane) ──

export const slice = { axis: 2, pos: 0 };
const sliceListeners = new Set<() => void>();
export function setSlice(patch: Partial<typeof slice>): void {
  Object.assign(slice, patch);
  for (const fn of sliceListeners) fn();
}
export const onSlice = (fn: () => void) => sliceListeners.add(fn);

/** The two data axes spanning a slice perpendicular to `axis` (map x, map y). */
export const sliceAxes = (axis: number): [number, number] => (axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1]);

/** Points closer to the slice than this (in data units) are shown on the slice map. */
export const slabWidth = (r: number) => Math.max(0.15, 0.16 * r);

/** True when data coordinates `c` lie on the current slice (within slabWidth), as every view reads it. */
export const onCurrentSlice = (c: ArrayLike<number>, r: number) => Math.abs(c[slice.axis] - slice.pos) <= slabWidth(r);

const DEG = Math.PI / 180;
const HOME = { yaw: 34 * DEG, pitch: 24 * DEG };
const PITCH_MAX = 85 * DEG;

interface Voxels {
  n: number;
  r: number;
  classes: number;
  /** Boundary surface in data coordinates. */
  mesh: Mesh;
}

/**
 * The cube evaluated a slice at a time: an n³ grid of nodes from face to face of [−r, r]³ (so the
 * surface, carried out to the border nodes, meets the faces of the drawn cube).
 */
class VoxelJob {
  readonly job: GridJob;
  constructor(
    ev: PointEvaluator,
    readonly n: number,
    readonly r: number,
  ) {
    const coords = new Float32Array(n * n * n * 3);
    const step = (2 * r) / (n - 1);
    let o = 0;
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          coords[o++] = -r + i * step;
          coords[o++] = -r + j * step;
          coords[o++] = -r + k * step;
        }
      }
    }
    this.job = new GridJob(ev, coords, 3);
  }
  step(ms: number, perPoint: number): Voxels | null {
    if (!this.job.step(ms, perPoint)) return null;
    const { n, r } = this;
    const probs = this.job.probs!;
    const classes = probs.length / (n * n * n);
    const mesh = classSurfaces(probs, classes, n, true);
    // Grid units → data coordinates.
    const step = (2 * r) / (n - 1);
    for (let i = 0; i < mesh.verts.length; i++) mesh.verts[i] = -r + mesh.verts[i] * step;
    return { n, r, classes, mesh };
  }
}

/** Grid budgets of the cube and the slice map (see Budget); passes are sliced across frames. */
export const BUDGET_3D: { cube: { running: Budget; paused: Budget }; slice: { running: Budget; paused: Budget } } = {
  cube: {
    running: { budget: 10, min: 8, max: 20, cap: 40, floor: 5 },
    paused: { budget: 120, min: 12, max: 30, cap: 400, floor: 6 },
  },
  slice: {
    running: { budget: 5, min: 16, max: 64, cap: 20, floor: 8 },
    paused: { budget: 16, min: 24, max: 72, cap: 120, floor: 10 },
  },
};

/** Opacity levels of the surface: quads are filled in one path per depth layer and level. */
const ALPHA_STEPS = 16;
/** Depth layers for drawing back to front (points and surface interleave by layer). */
const DEPTH_BINS = 12;

export function create3D(): View {
  const ev = new Evaluator();
  const cost = new Cost();
  let vox: Voxels | null = null;
  let field: PlaneField | null = null;
  let vjob: VoxelJob | null = null;
  let sjob: PlaneJob | null = null;
  /** Main-thread time of the pass under way, all frames together. */
  let passMs = 0;
  let sliceDirty = true;
  let yaw = HOME.yaw;
  let pitch = HOME.pitch;

  // ── Elements ──
  const orbitKeys = srHint('Arrow keys turn the cube, 10° per press or 30° with Shift; Home returns to the starting view. Clicking a point makes it the network’s input.');
  // An interactive plot, not a picture: it takes the arrow keys (role application).
  const canvas = h('canvas', { class: 'bd-orbit', tabindex: '0', role: 'application', 'aria-roledescription': '3-D plot', 'aria-describedby': orbitKeys.id }) as HTMLCanvasElement;
  const orbitBox = h('div', { class: 'bd-orbit-box' }, canvas);
  const btn = (label: string, title: string, fn: () => void, aria?: string) => h('button', { type: 'button', class: 'btn btn-sm', title, 'aria-label': aria, onclick: fn }, label);
  const turn = (dy: number, dp = 0) => {
    yaw = (((yaw + dy) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    pitch = Math.max(-PITCH_MAX, Math.min(PITCH_MAX, pitch + dp));
    live.request(false);
  };
  const viewBtns = h(
    'div',
    { class: 'bd-row', role: 'group', 'aria-label': 'Turn the cube' },
    btn('↺ Turn', 'Turn the cube 30° to the left', () => turn(-30 * DEG), 'Turn left 30°'),
    btn('Turn ↻', 'Turn the cube 30° to the right', () => turn(30 * DEG), 'Turn right 30°'),
    btn('Top', 'Look straight down the x₃ axis', () => {
      pitch = PITCH_MAX;
      live.request(false);
    }),
    btn('Reset view', 'Back to the starting angle', () => {
      yaw = HOME.yaw;
      pitch = HOME.pitch;
      live.request(false);
    }),
    h('span', { class: 'hint bd-hint-inline' }, 'Drag to turn, or use the arrow keys.'),
  );
  const note = h('p', { class: 'hint bd-res-note', hidden: true });

  const axisSeg = h('div', { class: 'seg bd-axis-seg', role: 'group', 'aria-label': 'Slice across axis' });
  const axisBtns = [0, 1, 2].map((a) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(a === slice.axis) }, axisName(a));
    b.addEventListener('click', () => setSlice({ axis: a }));
    axisSeg.append(b);
    return b;
  });
  const slider = h('input', { type: 'range', id: 'bd-slice', min: '-1.25', max: '1.25', step: '0.05', value: '0' }) as HTMLInputElement;
  const sliceVal = h('output', { class: 'mono bd-slice-val', for: 'bd-slice' });
  slider.addEventListener('input', () => setSlice({ pos: Number(slider.value) }));
  const sliceNote = h('p', { class: 'hint' });

  const host: PlaneHost = {
    field: () => field,
    markers: () => (store.data?.points ? sliceMarkers(store.data) : []),
    probe: () => {
      const c = store.probe?.coords;
      if (!c || c.length !== 3 || !field) return null;
      const [a, b] = sliceAxes(slice.axis);
      return { u: c[a], v: c[b], on: onCurrentSlice(c, field.r) };
    },
    coordsAt: (u, v) => {
      const c = new Float32Array(3);
      const [a, b] = sliceAxes(slice.axis);
      c[a] = u;
      c[b] = v;
      c[slice.axis] = slice.pos;
      return c;
    },
    probsAt: (coords) => {
      // A new network (architecture or features) replaces any pass made with the old one.
      if (netFits() && !ev.current && (vjob || sjob)) {
        vjob = sjob = null;
        live.restart();
      }
      return ev.probsAt(coords, 3);
    },
  };
  const plane = new PlaneCanvas(host, true, 'Slice map');
  const planeBox = h('div', { class: 'bd-slice-plot' }, plane.canvas);
  const sliceCtl = h(
    'div',
    { class: 'bd-slice-ctl' },
    h('span', { class: 'label' }, 'Slice'),
    h('div', { class: 'bd-row' }, h('span', { class: 'hint' }, 'Across'), axisSeg),
    h('label', { class: 'bd-slider', for: 'bd-slice' }, h('span', { class: 'hint' }, 'Position'), slider, sliceVal),
    sliceNote,
  );
  const el = h(
    'div',
    { class: 'bd-3d' },
    h('div', { class: 'bd-3d-main' }, orbitBox, viewBtns, note, orbitKeys),
    h('div', { class: 'bd-3d-side' }, sliceCtl, planeBox, plane.keysHint),
  );

  /** Points near the slice, faded with distance from it. */
  function sliceMarkers(d: Data): Marker[] {
    const p = d.points!;
    const r = field?.r ?? pointDomain(d);
    const slab = slabWidth(r);
    const [a, b] = sliceAxes(slice.axis);
    const out: Marker[] = [];
    const add = (split: 'train' | 'test', c: Float32Array, y: Uint8Array) => {
      for (let i = 0; i < y.length; i++) {
        const dist = Math.abs(c[3 * i + slice.axis] - slice.pos);
        if (dist > slab) continue;
        out.push({ split, i, u: c[3 * i + a], v: c[3 * i + b], cls: y[i], alpha: 1 - 0.65 * (dist / slab) });
      }
    };
    if (boundaryOpts.showTest) add('test', p.testCoords, d.testY);
    add('train', p.trainCoords, d.trainY);
    for (const c of store.custom) {
      if (c.origin !== 'point' || c.coords?.length !== 3) continue;
      const dist = Math.abs(c.coords[slice.axis] - slice.pos);
      if (dist <= slab) out.push({ split: 'custom', i: c.id, u: c.coords[a], v: c.coords[b], cls: c.y, alpha: 1 - 0.65 * (dist / slab) });
    }
    return out;
  }

  // ── Drawing the cube ──
  let screenPts: { x: number; y: number; ref: PointRef }[] = [];
  const tex = document.createElement('canvas');
  let texKey = '';

  function drawOrbit(d: Data): void {
    const p = palette();
    const W = Math.max(240, orbitBox.clientWidth || 480);
    const H = Math.round(Math.min(W * 0.86, 540));
    const ctx = fitCanvas(canvas, W, H);
    const dpr = canvas.width / W;
    const r = vox?.r ?? field?.r ?? pointDomain(d);
    const m = orbit(yaw, pitch);
    const scale = Math.min((W / 2 - 26) / (r * Math.SQRT2), (H / 2 - 22) / (r * Math.sqrt(3)));
    const cx = W / 2;
    const cy = H / 2;
    const P = (x: number, y: number, z: number) => project(m, x, y, z, scale, cx, cy);
    // The slice drawn is the one the slice map was computed for (it may trail the slider by a frame).
    const sAxis = field ? 3 - field.axes[0] - field.axes[1] : slice.axis;
    const sPos = field ? field.fixedAt[sAxis] : slice.pos;
    const near = nearSide(m, sAxis);
    const side = (c: number) => (c - sPos) * near >= 0; // true = between the slice and the viewer

    // Cube edges: those behind the centre first.
    const corners: number[][] = [];
    for (let k = 0; k < 8; k++) corners.push([k & 1 ? r : -r, k & 2 ? r : -r, k & 4 ? r : -r]);
    const edges: [number, number][] = [];
    for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) if ([1, 2, 4].includes(a ^ b)) edges.push([a, b]);
    const pc = corners.map((c) => P(c[0], c[1], c[2]));
    // One edge per axis carries its name: the lowest of the horizontal ones, the leftmost upright.
    const axisEdges = [0, 1, 2].map((a) => {
      let best: [number, number] = [0, 1 << a];
      let score = -Infinity;
      for (let k = 0; k < 8; k++) {
        if (k & (1 << a)) continue;
        const e: [number, number] = [k, k | (1 << a)];
        const sc = a === 2 ? -(pc[e[0]][0] + pc[e[1]][0]) : pc[e[0]][1] + pc[e[1]][1];
        if (sc > score + 1e-6) {
          score = sc;
          best = e;
        }
      }
      return best;
    });
    const drawEdges = (front: boolean) => {
      for (const [a, b] of edges) {
        const isFront = pc[a][2] + pc[b][2] < 0;
        if (isFront !== front) continue;
        const axisEdge = axisEdges.some((e) => e[0] === a && e[1] === b);
        ctx.beginPath();
        ctx.moveTo(pc[a][0], pc[a][1]);
        ctx.lineTo(pc[b][0], pc[b][1]);
        ctx.strokeStyle = axisEdge ? p.ink : front ? p.muted : p.hair;
        ctx.lineWidth = axisEdge ? 1.6 : 1;
        ctx.stroke();
      }
    };
    drawEdges(false);

    // Everything with a depth goes into layers, drawn back to front: in each layer the surface
    // first, filled in one path per opacity level (no seams where quads meet, and a few fills
    // instead of one per quad), then the points of that layer, one path per class and kind.
    const zr = r * Math.sqrt(3);
    const binOf = (z: number) => Math.max(0, Math.min(DEPTH_BINS - 1, Math.floor(((z + zr) / (2 * zr)) * DEPTH_BINS)));
    interface Layer {
      quads: number[][]; // per opacity level: quad indices
      pts: { kind: MarkKind; cls: number; i: number; x: number; y: number; z: number }[];
      probe: { x: number; y: number } | null;
    }
    const layers: Layer[][] = [0, 1].map(() => Array.from({ length: DEPTH_BINS }, () => ({ quads: Array.from({ length: ALPHA_STEPS }, () => []), pts: [], probe: null })));
    const layer = (z: number, isNear: boolean) => layers[isNear ? 1 : 0][binOf(z)];

    const mesh = vox?.mesh;
    const nq = mesh ? mesh.quads.length / 4 : 0;
    const vs = mesh ? new Float32Array(mesh.verts.length) : null;
    // Light from the upper left, toward the scene.
    const L = [-0.45, 0.6, 0.66];
    const A_MIN = 0.14;
    const A_MAX = 0.14 + 0.36 + 0.14;
    if (mesh && vs) {
      const tmp = [0, 0, 0];
      for (let v = 0; v < mesh.verts.length / 3; v++) {
        project(m, mesh.verts[3 * v], mesh.verts[3 * v + 1], mesh.verts[3 * v + 2], scale, cx, cy, tmp);
        vs[3 * v] = tmp[0];
        vs[3 * v + 1] = tmp[1];
        vs[3 * v + 2] = tmp[2];
      }
      const V = mesh.verts;
      for (let q = 0; q < nq; q++) {
        const qv = mesh.quads.subarray(4 * q, 4 * q + 4);
        let z = 0;
        let c = 0;
        for (let t = 0; t < 4; t++) {
          z += vs[3 * qv[t] + 2];
          c += V[3 * qv[t] + sAxis];
        }
        const ax = V[3 * qv[2]] - V[3 * qv[0]];
        const ay = V[3 * qv[2] + 1] - V[3 * qv[0] + 1];
        const az = V[3 * qv[2] + 2] - V[3 * qv[0] + 2];
        const bx = V[3 * qv[3]] - V[3 * qv[1]];
        const by = V[3 * qv[3] + 1] - V[3 * qv[1] + 1];
        const bz = V[3 * qv[3] + 2] - V[3 * qv[1] + 2];
        let nx = ay * bz - az * by;
        let ny = az * bx - ax * bz;
        let nz = ax * by - ay * bx;
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl;
        ny /= nl;
        nz /= nl;
        // Normal in view space (right, up, depth).
        const vr = m[0] * nx + m[1] * ny + m[2] * nz;
        const vu = m[3] * nx + m[4] * ny + m[5] * nz;
        const vd = m[6] * nx + m[7] * ny + m[8] * nz;
        const facing = Math.abs(vd);
        const lit = Math.abs(vr * L[0] + vu * L[1] - vd * L[2]);
        const a = A_MIN + 0.36 * Math.pow(1 - facing, 1.5) + 0.14 * (1 - lit);
        const level = Math.max(0, Math.min(ALPHA_STEPS - 1, Math.round(((a - A_MIN) / (A_MAX - A_MIN)) * (ALPHA_STEPS - 1))));
        layer(z / 4, side(c / 4)).quads[level].push(q);
      }
    }
    const pts = d.points!;
    const addPts = (kind: MarkKind, c: Float32Array, y: Uint8Array) => {
      for (let i = 0; i < y.length; i++) {
        const q = P(c[3 * i], c[3 * i + 1], c[3 * i + 2]);
        layer(q[2], side(c[3 * i + sAxis])).pts.push({ kind, cls: y[i], i, x: q[0], y: q[1], z: q[2] });
      }
    };
    if (boundaryOpts.showTest) addPts('test', pts.testCoords, d.testY);
    addPts('train', pts.trainCoords, d.trainY);
    for (const c of store.custom) {
      if (c.origin !== 'point' || c.coords?.length !== 3) continue;
      const q = P(c.coords[0], c.coords[1], c.coords[2]);
      layer(q[2], side(c.coords[sAxis])).pts.push({ kind: 'custom', cls: c.y, i: c.id, x: q[0], y: q[1], z: q[2] });
    }
    const pr = store.probe?.coords;
    if (pr && pr.length === 3) {
      const q = P(pr[0], pr[1], pr[2]);
      layer(q[2], side(pr[sAxis])).probe = { x: q[0], y: q[1] };
    }

    const rad = Math.max(2.2, Math.min(3.6, W / 170));
    const ink = p.rgb.ink;
    const hits: { x: number; y: number; z: number; ref: PointRef }[] = [];
    const drawLayer = (ly: Layer, bin: number) => {
      if (mesh && vs) {
        for (let level = 0; level < ALPHA_STEPS; level++) {
          const qs = ly.quads[level];
          if (!qs.length) continue;
          ctx.beginPath();
          for (const q of qs) {
            const qv = mesh.quads.subarray(4 * q, 4 * q + 4);
            // Same winding on screen for every quad, so overlapping ones fill once (nonzero rule).
            let area = 0;
            for (let t = 0; t < 4; t++) {
              const a = qv[t];
              const b = qv[(t + 1) % 4];
              area += vs[3 * a] * vs[3 * b + 1] - vs[3 * b] * vs[3 * a + 1];
            }
            const order = area >= 0 ? [0, 1, 2, 3] : [3, 2, 1, 0];
            ctx.moveTo(vs[3 * qv[order[0]]], vs[3 * qv[order[0]] + 1]);
            for (let t = 1; t < 4; t++) ctx.lineTo(vs[3 * qv[order[t]]], vs[3 * qv[order[t]] + 1]);
            ctx.closePath();
          }
          const a = A_MIN + (level / (ALPHA_STEPS - 1)) * (A_MAX - A_MIN);
          ctx.fillStyle = `rgba(${ink[0] | 0},${ink[1] | 0},${ink[2] | 0},${a.toFixed(3)})`;
          ctx.fill();
        }
      }
      if (ly.pts.length) {
        const dn = (bin + 0.5) / DEPTH_BINS; // 0 near, 1 far
        const s = rad * (1.22 - 0.5 * dn);
        ctx.globalAlpha = 1 - 0.55 * dn;
        const groups = new Map<string, { kind: MarkKind; cls: number; xy: number[] }>();
        for (const kind of ['test', 'train', 'custom'] as const) {
          for (const it of ly.pts) {
            if (it.kind !== kind) continue;
            const key = `${kind}:${it.cls}`;
            let g = groups.get(key);
            if (!g) groups.set(key, (g = { kind, cls: it.cls, xy: [] }));
            g.xy.push(it.x, it.y);
            hits.push({ x: it.x, y: it.y, z: it.z, ref: { split: kind, i: it.i } });
          }
        }
        for (const g of groups.values()) drawMarks(ctx, g.cls, g.kind, g.xy, s, { color: p.cat[g.cls % 10], ink: p.ink, surface: p.surface });
        ctx.globalAlpha = 1;
      }
      if (ly.probe) probeCross(ctx, ly.probe.x, ly.probe.y, 6, p.accent, p.surface);
    };
    for (let b = DEPTH_BINS - 1; b >= 0; b--) drawLayer(layers[0][b], b);

    // The slice: a quad painted with the class regions on that plane.
    if (field) {
      const [ua, va] = field.axes;
      const world = (u: number, v: number) => {
        const c = [0, 0, 0];
        c[ua] = u;
        c[va] = v;
        c[sAxis] = sPos;
        return P(c[0], c[1], c[2]).slice();
      };
      const P00 = world(-r, r);
      const P10 = world(r, r);
      const P01 = world(-r, -r);
      const P11 = world(r, -r);
      const key = `${field.rev}:${boundaryOpts.discrete}:${p.surface}:${p.cat.join()}`;
      if (key !== texKey) {
        regionCanvas(field, boundaryOpts.discrete, tex);
        texKey = key;
      }
      const res = field.res;
      ctx.save();
      ctx.globalAlpha = 0.6;
      ctx.imageSmoothingEnabled = true;
      ctx.setTransform(
        (dpr * (P10[0] - P00[0])) / res,
        (dpr * (P10[1] - P00[1])) / res,
        (dpr * (P01[0] - P00[0])) / res,
        (dpr * (P01[1] - P00[1])) / res,
        dpr * P00[0],
        dpr * P00[1],
      );
      ctx.drawImage(tex, 0, 0);
      ctx.restore();
      // Boundary line on the slice.
      const s = field.segs;
      const map = (tx: number, ty: number): [number, number] => [
        P00[0] + (tx / res) * (P10[0] - P00[0]) + (ty / res) * (P01[0] - P00[0]),
        P00[1] + (tx / res) * (P10[1] - P00[1]) + (ty / res) * (P01[1] - P00[1]),
      ];
      ctx.beginPath();
      for (let i = 0; i < s.length; i += 4) {
        const a = map(s[i] + 0.5, s[i + 1] + 0.5);
        const b = map(s[i + 2] + 0.5, s[i + 3] + 0.5);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.6;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.moveTo(P00[0], P00[1]);
      ctx.lineTo(P10[0], P10[1]);
      ctx.lineTo(P11[0], P11[1]);
      ctx.lineTo(P01[0], P01[1]);
      ctx.closePath();
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    for (let b = DEPTH_BINS - 1; b >= 0; b--) drawLayer(layers[1][b], b);
    drawEdges(true);
    // Hit testing prefers the point nearest the viewer: later entries win ties.
    const order = backToFront(hits.map((t) => t.z));
    screenPts = Array.from(order, (i) => hits[i]);

    // Axis names just outside the positive end of their edges.
    ctx.font = '700 13px Archivo, "Helvetica Neue", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let a = 0; a < 3; a++) {
      const [i0, i1] = axisEdges[a];
      const o = pc[i0];
      const e = pc[i1];
      const dx = e[0] - o[0];
      const dy = e[1] - o[1];
      const len = Math.hypot(dx, dy);
      if (len < 12) continue; // seen end-on
      const mx = (o[0] + e[0]) / 2 - cx;
      const my = (o[1] + e[1]) / 2 - cy;
      const ml = Math.hypot(mx, my) || 1;
      const x = e[0] + (dx / len) * 10 + (mx / ml) * 10;
      const y = e[1] + (dy / len) * 10 + (my / ml) * 10;
      ctx.lineWidth = 4;
      ctx.strokeStyle = p.bg;
      ctx.strokeText(axisName(a), x, y);
      ctx.fillStyle = p.ink;
      ctx.fillText(axisName(a), x, y);
    }
    const yd = Math.round(yaw / DEG);
    const pd = Math.round(pitch / DEG);
    canvas.dataset.yaw = String(yd);
    canvas.dataset.pitch = String(pd);
    canvas.dataset.quads = String(nq);
    if (vox) canvas.dataset.res = String(vox.n);
    canvas.setAttribute(
      'aria-label',
      `3-D decision boundary of ${d.info.name}, turned to azimuth ${yd}°, elevation ${pd}°. The grey surface is where the predicted class changes; the coloured square is the slice ${axisName(slice.axis)} = ${fixed(slice.pos, 2)}. Each class has its own shape. Drag, or use the arrow keys, to turn it.`,
    );
  }

  function drawSlice(): void {
    if (!field) return;
    const w = Math.min(340, planeBox.clientWidth || 300);
    plane.draw(w);
    const slab = slabWidth(field.r);
    plane.canvas.dataset.slice = `${slice.axis}:${slice.pos.toFixed(2)}`;
    const [a, b] = sliceAxes(slice.axis);
    plane.canvas.setAttribute(
      'aria-label',
      `Slice map at ${axisName(slice.axis)} = ${fixed(slice.pos, 2)}: the predicted class over ${axisName(a)} and ${axisName(b)}, with the points within ${fixed(slab, 2)} of the slice.`,
    );
  }

  const syncControls = (r: number) => {
    axisBtns.forEach((b, a) => b.setAttribute('aria-pressed', String(a === slice.axis)));
    slider.min = String(-r);
    slider.max = String(r);
    if (Math.abs(slice.pos) > r) slice.pos = Math.sign(slice.pos) * r;
    slider.value = String(slice.pos);
    slider.setAttribute('aria-valuetext', `${axisName(slice.axis)} = ${fixed(slice.pos, 2)}`);
    sliceVal.textContent = `${axisName(slice.axis)} = ${fixed(slice.pos, 2)}`;
    const [ua, va] = sliceAxes(slice.axis);
    sliceNote.textContent = `The flat map shows the plane where ${axisName(slice.axis)} = ${fixed(slice.pos, 2)}: ${axisName(ua)} across, ${axisName(va)} up, with the points within ±${fixed(slabWidth(r), 2)} of it. Click it to try a point; with Add points on, click to add one.`;
  };

  const startSlice = (r: number) => {
    if (!ev.current) ev.sync();
    const res = cost.res(store.running ? BUDGET_3D.slice.running : BUDGET_3D.slice.paused, 2);
    const fixedAt = [0, 0, 0];
    fixedAt[slice.axis] = slice.pos;
    sjob = new PlaneJob(ev.ev, 3, sliceAxes(slice.axis), fixedAt, r, res);
    sliceDirty = false;
  };

  const render = (evaluate: boolean, draw: boolean): boolean => {
    const d = store.data;
    if (!d?.points || d.points.dims !== 3 || !netFits()) {
      vjob = sjob = null;
      return false;
    }
    const t0 = performance.now();
    const r = pointDomain(d);
    syncControls(r);
    if ((vjob || sjob) && !ev.current) vjob = sjob = null; // made for a network that is gone
    const idle = !vjob && !sjob;
    if (idle && (evaluate || !vox)) {
      ev.sync();
      vjob = new VoxelJob(ev.ev, cost.res(store.running ? BUDGET_3D.cube.running : BUDGET_3D.cube.paused, 3), r);
      sliceDirty = true;
      passMs = 0;
    } else if (idle && (sliceDirty || !field)) {
      startSlice(r);
      passMs = 0;
    }
    const inPass = !!(vjob || sjob);
    let fresh = false;
    if (vjob) {
      const out = vjob.step(sliceMs(), cost.perPoint);
      if (out) {
        cost.add(vjob.job.busy, vjob.n ** 3);
        vox = out;
        vjob = null;
        canvas.dataset.weights = String(store.weightsRev);
        canvas.dataset.ms = (passMs + performance.now() - t0).toFixed(1);
        startSlice(r);
        fresh = true;
      }
    } else if (sjob) {
      const f = sjob.step(sliceMs(), cost.perPoint);
      if (f) {
        cost.add(sjob.busy, sjob.res * sjob.res);
        field = f;
        sjob = null;
        fresh = true;
      }
    }
    if (draw || fresh) {
      drawOrbit(d);
      drawSlice();
      const coarse = vox && vox.n < 12;
      note.hidden = !coarse;
      if (coarse) note.textContent = coarseNote(vox!.n, 'the surface', 3);
      live.noteDraw(t0, performance.now() - t0);
    }
    if (inPass) passMs += performance.now() - t0;
    // Every millisecond of a pass counts against the budget: evaluation, meshing, drawing, and the
    // browser rasterising the drawing afterwards.
    if (fresh && !vjob && !sjob) live.gap = Math.max(100, 4 * (passMs + live.drawCost));
    // A slider move during a pass is picked up when the pass is done.
    if (!vjob && !sjob && sliceDirty) {
      startSlice(r);
      passMs = 0;
    }
    return !!(vjob || sjob);
  };
  const live = new Live(el, render);
  live.active = false;

  // ── Interaction ──
  let drag: { x: number; y: number; moved: boolean; id: number } | null = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (drag && drag.id === e.pointerId) {
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true;
      drag.x = e.clientX;
      drag.y = e.clientY;
      hideTip();
      turn(dx * 0.01, dy * 0.01);
      return;
    }
    hoverOrbit(e);
  });
  const endDrag = (e: PointerEvent) => {
    if (!drag || drag.id !== e.pointerId) return;
    const wasClick = !drag.moved;
    drag = null;
    if (wasClick) {
      const hit = hitOrbit(e);
      const d = store.data;
      if (hit && d) pickAt(pointCoords(d, hit)!, hit, false);
    }
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', () => (drag = null));
  canvas.addEventListener('pointerleave', () => hideTip());
  canvas.addEventListener('keydown', (e) => {
    const step = (e.shiftKey ? 30 : 10) * DEG;
    if (e.key === 'ArrowLeft') turn(-step);
    else if (e.key === 'ArrowRight') turn(step);
    else if (e.key === 'ArrowUp') turn(0, step);
    else if (e.key === 'ArrowDown') turn(0, -step);
    else if (e.key === 'Home') {
      yaw = HOME.yaw;
      pitch = HOME.pitch;
      live.request(false);
    } else return;
    e.preventDefault();
  });

  function hitOrbit(e: MouseEvent): PointRef | null {
    const b = canvas.getBoundingClientRect();
    const x = e.clientX - b.left;
    const y = e.clientY - b.top;
    let best = 7 * 7;
    let hit: PointRef | null = null;
    // Later entries are nearer the viewer; prefer them on ties.
    for (const s of screenPts) {
      const d2 = (s.x - x) ** 2 + (s.y - y) ** 2;
      if (d2 <= best) {
        best = d2;
        hit = s.ref;
      }
    }
    return hit;
  }

  function hoverOrbit(e: MouseEvent): void {
    const d = store.data;
    const hit = hitOrbit(e);
    canvas.style.cursor = hit ? 'pointer' : 'grab';
    if (!hit || !d) {
      hideTip();
      return;
    }
    const c = pointCoords(d, hit);
    if (!c) return;
    showTip(`${probsText(d, c, host.probsAt(c), hit)}\nClick to use this point as the input`, e.clientX, e.clientY);
  }

  // ── Events ──
  const again = () => live.request(true);
  const redraw = () => live.request(false);
  store.on('weights', again);
  store.on('model', () => {
    vjob = sjob = null;
    live.restart();
  });
  store.on('data', () => {
    vox = null;
    field = null;
    vjob = sjob = null;
    live.restart();
  });
  let wasRunning = false;
  store.on('status', () => {
    if (wasRunning && !store.running) again();
    wasRunning = store.running;
  });
  for (const e of ['probe', 'custom'] as const) store.on(e, redraw);
  onSlice(() => {
    sliceDirty = true;
    redraw();
  });
  onBoundaryOpts(redraw);
  onThemeChange(redraw);
  new ResizeObserver(redraw).observe(el);

  return {
    el,
    setActive(on: boolean) {
      live.active = on;
      if (on) again();
    },
  };
}
