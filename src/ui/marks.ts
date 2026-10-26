/**
 * Class marks for point plots (sections 02, 03 and 07). Every class has its own shape (circle,
 * square, triangle, diamond, …) drawn in its class colour, so the classes stay apart for readers
 * who cannot tell the colours apart. The fill tells the split: training points are filled, test
 * points are outlines, and the points you added are filled, larger and heavily outlined.
 *
 * The geometry is pure (unit-tested); the canvas helpers draw many marks of one class in a single
 * path, and markSvg makes the same mark for legends and tables.
 */

export const MARK_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'triangle-down', 'pentagon', 'hexagon', 'star', 'plus', 'cross'] as const;
export type MarkShape = (typeof MARK_SHAPES)[number];
export type MarkKind = 'train' | 'test' | 'custom';

/** Shape of class `cls` (class k uses shape k, as it uses colour --cat-k). */
export const markShape = (cls: number): MarkShape => MARK_SHAPES[((cls % 10) + 10) % 10];

/** Plain-language name of a shape, for keys and labels. */
export const shapeName = (cls: number): string => markShape(cls).replace('-', ' ');

/** Points you added are drawn this much larger than the others. */
export const CUSTOM_SCALE = 1.3;

const ring = (n: number, r: number, start: number): number[] => {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = start + (2 * Math.PI * i) / n;
    out.push(r * Math.cos(a), r * Math.sin(a));
  }
  return out;
};

const UP = -Math.PI / 2;

/** Unit outlines, y down, clockwise on screen, centred on their centroid; scaled below to equal area. */
function unitOutline(shape: MarkShape): number[] | null {
  switch (shape) {
    case 'circle':
      return null;
    case 'square':
      return [-1, -1, 1, -1, 1, 1, -1, 1];
    case 'triangle':
      return ring(3, 1, UP);
    case 'diamond':
      return ring(4, 1, UP);
    case 'triangle-down':
      return ring(3, 1, -UP);
    case 'pentagon':
      return ring(5, 1, UP);
    case 'hexagon':
      return ring(6, 1, 0);
    case 'star': {
      const out: number[] = [];
      for (let i = 0; i < 10; i++) {
        const a = UP + (Math.PI * i) / 5;
        const rr = i % 2 ? 0.5 : 1;
        out.push(rr * Math.cos(a), rr * Math.sin(a));
      }
      return out;
    }
    case 'plus':
    case 'cross': {
      const t = 0.36;
      const pts = [-t, -1, t, -1, t, -t, 1, -t, 1, t, t, t, t, 1, -t, 1, -t, t, -1, t, -1, -t, -t, -t];
      if (shape === 'plus') return pts;
      const c = Math.SQRT1_2;
      const out: number[] = [];
      for (let i = 0; i < pts.length; i += 2) out.push(c * (pts[i] - pts[i + 1]), c * (pts[i] + pts[i + 1]));
      return out;
    }
  }
}

/** Signed area of a polygon [x0, y0, …] (positive when clockwise on screen, y down). */
export function polygonArea(v: ArrayLike<number>): number {
  let a = 0;
  const n = v.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    a += v[2 * i] * v[2 * j + 1] - v[2 * j] * v[2 * i + 1];
  }
  return a / 2;
}

interface Unit {
  /** Outline at unit area-equivalent radius (area π), or null for the circle. */
  v: number[] | null;
  /** Largest distance of a vertex from the centre, in units of r. */
  extent: number;
}

const units: Unit[] = MARK_SHAPES.map((shape) => {
  const raw = unitOutline(shape);
  if (!raw) return { v: null, extent: 1 };
  const s = Math.sqrt(Math.PI / Math.abs(polygonArea(raw)));
  const v = raw.map((c) => c * s);
  let extent = 0;
  for (let i = 0; i < v.length; i += 2) extent = Math.max(extent, Math.hypot(v[i], v[i + 1]));
  return { v, extent };
});

/**
 * Outline of class `cls`'s mark centred on (0, 0) as polygon vertices [x0, y0, x1, y1, …] (y
 * down, clockwise on screen), scaled so that it covers the same area as a circle of radius `r`.
 * Null for the circle.
 */
export function markPolygon(cls: number, r: number): number[] | null {
  const u = units[((cls % 10) + 10) % 10];
  return u.v ? u.v.map((c) => c * r) : null;
}

/** How far class `cls`'s mark reaches from its centre, as a multiple of `r`. */
export const markExtent = (cls: number): number => units[((cls % 10) + 10) % 10].extent;

type PathLike = Pick<CanvasRenderingContext2D, 'moveTo' | 'lineTo' | 'arc' | 'closePath'>;

/** Adds class `cls`'s mark at (x, y) to the current path (a new closed sub-path). */
export function traceMark(ctx: PathLike, cls: number, x: number, y: number, r: number): void {
  const v = units[((cls % 10) + 10) % 10].v;
  if (!v) {
    ctx.moveTo(x + r, y);
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.closePath();
    return;
  }
  ctx.moveTo(x + v[0] * r, y + v[1] * r);
  for (let i = 2; i < v.length; i += 2) ctx.lineTo(x + v[i] * r, y + v[i + 1] * r);
  ctx.closePath();
}

export interface MarkColors {
  /** Class colour. */
  color: string;
  ink: string;
  surface: string;
}

/** Line widths of a mark of radius r: outline of filled marks, ring of test marks, and its halo. */
export function markStrokes(kind: MarkKind, r: number): { line: number; halo: number } {
  if (kind === 'test') {
    const line = Math.max(1.1, Math.min(1.7, 0.45 * r));
    return { line, halo: line + Math.min(1.7, 0.4 * r + 0.4) };
  }
  return { line: kind === 'custom' ? 1.7 : Math.max(0.6, Math.min(0.9, 0.3 * r)), halo: 0 };
}

/**
 * Draws marks of one class and kind at the points xy = [x0, y0, x1, y1, …], all in one path:
 * training marks filled with a thin ink outline, test marks as outlines in the class colour on a
 * surface halo (so they read on any region), your points filled, larger, heavily outlined.
 */
export function drawMarks(ctx: CanvasRenderingContext2D, cls: number, kind: MarkKind, xy: ArrayLike<number>, r: number, c: MarkColors): void {
  if (xy.length < 2) return;
  const rr = kind === 'custom' ? r * CUSTOM_SCALE : r;
  ctx.beginPath();
  for (let i = 0; i + 1 < xy.length; i += 2) traceMark(ctx, cls, xy[i], xy[i + 1], rr);
  const { line, halo } = markStrokes(kind, rr);
  ctx.lineJoin = 'round';
  if (kind === 'test') {
    ctx.lineWidth = halo;
    ctx.strokeStyle = c.surface;
    ctx.stroke();
    ctx.lineWidth = line;
    ctx.strokeStyle = c.color;
    ctx.stroke();
  } else {
    ctx.fillStyle = c.color;
    ctx.fill();
    ctx.lineWidth = line;
    ctx.strokeStyle = c.ink;
    ctx.stroke();
  }
  ctx.lineJoin = 'miter';
}

/** Draws one mark (see drawMarks). */
export function drawMark(ctx: CanvasRenderingContext2D, cls: number, kind: MarkKind, x: number, y: number, r: number, c: MarkColors): void {
  drawMarks(ctx, cls, kind, [x, y], r, c);
}

const SVG = 'http://www.w3.org/2000/svg';

/**
 * The same mark as a small inline SVG for keys, legends and tables. `color` defaults to the
 * class colour token, so the mark follows the theme. Hidden from assistive tech: the text next
 * to it names the class.
 */
export function markSvg(cls: number, kind: MarkKind, opts: { size?: number; color?: string } = {}): SVGSVGElement {
  const size = opts.size ?? 12;
  const color = opts.color ?? `var(--cat-${((cls % 10) + 10) % 10})`;
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', `mark mark-${markShape(cls)} is-${kind}`);
  svg.dataset.shape = markShape(cls);
  const scale = kind === 'custom' ? CUSTOM_SCALE : 1;
  const pad = kind === 'test' ? 1.2 : kind === 'custom' ? 1.2 : 0.8;
  const r = Math.min(size * 0.36, (size / 2 - pad) / (markExtent(cls) * scale)) * scale;
  const c = size / 2;
  const v = markPolygon(cls, r);
  let el: SVGElement;
  if (!v) {
    el = document.createElementNS(SVG, 'circle');
    el.setAttribute('cx', String(c));
    el.setAttribute('cy', String(c));
    el.setAttribute('r', r.toFixed(2));
  } else {
    el = document.createElementNS(SVG, 'polygon');
    const pts: string[] = [];
    for (let i = 0; i < v.length; i += 2) pts.push(`${(c + v[i]).toFixed(2)},${(c + v[i + 1]).toFixed(2)}`);
    el.setAttribute('points', pts.join(' '));
  }
  const { line } = markStrokes(kind, r);
  el.setAttribute('stroke-linejoin', 'round');
  if (kind === 'test') {
    el.setAttribute('fill', 'none');
    el.setAttribute('stroke', color);
    el.setAttribute('stroke-width', String(Math.max(1.4, line)));
  } else {
    el.setAttribute('fill', color);
    el.setAttribute('stroke', 'var(--ink)');
    el.setAttribute('stroke-width', String(kind === 'custom' ? 1.5 : 0.8));
  }
  svg.append(el);
  return svg;
}
