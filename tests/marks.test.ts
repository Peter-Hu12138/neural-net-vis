import { describe, expect, it } from 'vitest';
import { CUSTOM_SCALE, MARK_SHAPES, drawMarks, markExtent, markPolygon, markShape, markStrokes, polygonArea, traceMark } from '../src/ui/marks';

/** Records the path calls a mark makes. */
function recorder() {
  const calls: { op: string; args: number[] }[] = [];
  const style: Record<string, unknown> = {};
  const ctx = new Proxy(style, {
    get(target, op: string) {
      if (op in target) return target[op];
      return (...args: number[]) => calls.push({ op, args });
    },
    set(target, op: string, v) {
      target[op] = v;
      calls.push({ op: `set ${op}`, args: [] });
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
  return { ctx, calls, style };
}

describe('class marks', () => {
  it('gives the first classes distinct shapes, cycling with the colours', () => {
    expect(markShape(0)).toBe('circle');
    expect(markShape(1)).toBe('square');
    expect(markShape(2)).toBe('triangle');
    expect(markShape(3)).toBe('diamond');
    expect(new Set(MARK_SHAPES).size).toBe(10);
    expect(markShape(13)).toBe(markShape(3)); // class k uses --cat-(k % 10) and the same shape
  });

  it('draws every shape with the area of a circle of the same radius, centred on the point', () => {
    const r = 4;
    for (let k = 1; k < 10; k++) {
      const v = markPolygon(k, r)!;
      expect(v, markShape(k)).not.toBeNull();
      expect(Math.abs(polygonArea(v)), markShape(k)).toBeCloseTo(Math.PI * r * r, 6);
      // Centroid of the vertices at the point (all shapes are symmetric about it).
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < v.length; i += 2) {
        cx += v[i];
        cy += v[i + 1];
      }
      expect(cx / (v.length / 2)).toBeCloseTo(0, 6);
      expect(cy / (v.length / 2)).toBeCloseTo(0, 6);
      // The reach of the shape is what markExtent reports.
      let far = 0;
      for (let i = 0; i < v.length; i += 2) far = Math.max(far, Math.hypot(v[i], v[i + 1]));
      expect(far).toBeCloseTo(markExtent(k) * r, 6);
    }
    expect(markPolygon(0, r)).toBeNull(); // the circle
    expect(markExtent(0)).toBe(1);
  });

  it('winds every outline clockwise on screen, so a batch of overlapping marks fills as one', () => {
    for (let k = 1; k < 10; k++) expect(polygonArea(markPolygon(k, 3)!), markShape(k)).toBeGreaterThan(0);
  });

  it('tells the four point-data shapes apart by outline, not only by colour', () => {
    // Vertex counts: circle (arc), square 4, triangle 3, diamond 4 but rotated 45° from the square.
    const sq = markPolygon(1, 1)!;
    const tri = markPolygon(2, 1)!;
    const dia = markPolygon(3, 1)!;
    expect(sq.length / 2).toBe(4);
    expect(tri.length / 2).toBe(3);
    expect(dia.length / 2).toBe(4);
    // The square has horizontal edges, the diamond a vertex straight up.
    expect(sq[1]).toBeCloseTo(sq[3], 9);
    expect(dia[0]).toBeCloseTo(0, 9);
    expect(dia[1]).toBeLessThan(0);
    // The triangle points up: one vertex above the centre, two below it.
    expect(tri.filter((_, i) => i % 2 === 1 && tri[i] < 0)).toHaveLength(1);
  });

  it('traces each mark as one closed sub-path at the point', () => {
    const { ctx, calls } = recorder();
    traceMark(ctx, 0, 10, 20, 3);
    expect(calls.map((c) => c.op)).toEqual(['moveTo', 'arc', 'closePath']);
    expect(calls[1].args.slice(0, 3)).toEqual([10, 20, 3]);
    const t = recorder();
    traceMark(t.ctx, 2, 10, 20, 3);
    expect(t.calls.map((c) => c.op)).toEqual(['moveTo', 'lineTo', 'lineTo', 'closePath']);
    const v = markPolygon(2, 3)!;
    expect(t.calls[0].args[0]).toBeCloseTo(10 + v[0], 9);
    expect(t.calls[0].args[1]).toBeCloseTo(20 + v[1], 9);
  });

  it('fills training marks, outlines test marks on a halo, and draws a batch in one path', () => {
    const colors = { color: 'C', ink: 'I', surface: 'S' };
    const train = recorder();
    drawMarks(train.ctx, 1, 'train', [0, 0, 10, 10, 20, 20], 3, colors);
    const ops = train.calls.map((c) => c.op);
    expect(ops.filter((o) => o === 'beginPath')).toHaveLength(1);
    expect(ops.filter((o) => o === 'moveTo')).toHaveLength(3);
    expect(ops.filter((o) => o === 'fill')).toHaveLength(1);
    expect(ops.filter((o) => o === 'stroke')).toHaveLength(1);
    const test = recorder();
    drawMarks(test.ctx, 1, 'test', [0, 0, 10, 10], 3, colors);
    const tops = test.calls.map((c) => c.op);
    expect(tops).not.toContain('fill');
    expect(tops.filter((o) => o === 'stroke')).toHaveLength(2); // halo, then the class colour
    expect(test.style.strokeStyle).toBe('C');
    // Your points: larger, heavier outline.
    const mine = recorder();
    drawMarks(mine.ctx, 0, 'custom', [5, 5], 3, colors);
    expect(mine.calls.find((c) => c.op === 'arc')!.args[2]).toBeCloseTo(3 * CUSTOM_SCALE, 9);
    expect(markStrokes('custom', 3).line).toBeGreaterThan(markStrokes('train', 3).line);
    const s = markStrokes('test', 3);
    expect(s.halo).toBeGreaterThan(s.line);
  });
});
