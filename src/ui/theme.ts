/** Reads the CSS colour tokens so canvas drawings follow the page theme. */

export interface Palette {
  bg: string;
  surface: string;
  ink: string;
  ink2: string;
  muted: string;
  hair: string;
  accent: string;
  neg: string;
  /** Class colours --cat-0 … --cat-9 (class k uses cat[k % 10]). */
  cat: string[];
  rgb: { surface: RGB; ink: RGB; accent: RGB; neg: RGB; hair: RGB; cat: RGB[] };
}

export type RGB = [number, number, number];

let cached: Palette | null = null;
const listeners = new Set<() => void>();

function parse(c: string): RGB {
  const s = c.trim();
  if (s.startsWith('#')) {
    const hex = s.length === 4 ? s.slice(1).split('').map((x) => x + x).join('') : s.slice(1, 7);
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
  }
  const m = s.match(/[\d.]+/g);
  return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [0, 0, 0];
}

export function palette(): Palette {
  if (cached) return cached;
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim() || '#000';
  const p = {
    bg: v('--bg'),
    surface: v('--surface'),
    ink: v('--ink'),
    ink2: v('--ink-2'),
    muted: v('--muted'),
    hair: v('--hair'),
    accent: v('--accent'),
    neg: v('--neg'),
  };
  const cat = Array.from({ length: 10 }, (_, k) => cs.getPropertyValue(`--cat-${k}`).trim() || p.ink);
  cached = {
    ...p,
    cat,
    rgb: { surface: parse(p.surface), ink: parse(p.ink), accent: parse(p.accent), neg: parse(p.neg), hair: parse(p.hair), cat: cat.map(parse) },
  };
  return cached;
}

/** Colour of class `k` in the current theme. */
export const classColor = (k: number): string => palette().cat[k % 10];

export function onThemeChange(fn: () => void): void {
  listeners.add(fn);
}

function invalidate() {
  cached = null;
  for (const fn of listeners) fn();
}

matchMedia('(prefers-color-scheme: dark)').addEventListener('change', invalidate);
new MutationObserver(invalidate).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Diverging map for signed values in [−1, 1]: blue · surface · red. */
export function diverging(v: number, out: RGB = [0, 0, 0]): RGB {
  const p = palette().rgb;
  const t = Math.max(-1, Math.min(1, v));
  const c = t >= 0 ? mix(p.surface, p.accent, Math.pow(t, 0.8)) : mix(p.surface, p.neg, Math.pow(-t, 0.8));
  out[0] = c[0];
  out[1] = c[1];
  out[2] = c[2];
  return out;
}

/** Sequential map for [0, 1]: surface → ink (white paper, black ink). */
export function sequential(v: number, out: RGB = [0, 0, 0]): RGB {
  const p = palette().rgb;
  const t = Math.max(0, Math.min(1, v));
  out[0] = p.surface[0] + (p.ink[0] - p.surface[0]) * t;
  out[1] = p.surface[1] + (p.ink[1] - p.surface[1]) * t;
  out[2] = p.surface[2] + (p.ink[2] - p.surface[2]) * t;
  return out;
}

export const css = (c: RGB, a = 1) => (a >= 1 ? `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})` : `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`);
