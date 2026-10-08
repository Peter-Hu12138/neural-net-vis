/** Number formatting shared by the UI. Pure functions, no DOM. */

export const fmt = (v: number, d = 3): string => {
  if (!Number.isFinite(v)) return '—';
  if (v !== 0 && Math.abs(v) < 10 ** -d) return v.toExponential(1);
  return v.toFixed(d);
};

export const pct = (v: number, d = 1): string => `${(v * 100).toFixed(d)}%`;

export const int = (v: number): string => Math.round(v).toLocaleString('en-US');

/** Compact label for a value printed inside a small matrix cell (at most ~5 characters). */
export function formatCell(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.005) return v.toFixed(2).replace(/^(-?)0\./, '$1.');
  return v.toExponential(0);
}
