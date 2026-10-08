import { describe, expect, it } from 'vitest';
import { fmt, formatCell, int, pct } from '../src/ui/format';

describe('format', () => {
  it('prints matrix cells in five characters or fewer', () => {
    expect(formatCell(0)).toBe('0');
    expect(formatCell(0.1234)).toBe('.12');
    expect(formatCell(-0.1234)).toBe('-.12');
    expect(formatCell(1.234)).toBe('1.23');
    expect(formatCell(12.34)).toBe('12.3');
    expect(formatCell(123.4)).toBe('123');
    expect(formatCell(0.0004)).toBe('4e-4');
    for (const v of [0.5, -0.99, 3.14159, -42.5, 0.00071]) expect(formatCell(v).length).toBeLessThanOrEqual(5);
  });

  it('formats stats for the control bar', () => {
    expect(fmt(0.123456)).toBe('0.123');
    expect(fmt(0.00001)).toBe('1.0e-5');
    expect(fmt(NaN)).toBe('—');
    expect(pct(0.9734)).toBe('97.3%');
    expect(int(20000)).toBe('20,000');
  });
});
