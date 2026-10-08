import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

/** Minimal PNG reader for the 8-bit greyscale sprites written by scripts/build-datasets.py. */
function readPng(path: string) {
  const buf = readFileSync(path);
  expect(buf.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  let off = 8;
  let width = 0;
  let height = 0;
  const idat: Buffer[] = [];
  const chunks: string[] = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    chunks.push(type);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      expect(data[8]).toBe(8); // bit depth
      expect(data[9]).toBe(0); // greyscale
    }
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const px = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    expect(raw[y * (width + 1)]).toBe(0); // filter: none
    px.set(raw.subarray(y * (width + 1) + 1, (y + 1) * (width + 1)), y * width);
  }
  return { width, height, px, chunks };
}

const digit = (sprite: ReturnType<typeof readPng>, i: number) => {
  const out = new Uint8Array(784);
  const ox = (i % 100) * 28;
  const oy = Math.floor(i / 100) * 28;
  for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) out[r * 28 + c] = sprite.px[(oy + r) * sprite.width + ox + c];
  return out;
};

describe('bundled MNIST subset', () => {
  const text = readFileSync('public/data/mnist-labels.txt', 'utf8').trim();
  const labels = Uint8Array.from(text, (ch) => ch.charCodeAt(0) - 48);

  it('has 20,000 training and 2,000 test labels, all digits 0–9', () => {
    expect(text).toMatch(/^[0-9]{22000}$/);
    expect(labels.length).toBe(22_000);
    const counts = new Array(10).fill(0);
    for (const y of labels) counts[y]++;
    expect(counts.every((c) => c > 1500)).toBe(true);
  });

  it('starts the test split with the canonical MNIST labels', () => {
    expect(Array.from(labels.subarray(20_000, 20_010))).toEqual([7, 2, 1, 0, 4, 1, 4, 9, 5, 9]);
    expect(Array.from(labels.subarray(0, 10))).toEqual([5, 0, 4, 1, 9, 2, 1, 3, 1, 4]);
  });

  it('stores sprites as plain greyscale PNGs with no colour-management chunks', () => {
    for (const [file, rows] of [['mnist-train-0.png', 50], ['mnist-train-3.png', 50], ['mnist-test.png', 20]] as const) {
      const png = readPng(`public/data/${file}`);
      expect(png.width).toBe(2800);
      expect(png.height).toBe(rows * 28);
      expect(png.chunks.filter((c) => !['IHDR', 'IDAT', 'IEND'].includes(c))).toEqual([]);
    }
  });

  it('holds real digits: ink in the middle, empty borders', () => {
    const png = readPng('public/data/mnist-test.png');
    for (let i = 0; i < 2000; i += 97) {
      const d = digit(png, i);
      let border = 0;
      let centre = 0;
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          if (r === 0 || r === 27 || c === 0 || c === 27) border += d[r * 28 + c];
          else centre += d[r * 28 + c];
        }
      }
      expect(border).toBeLessThan(0.02 * centre);
      expect(centre).toBeGreaterThan(255 * 20);
    }
  });
});
