import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative asset URLs so the build works from any sub-path (GitHub Pages, file servers).
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022' },
  // Unit tests only; the Playwright browser suite lives in e2e/.
  test: { include: ['tests/**/*.test.ts'] },
});
