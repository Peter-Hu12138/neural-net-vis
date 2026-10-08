import { defineConfig } from '@playwright/test';

// E2E_PORT / E2E_DIST let several builds be tested side by side.
const port = Number(process.env.E2E_PORT ?? 4173);
const dist = process.env.E2E_DIST ?? 'dist';

/** Browser tests run against the production build (`npm run test:e2e` builds first). */
export default defineConfig({
  testDir: 'e2e',
  timeout: 120_000,
  expect: { timeout: 20_000 },
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://localhost:${port}`,
    viewport: { width: 1600, height: 1000 },
    browserName: 'chromium',
  },
  webServer: {
    command: `npx vite preview --outDir ${dist} --port ${port} --strictPort`,
    url: `http://localhost:${port}`,
    reuseExistingServer: true,
  },
});
