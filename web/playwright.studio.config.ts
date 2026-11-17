import { defineConfig, devices } from '@playwright/test';
import { availableParallelism } from 'node:os';

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: './node_modules/.cache/studio-e2e-results',
  globalSetup: './e2e/studio/globalSetup.ts',
  grepInvert: process.env.FEEDBAX_STUDIO_MEASURE === '1' ? undefined : /@responsiveness/,
  fullyParallel: true,
  workers: Number(process.env.FEEDBAX_STUDIO_E2E_WORKERS ?? Math.max(1, Math.floor(availableParallelism() / 2))),
  timeout: 30_000,
  use: {
    ...devices['Desktop Chrome'],
    headless: true,
    actionTimeout: 5_000,
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
  },
});
