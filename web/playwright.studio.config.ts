import { defineConfig, devices } from '@playwright/test';
import { mkdtempSync } from 'node:fs';
import { availableParallelism, tmpdir } from 'node:os';
import { join } from 'node:path';

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: mkdtempSync(join(tmpdir(), 'feedbax-studio-results-')),
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
