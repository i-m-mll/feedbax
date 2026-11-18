import { defineConfig, devices } from '@playwright/test';
import { availableParallelism } from 'node:os';
import { requireBrowserTestRunner } from './e2e/runner';

requireBrowserTestRunner();

// Iteration runs the tests that start no job; round adds those tagged @round; release adds @responsiveness.
const tiers = { iteration: /@round|@responsiveness/, round: /@responsiveness/, release: undefined };
const tier = (process.env.FEEDBAX_STUDIO_TIER ?? 'iteration') as keyof typeof tiers;
if (!(tier in tiers)) throw new Error(`FEEDBAX_STUDIO_TIER must be one of ${Object.keys(tiers).join(', ')}.`);

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: './node_modules/.cache/studio-e2e-results',
  globalSetup: './e2e/studio/globalSetup.ts',
  grepInvert: tiers[tier],
  fullyParallel: true,
  workers: Number(process.env.FEEDBAX_STUDIO_E2E_WORKERS ?? Math.max(1, Math.floor(availableParallelism() / 2))),
  timeout: 30_000,
  use: {
    ...devices['Desktop Chrome'],
    // Worker-scoped options are set here only: specs that differ in one need their own workers,
    // and each worker starts its own Studio.
    headless: true,
    launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
    trace: 'off',
    actionTimeout: 5_000,
    serviceWorkers: 'block',
  },
});
