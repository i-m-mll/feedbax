import { defineConfig, devices } from '@playwright/test';
import { availableParallelism } from 'node:os';
import { requireBrowserTestRunner } from './e2e/runner';

requireBrowserTestRunner();

// Iteration runs the tests tagged neither @round (each starts a job) nor @slow (each takes more than about 4 s of a
// loaded worker, or needs a Studio of its own), so that a lane can run it in about a minute. Round adds both: the
// @round tests on at most `jobWorkers` of the workers, so the jobs they start leave CPU for the tests beside them.
// Release runs only the @responsiveness measurements, on one worker, since a responsiveness budget is read only from
// a run with nothing else beside it.
const tiers = ['iteration', 'round', 'release'] as const;
const tier = (process.env.FEEDBAX_STUDIO_TIER ?? 'iteration') as (typeof tiers)[number];
if (!tiers.includes(tier)) throw new Error(`FEEDBAX_STUDIO_TIER must be one of ${tiers.join(', ')}.`);
const jobWorkers = 3;

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: './node_modules/.cache/studio-e2e-results',
  globalSetup: './e2e/studio/globalSetup.ts',
  fullyParallel: true,
  workers: tier === 'release' ? 1 : Number(process.env.FEEDBAX_STUDIO_E2E_WORKERS ?? Math.max(1, Math.floor(availableParallelism() / 2))),
  // Waits are on state. Their bounds, and each test's, say when a state is surely not coming on a loaded machine, not
  // how fast it comes; a test that starts jobs gets the jobs project's.
  timeout: 60_000,
  expect: { timeout: 20_000, toPass: { timeout: 20_000 } },
  projects: tier === 'release' ? [{ name: 'release', grep: /@responsiveness/ }] : tier === 'iteration'
    ? [{ name: 'interface', grepInvert: /@round|@slow|@responsiveness/ }]
    : [
      { name: 'interface', grepInvert: /@round|@responsiveness/ },
      { name: 'jobs', grep: /@round/, grepInvert: /@responsiveness/, workers: jobWorkers, timeout: 300_000 },
    ],
  use: {
    ...devices['Desktop Chrome'],
    // Worker-scoped options are set here only: specs that differ in one need their own workers,
    // and each worker starts its own Studio.
    headless: true,
    launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
    trace: 'off',
    actionTimeout: 20_000,
  },
});
