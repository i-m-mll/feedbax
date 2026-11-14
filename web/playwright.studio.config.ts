import { defineConfig, devices } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const port = Number(process.env.FEEDBAX_STUDIO_E2E_PORT ?? 3109);
const backendPort = Number(process.env.FEEDBAX_STUDIO_E2E_API_PORT ?? 8109);
const baseURL = `http://127.0.0.1:${port}`;
const ownsHome = process.env.FEEDBAX_STUDIO_TEST_HOME === undefined;
const home = process.env.FEEDBAX_STUDIO_TEST_HOME ??= join(tmpdir(), `feedbax-studio-test-${randomUUID()}`);
const preview = process.env.FEEDBAX_STUDIO_E2E_PREVIEW === '1' ? ' --preview' : '';
const dispose = ownsHome ? ' --remove-test-home' : '';
const seriesFixture = process.env.FEEDBAX_STUDIO_E2E_SERIES_FIXTURE === '1';
const launcher = seriesFixture
  ? 'cd .. && exec uv run --offline --no-sync python -B -m web.e2e.studio.fixtures.evaluate_series_backend'
  : 'exec ../scripts/dev.sh';

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: mkdtempSync(join(tmpdir(), 'feedbax-studio-results-')),
  metadata: { home, backendPort },
  grepInvert: preview ? undefined : /@responsiveness/,
  workers: 1,
  timeout: 30_000,
  use: {
    ...devices['Desktop Chrome'],
    channel: process.env.FEEDBAX_STUDIO_E2E_CHROME === '1' ? 'chrome' : undefined,
    baseURL,
    headless: true,
    actionTimeout: 5_000,
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `${launcher} --home "$FEEDBAX_STUDIO_TEST_HOME" --port ${port} --backend-port ${backendPort}${preview}${dispose}`,
    env: { FEEDBAX_STUDIO_TEST_OWNER_PID: String(process.pid) },
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    gracefulShutdown: { signal: 'SIGTERM', timeout: 30_000 },
    timeout: 55_000,
  },
});
