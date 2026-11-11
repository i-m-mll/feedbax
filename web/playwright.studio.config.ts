import { defineConfig, devices } from '@playwright/test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const port = Number(process.env.FEEDBAX_STUDIO_E2E_PORT ?? 3109);
const backendPort = Number(process.env.FEEDBAX_STUDIO_E2E_API_PORT ?? 8109);
const baseURL = `http://127.0.0.1:${port}`;
const ownsHome = process.env.FEEDBAX_STUDIO_TEST_HOME === undefined;
const home = process.env.FEEDBAX_STUDIO_TEST_HOME ??= mkdtempSync(join(tmpdir(), 'feedbax-studio-test-'));
const preview = process.env.FEEDBAX_STUDIO_E2E_PREVIEW === '1' ? ' --preview' : '';

export default defineConfig({
  testDir: './e2e/studio',
  outputDir: mkdtempSync(join(tmpdir(), 'feedbax-studio-results-')),
  globalTeardown: './e2e/studio/teardown.ts',
  metadata: { home, ownsHome, backendPort },
  workers: 1,
  timeout: 30_000,
  use: {
    ...devices['Desktop Chrome'],
    baseURL,
    actionTimeout: 5_000,
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `../scripts/dev.sh --home "$FEEDBAX_STUDIO_TEST_HOME" --port ${port} --backend-port ${backendPort}${preview}`,
    env: { FEEDBAX_STUDIO_TEST_OWNER_PID: String(process.pid) },
    url: `${baseURL}/api/provider/health`,
    reuseExistingServer: false,
    gracefulShutdown: { signal: 'SIGTERM', timeout: 10_000 },
    timeout: 55_000,
  },
});
