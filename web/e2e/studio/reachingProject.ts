import { expect, type APIRequestContext } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { GraphCreateResponse, StudioPersistenceDocument } from '../../src/generated/studioContracts';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);

export async function reachingDocument(batches = 30, name = 'Point-mass reach'): Promise<StudioPersistenceDocument> {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', 'web/e2e/studio/fixtures/reaching_project.py',
    '--batches', String(batches), '--name', name,
  ], { cwd: repo, timeout: 10_000 });
  return JSON.parse(stdout);
}

export async function saveFixture(request: APIRequestContext, document: StudioPersistenceDocument) {
  const response = await request.post('/api/graphs', { data: document });
  expect(response.ok(), await response.text()).toBe(true);
  const { data } = await response.json() as GraphCreateResponse;
  return data;
}
