import { expect, type APIRequestContext, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { StudioProjectDocument } from '../../../src/generated/studioContracts';

const repo = fileURLToPath(new URL('../../../../', import.meta.url));
const run = promisify(execFile);

export async function fixture(...arguments_: string[]) {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', '-m', 'web.e2e.studio.fixtures.canonical_train', ...arguments_,
  ], { cwd: repo, timeout: 12_000 });
  return JSON.parse(stdout);
}

export async function seedManagedProject(request: APIRequestContext, files: readonly unknown[]) {
  const response = await request.post('/api/projects');
  expect(response.status()).toBe(201);
  let { snapshot } = await response.json() as StudioProjectDocument;
  const projectPath = `/api/projects/${snapshot.project_id}`;
  for (const file of files) {
    const saved = await request.post(`${projectPath}/files`, { data: { revision: snapshot.revision, file } });
    expect(saved.status(), await saved.text()).toBe(201);
    ({ snapshot } = await saved.json() as StudioProjectDocument);
  }
  return { snapshot, projectPath };
}

export function mapStep(page: Page, stepId: string) {
  return page.locator('[data-map-card]').getByRole('button', { name: `Select ${stepId}`, exact: true });
}

export async function openResultsSlot(page: Page) {
  const split = page.getByRole('button', { name: 'Split View', exact: true });
  if (await split.isVisible()) await split.click();
  await page.locator('[data-slot="secondary"] > section').focus();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('button', { name: 'Results', exact: true }).click();
}
