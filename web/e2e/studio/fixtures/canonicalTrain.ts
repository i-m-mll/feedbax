import { expect, type APIRequestContext, type Locator, type Page, type TestInfo } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { StudioProjectDocument } from '../../../src/generated/studioContracts';

const repo = fileURLToPath(new URL('../../../../', import.meta.url));
const run = promisify(execFile);

export async function runProjectFixture(module: string, ...arguments_: string[]) {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', '-m', module, ...arguments_,
  ], { cwd: repo, timeout: 12_000 });
  return stdout;
}

export async function fixture(...arguments_: string[]) {
  return JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_train', ...arguments_));
}

export async function seedManagedProject(request: APIRequestContext, files: readonly unknown[], origin = '') {
  const response = await request.post(origin + '/api/projects');
  expect(response.status()).toBe(201);
  const { snapshot } = await response.json() as StudioProjectDocument;
  return appendManagedFiles(request, snapshot, files, origin);
}

export async function appendManagedFiles(
  request: APIRequestContext, initial: StudioProjectDocument['snapshot'], files: readonly unknown[], origin = '',
) {
  let snapshot = initial;
  const projectPath = `/api/projects/${snapshot.project_id}`;
  for (const file of files) {
    const saved = await request.post(`${origin}${projectPath}/files`, { data: { revision: snapshot.revision, file } });
    expect(saved.status(), await saved.text()).toBe(201);
    ({ snapshot } = await saved.json() as StudioProjectDocument);
  }
  return { snapshot, projectPath };
}

export function mapStep(page: Page, stepId: string) {
  return page.locator('[data-map-card]').getByRole('button', { name: `Select ${stepId}`, exact: true });
}

export async function captureStudioWindow(page: Page, info: TestInfo, name: string) {
  const path = info.outputPath(`${name}.png`);
  await page.screenshot({ path, animations: 'disabled' });
  await info.attach(name, { path, contentType: 'image/png' });
}

export async function openResultsSlot(page: Page) {
  const split = page.getByRole('button', { name: 'Split Details', exact: true });
  if (await split.isVisible()) await split.click();
  await page.locator('[data-slot="secondary"] > section').focus();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('tab', { name: 'Results', exact: true }).click();
}

export async function enter(field: Locator, value: string) {
  await field.fill(value);
  await field.press('Enter');
}

export async function openSweepEditor(field: Locator, name: string) {
  const edit = field.getByRole('button', { name: `Edit ${name}`, exact: true });
  await expect(edit).toBeEnabled();
  if (await edit.getAttribute('aria-expanded') !== 'true') await edit.click();
}

export async function authorLearningRateSweep(rate: Locator) {
  await rate.getByRole('button', { name: 'To Sweep Learning Rate', exact: true }).click();
  await openSweepEditor(rate, 'Learning Rate 0.001');
  await expect(rate.getByRole('spinbutton')).toHaveValue('0.001');
  await enter(rate.getByRole('textbox'), 'Slow');
  await expect(rate.getByRole('spinbutton', { name: 'Edit Learning Rate Slow' })).toBeEnabled();
  for (const [index, [value, label]] of [['0.002', 'Medium'], ['0.003', 'Fast']].entries()) {
    await rate.getByRole('button', { name: 'Add Value Learning Rate' }).click();
    const edits = rate.getByRole('button', { name: /^Edit Learning Rate / });
    await expect(edits).toHaveCount(index + 2);
    await edits.last().click();
    await enter(rate.getByRole('spinbutton'), value);
    await expect(rate.getByRole('spinbutton', { name: `Edit Learning Rate ${value}`, exact: true })).toBeEnabled();
    await enter(rate.getByRole('textbox'), label);
    await expect(rate.getByRole('spinbutton', { name: `Edit Learning Rate ${label}`, exact: true })).toBeEnabled();
  }
}
