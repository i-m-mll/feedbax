import { expect, type APIRequestContext, type Locator, type Page, type TestInfo } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { StudioProjectDocument } from '../../../src/generated/studioContracts';
import { activate, activateMenuItem, chooseStepKind, number } from './studioKeyboard';

const repo = fileURLToPath(new URL('../../../../', import.meta.url));
const run = promisify(execFile);

export async function runProjectFixture(
  fixture: string | { module: string; home: string }, ...arguments_: string[]
) {
  const module = typeof fixture === 'string' ? fixture : fixture.module;
  const env = typeof fixture === 'string' ? process.env : { ...process.env, FEEDBAX_STUDIO_TEST_HOME: fixture.home };
  try {
    const { stdout } = await run('uv', [
      'run', '--no-sync', 'python', '-B', '-m', module, ...arguments_,
    ], { cwd: repo, env, timeout: 12_000 });
    return stdout;
  } catch (error) {
    const failure = error as Error & { code?: number | string; signal?: string; killed?: boolean; stderr?: string };
    throw new Error(`${failure.message}\nExit ${failure.code ?? 'none'}, signal ${failure.signal ?? 'none'}, killed ${failure.killed ?? false}\n${failure.stderr ?? ''}`,
      { cause: error });
  }
}

export async function fixture(...arguments_: string[]) {
  return JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_train', ...arguments_));
}

export async function seedManagedProject(request: APIRequestContext, files: readonly unknown[], origin = '') {
  const response = await request.post(origin + '/api/projects');
  expect(response.status()).toBe(201);
  const { snapshot, configuration } = await response.json() as StudioProjectDocument;
  return { ...await appendManagedFiles(request, snapshot, files, origin), configuration };
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
  return page.locator(`[data-map-step="${stepId}"] button[data-canvas-focus]`);
}

export async function chooseTrainSuccessor(page: Page) {
  await chooseStepKind(page, 'Train');
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

export async function authorLearningRateSweep(rate: Locator) {
  const page = rate.page();
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activateMenuItem(page, page.getByRole('menuitem', { name: 'Sweep', exact: true }));
  const text = rate.getByRole('textbox', { name: 'Learning Rate', exact: true });
  await expect(text).toHaveValue('0.001,');
  await expect(text).toBeEditable();
  await number(page, text, '0.001, 0.002, 0.003');
  for (const value of ['0.001', '0.002', '0.003']) await expect(rate.getByText(value, { exact: true })).toBeVisible();
}
