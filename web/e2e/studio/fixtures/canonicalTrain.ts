import { expect, type APIRequestContext, type Locator, type Page, type TestInfo } from '@playwright/test';
import type { StudioProjectDocument } from '../../../src/generated/studioContracts';
import { runPython } from './python';
import { activate, activateMenuItem, chooseStepKind, number } from './studioKeyboard';


export async function runProjectFixture(
  fixture: string | { module: string; home: string }, ...arguments_: string[]
) {
  if (typeof fixture === 'string') return runPython({ module: fixture }, arguments_);
  return runPython({ module: fixture.module }, arguments_, { FEEDBAX_STUDIO_TEST_HOME: fixture.home });
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
  await expect(rate.getByRole('textbox', { name: 'Learning Rate value 3', exact: true })).toHaveValue('0.003');
  for (const [index, name] of ['Slow', 'Medium', 'Fast'].entries()) {
    const valueName = rate.getByRole('textbox', { name: `Learning Rate value ${index + 1} name`, exact: true });
    await expect(valueName).toBeEditable();
    await number(page, valueName, name);
    await expect(rate.getByRole('textbox', { name: `Value ${name}`, exact: true })).toBeVisible();
  }
}
