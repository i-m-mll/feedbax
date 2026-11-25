import { expect, type APIRequestContext, type Locator, type Page, type TestInfo } from '@playwright/test';
import type { StudioProjectDocument } from '../../../src/generated/studioContracts';
import { createHash } from 'node:crypto';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { runPython } from './python';
import { activate, activateMenuItem, chooseStepKind, number, testProjectName } from './studioKeyboard';


/** Runs a pure project-file generator (an action named `…files`) once per arguments and Python source. */
async function generatedFiles(module: string, arguments_: readonly string[]) {
  const root = process.env.FEEDBAX_STUDIO_E2E_FIXTURES;
  if (root === undefined) throw new Error('The Studio global setup did not create the fixture cache');
  const path = join(root, createHash('sha256').update(JSON.stringify([module, ...arguments_])).digest('hex') + '.json');
  const cached = await readFile(path, 'utf8').catch(() => undefined);
  if (cached !== undefined) return cached;
  const output = await runPython({ module }, arguments_);
  const staging = `${path}.${process.pid}`;
  await writeFile(staging, output);
  await rename(staging, path);
  return output;
}

export async function runProjectFixture(module: string, ...arguments_: string[]) {
  if (/(^|-)files$/.test(arguments_[0] ?? '')) return generatedFiles(module, arguments_);
  return runPython({ module }, arguments_);
}

export async function fixture(...arguments_: string[]) {
  return JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_train', ...arguments_));
}

export async function seedManagedProject(request: APIRequestContext, files: readonly unknown[], origin = '') {
  const response = await request.post(origin + '/api/projects', { data: { name: testProjectName() } });
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

/** A Runs table row's accessible name: its ordinal, then the run's label. */
export function runName(label: string) {
  return new RegExp(`^\\d+ ${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`);
}

type RowKey = { readonly variant: string; readonly step: string; readonly row: string };

/** The name the Runs table draws for a saved row of a swept step, which always has a label. */
export function shownRunLabel(row: { readonly key: RowKey; readonly label: string | null }) {
  if (row.label === null) throw new Error(`Run ${JSON.stringify(row.key)} has no label; read its step's name with drawnRunLabel.`);
  return row.label;
}

type DrawnMap = {
  readonly rows: readonly { readonly key: RowKey; readonly label: string | null }[];
  readonly steps: readonly { readonly variant: string; readonly step_id: string; readonly label: string }[];
};

/** The name the Runs table draws for the saved row at `key`: its label, or for a step's only run its step's name. */
export function drawnRunLabel(map: DrawnMap, key: RowKey) {
  const row = map.rows.find((item) => item.key.variant === key.variant && item.key.step === key.step && item.key.row === key.row)!;
  return row.label ?? drawnStepLabel(map, key);
}

/** The name the window map gives a run's step. */
export function drawnStepLabel(map: Pick<DrawnMap, 'steps'>, key: Omit<RowKey, 'row'>) {
  return map.steps.find((step) => step.variant === key.variant && step.step_id === key.step)!.label;
}

/** A step's card on the map, found by its id since the card reads the step's name. */
export function mapCard(scope: Page | Locator, stepId: string) {
  return scope.locator(`[data-map-card][data-map-step="${stepId}"]`);
}

export function mapStep(page: Page, stepId: string) {
  return page.locator(`[data-map-step="${stepId}"] button[data-canvas-focus]`);
}

/** A sweep's label on a map card, which opens the list of its values. */
export function sweepLabel(card: Locator, sweep: string) {
  return card.getByRole('button', { name: new RegExp(`^Sweep ${sweep} \\d+$`) });
}

/** Chooses a value from a sweep's label on a map card; with Shift, adds it to the chosen values. */
export async function chooseSweepValue(card: Locator, sweep: string, value: string, extend = false) {
  await sweepLabel(card, sweep).click();
  await card.page().getByRole('menuitem', { name: value, exact: true }).click({ modifiers: extend ? ['Shift'] : [] });
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

/** Sweeps the learning rate over three values; `store` is the Apply To choice when another run reads the step. */
export async function authorLearningRateSweep(rate: Locator, store?: string) {
  const page = rate.page();
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activateMenuItem(page, page.getByRole('menuitemradio', { name: 'Sweep', exact: true }));
  const text = rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true });
  await expect(text).toBeFocused();
  await expect(text).toHaveValue('0.001');
  await number(page, text, '0.001, 0.002, 0.003');
  if (store) await activateMenuItem(page, page.getByRole('menu', { name: 'Apply To', exact: true }).getByRole('menuitem', { name: store, exact: true }));
  for (const value of ['0.001', '0.002', '0.003']) await expect(rate.getByText(value, { exact: true })).toBeVisible();
}
