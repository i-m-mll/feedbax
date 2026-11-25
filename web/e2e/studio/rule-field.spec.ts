import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StepReference, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { chooseSweepValue, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { hold } from './fixtures/heldRequest';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const primary = (page: Page) => page.locator('[data-studio-focus-region="primary"]');
const runRows = (page: Page) => primary(page).getByRole('table', { name: 'Runs', exact: true }).getByRole('rowheader');

async function select(page: Page, step: string) {
  await mapStep(page, step).click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
}

// Catches a rule field that saves while it is written, a count or boxes drawn from the saved rule rather than the one
// being written, a rule the check cannot read that Enter drops instead of saving, its Problem lost on reopening or
// left after its repair, checkpoint text that is dropped or offered To Sweep, To Rows offered on an unranked rule, and
// a field for Evaluate subjects that Train Start lacks.
test('a run rule narrows an Evaluate subject and a Train Start, and a rule naming a name that is not swept is kept with its Problem', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'walkthrough-files'));
  const train = files.at(-1).step;
  train.base.inline.seed = 0;
  train.sweep = { axes: { seed: { path: 'seed', values: [0, 1, 2, 3] } } };
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const project = async () => (await (await request.get(projectPath)).json() as StudioProjectSnapshot).project;
  const reads = async (step: string, role: string) => (await project()).steps![step].inputs![role] as StepReference;
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const rule = inspector.getByRole('combobox', { name: 'Runs', exact: true });
  const ruleRow = rule.locator('..');
  const problem = ruleRow.getByRole('img', { name: 'Problem', exact: true });
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  const saving = () => page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === projectPath);

  await page.goto(`/?project=${snapshot.project_id}`);
  await select(page, 'train');
  await card.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Evaluate');
  await expect(rule).toHaveValue('');
  await expect(ruleRow).toContainText('4/4');

  await rule.fill('seed=0..1');
  await expect(ruleRow).toContainText('2/4');
  await expect(card.locator('.studio-map-run-mark[data-boxed]')).toHaveCount(2);
  expect((await reads('evaluate', 'subject')).where).toBeUndefined();
  const saved = saving();
  await rule.press('Enter');
  expect((await saved).ok()).toBe(true);
  await expect.poll(async () => (await reads('evaluate', 'subject')).where).toEqual({ rule: 'seed=0..1' });
  await expect(runRows(page)).toHaveCount(2);

  await page.reload();
  await select(page, 'evaluate');
  await expect(rule).toHaveValue('seed=0..1');
  await expect(ruleRow).toContainText('2/4');
  await rule.fill('sed=0');
  const kept = saving();
  await rule.press('Enter');
  expect((await kept).ok()).toBe(true);
  await expect.poll(async () => (await reads('evaluate', 'subject')).where).toEqual({ rule: 'sed=0' });
  await expect(rule).toHaveAttribute('aria-invalid', 'true');
  await expect(problem).toHaveAttribute('data-app-tip', /sed is not swept/);

  await select(page, 'train');
  await page.reload();
  await select(page, 'evaluate');
  await expect(rule).toHaveValue('sed=0');
  await expect(rule).toHaveAccessibleDescription(/sed is not swept/);
  await expect(problem).toHaveAttribute('data-app-tip', /sed is not swept/);
  await rule.fill('seed=0..1');
  const repaired = saving();
  await rule.press('Enter');
  expect((await repaired).ok()).toBe(true);
  await expect(problem).toHaveCount(0);
  await expect(rule).not.toHaveAttribute('aria-invalid', 'true');
  await expect(ruleRow).toContainText('2/4');
  await expect(runRows(page)).toHaveCount(2);

  const checkpoint = inspector.getByRole('region', { name: 'Checkpoint', exact: true });
  const updates = checkpoint.getByRole('textbox', { name: 'Updates', exact: true });
  await pick(checkpoint.getByRole('combobox', { name: 'Checkpoint', exact: true }), 'List');
  await updates.fill('2 x');
  const held = saving();
  await updates.press('Enter');
  expect((await held).ok()).toBe(true);
  await expect.poll(async () => (await reads('evaluate', 'subject')).checkpoint).toEqual({ kind: 'unresolved', text: '2 x' });
  await page.reload();
  await select(page, 'evaluate');
  const toSweep = checkpoint.getByRole('button', { name: 'To Sweep', exact: true });
  await expect(updates).toHaveValue('2 x');
  await expect(updates).toHaveAttribute('aria-invalid', 'true');
  await expect(toSweep).toHaveCount(0);
  await updates.fill('2');
  const fixed = saving();
  await updates.press('Enter');
  expect((await fixed).ok()).toBe(true);
  await expect.poll(async () => (await reads('evaluate', 'subject')).checkpoint).toEqual([{ at: 2 }]);
  await expect(updates).not.toHaveAttribute('aria-invalid', 'true');
  await expect(toSweep).toBeEnabled();
  await rule.click({ button: 'right' });
  await expect(page.getByRole('menuitem', { name: 'Pick Runs', exact: true })).toBeVisible();
  await expect(page.getByRole('menuitem', { name: 'To Rows', exact: true })).toHaveCount(0);
  await page.keyboard.press('Escape');

  await select(page, 'train');
  await card.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await expect(mapStep(page, 'train-2')).toHaveAttribute('aria-pressed', 'true');
  await expect(runRows(page)).toHaveCount(4);
  await rule.fill('seed=3');
  await expect(ruleRow).toContainText('1/4');
  const started = saving();
  await rule.press('Enter');
  expect((await started).ok()).toBe(true);
  await expect.poll(async () => (await reads('train-2', 'checkpoint')).where).toEqual({ rule: 'seed=3' });
  await expect(runRows(page)).toHaveCount(1);
});

// Catches the new step's first field taking the keyboard when its form lands after the user has started writing the
// Train Start rule: the rule is then saved on that blur, or dropped, and Enter saves nothing.
test('a rule written before the new step\'s form lands keeps the keyboard and saves on Enter', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'walkthrough-files'));
  const train = files.at(-1).step;
  train.base.inline.seed = 0;
  train.sweep = { axes: { seed: { path: 'seed', values: [0, 1, 2, 3] } } };
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const reads = async (step: string, role: string) =>
    ((await (await request.get(projectPath)).json() as StudioProjectSnapshot).project.steps![step].inputs![role] as StepReference);
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const rule = inspector.getByRole('combobox', { name: 'Runs', exact: true });
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  const form = `${projectPath}/authoring/train/train-2`;

  await page.goto(`/?project=${snapshot.project_id}`);
  await select(page, 'train');
  const held = await hold(page, form, (_, sent) => sent.method() === 'POST');
  await card.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await held.sent;
  await rule.fill('seed=3');
  await expect(rule.locator('..')).toContainText('1/4');
  const landed = page.waitForResponse((response) => new URL(response.url()).pathname === form);
  held.release();
  await landed;
  await expect(inspector.getByRole('combobox', { name: 'Method', exact: true })).toBeVisible();
  await expect(rule).toBeFocused();
  await expect(rule).toHaveValue('seed=3');
  expect((await reads('train-2', 'checkpoint')).where).toBeUndefined();
  const started = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === projectPath);
  await rule.press('Enter');
  expect((await started).ok()).toBe(true);
  await expect.poll(async () => (await reads('train-2', 'checkpoint')).where).toEqual({ rule: 'seed=3' });
});

// Catches a stored rule that chooses no run explained by a sentence under the Subject picker instead of on the rule
// field, or its text replaced.
test('a stored rule that asks for a run that does not exist keeps its text and is marked on the field', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'missing-run-files'));
  const { snapshot } = await seedManagedProject(request, files);
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const rule = inspector.getByRole('combobox', { name: 'Runs', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await select(page, 'evaluate');
  await expect(rule).toHaveValue('seed=7');
  await expect(rule).toHaveAttribute('aria-invalid', 'true');
  const mark = rule.locator('..').getByRole('img', { name: 'Problem', exact: true });
  await expect(mark).toHaveAttribute('data-app-tip', /seed=7/);
  await expect(rule).toHaveAccessibleDescription(/seed=7/);
  await expect(inspector.getByRole('alert')).toHaveCount(0);
});

/** A Train swept over `axes`, with a new Evaluate reading it selected and its Runs field shown. */
async function pickingProject(page: Page, request: APIRequestContext, axes: Record<string, { path: string; values: number[] }>) {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'walkthrough-files'));
  const train = files.at(-1).step;
  train.base.inline.seed = 0;
  train.sweep = { axes };
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const rule = page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('combobox', { name: 'Runs', exact: true });
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await select(page, 'train');
  await card.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Evaluate');
  await expect(mapStep(page, 'evaluate')).toHaveAttribute('aria-pressed', 'true');
  await rule.click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Pick Runs', exact: true }).click();
  const subject = async () => ((await (await request.get(projectPath)).json() as StudioProjectSnapshot)
    .project.steps!.evaluate.inputs!.subject as StepReference).where;
  return { rule, card, subject };
}

// Catches a press on the card that selects the producer's runs instead of picking them, a rule that is not the picked
// runs' own, and picks that are not saved on Enter.
test('values picked on the producer\'s card pin their IDs on Enter', async ({ page, request }) => {
  const { rule, card, subject } = await pickingProject(page, request, { seed: { path: 'seed', values: [0, 1, 2, 3] } });
  await chooseSweepValue(card, 'seed', '0');
  await chooseSweepValue(card, 'seed', '2');
  await expect(rule).toHaveValue('');
  await expect(rule.locator('..')).toContainText('Rows');
  await expect(rule.locator('..')).toContainText('2/4');
  await expect(card.locator('.studio-map-run-mark[data-boxed]')).toHaveCount(2);
  await expect(mapStep(page, 'evaluate')).toHaveAttribute('aria-pressed', 'true');
  await expect(rule).toBeFocused();
  await rule.press('Enter');
  await expect.poll(subject).toMatchObject({ clauses: [{ seed: expect.arrayContaining([
    { id: expect.stringMatching(/^value-/), label: '0' }, { id: expect.stringMatching(/^value-/), label: '2' },
  ]) }] });
  await expect(runRows(page)).toHaveCount(2);
});

// Catches a diagonal pick that widens to the whole rectangle, and Escape saving the picks.
test('diagonal picks save two clauses, and Escape keeps the saved rule', async ({ page, request }) => {
  const { rule, card, subject } = await pickingProject(page, request, {
    seed: { path: 'seed', values: [0, 1] }, batches: { path: 'training_config.n_batches', values: [2, 3] },
  });
  const mark = (seed: number, batches: number) => card.getByRole('button', { name: new RegExp(`seed ${seed} · batches ${batches}\\b`) });
  await mark(0, 2).click();
  await expect(rule.locator('..')).toContainText('Rows');
  await page.keyboard.press('Escape');
  await expect(rule).toHaveValue('');
  expect(await subject()).toBeUndefined();

  await rule.click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Pick Runs', exact: true }).click();
  await mark(0, 2).click();
  await mark(1, 3).click();
  await expect(rule).toHaveValue('');
  for (const [seed, batches] of [[0, 2], [1, 3]]) await expect(mark(seed, batches)).toHaveAttribute('data-boxed', '');
  await expect(card.locator('.studio-map-run-mark[data-boxed]')).toHaveCount(2);
  await expect(rule.locator('..')).toContainText('2/4');
  await rule.press('Enter');
  const value = (label: string) => [{ id: expect.stringMatching(/^value-/), label }];
  await expect.poll(subject).toEqual({ clauses: expect.arrayContaining([
    { seed: value('0'), batches: value('2') }, { seed: value('1'), batches: value('3') },
  ]) });
  await expect(runRows(page)).toHaveCount(2);
});
