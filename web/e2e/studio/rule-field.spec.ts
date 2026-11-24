import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StepReference, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

const primary = (page: Page) => page.locator('[data-studio-focus-region="primary"]');
const runRows = (page: Page) => primary(page).getByRole('table', { name: 'Runs', exact: true }).getByRole('rowheader');

async function select(page: Page, step: string) {
  await mapStep(page, step).click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
}

// Catches a rule field that saves while it is written, a count or boxes drawn from the saved rule rather than the one
// being written, a refused rule that is stored anyway, and a field for Evaluate subjects that Train Start lacks.
test('a run rule narrows an Evaluate subject and a Train Start, and a name that is not swept is refused on the field', async ({ page, request }) => {
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
  await rule.press('Enter');
  await expect(inspector.getByRole('alert').filter({ hasText: 'sed is not swept' })).toBeVisible();
  await expect(rule).toHaveAttribute('aria-invalid', 'true');
  expect((await reads('evaluate', 'subject')).where).toEqual({ rule: 'seed=0..1' });
  await expect(runRows(page)).toHaveCount(2);
  await page.keyboard.press('Escape');
  await expect(rule).toHaveValue('seed=0..1');

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
