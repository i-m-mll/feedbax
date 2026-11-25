import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { pick, reveal } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) =>
  page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

// Catches: a Synthesize step that New cannot make, or that is authored in a form the Service does not read, so its
// Method and settings are lost.
test('New makes a Synthesize step whose Method and settings are edited in its inspector', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const saved = async () => (await (await request.get(projectPath)).json() as StudioProjectSnapshot).project.steps?.synthesize;
  await page.goto(`/?project=${snapshot.project_id}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Synthesize');
  await expect(page.locator('[data-map-step="synthesize"]')).toContainText('Constant parameters');
  expect(await saved()).toMatchObject({ kind: 'synthesize', base: { inline: {
    model: { document: expect.any(String) }, task: { document: expect.any(String) },
    objective: { document: expect.any(String) }, method: 'feedbax.constant_parameters',
  } } });

  await mapStep(page, 'synthesize').click();
  await tab(page, 'Runs').click();
  await reveal(page, 'Inspector');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('combobox', { name: 'Method', exact: true })).toHaveValue('Constant parameters');
  const value = inspector.getByRole('textbox', { name: 'Value', exact: true });
  await value.fill('2');
  await value.press('Enter');
  await expect.poll(async () => (await saved())?.base).toMatchObject({ inline: { settings: { value: 2 } } });
});

// Catches: a synthesized model an Evaluate or a Train start cannot read, or one read as a Train checkpoint, so the
// Evaluate's result names no origin.
test('a Synthesize step on the shipped Model is run, read by an Evaluate made from it, and starts a Train', { tag: '@round' }, async ({ page, request, studio }) => {
  test.setTimeout(90_000);
  const project: string = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.synthesis', 'seed', studio.home, 'synthesis'));
  const projectPath = `/api/projects/${project}`;
  const runsPath = `${projectPath}/train-runs`;
  const saved = async () => await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  const step = async (id: string) => (await saved()).project.steps?.[id];
  async function status(id: string) {
    const { revision } = await saved();
    const response = await request.get(`${runsPath}?window_map=true&revision=${revision}`);
    const runs = await response.json() as StudioSavedTrainRuns;
    return runs.window_map?.rows.find(({ key }) => key.step === id)?.status;
  }
  async function run(id: string) {
    await mapStep(page, id).click();
    await tab(page, 'Runs').click();
    await page.getByRole('table', { name: 'Runs', exact: true }).getByRole('button').first().click();
    const admission = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === runsPath);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admission;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0]?.disposition).toBe('admitted');
    await expect.poll(() => status(id), { timeout: 30_000 }).toBe('succeeded');
  }
  const header = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header');
  const picker = (label: string) => header.getByRole('button', { name: new RegExp(`^${label} Document `) });

  await page.goto(`/?project=${project}`);
  const train = mapStep(page, 'simple-reaches');
  if (await train.getAttribute('aria-pressed') === 'true') await train.click({ modifiers: ['ControlOrMeta'] });
  await expect(train).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Synthesize');
  await expect(mapStep(page, 'synthesize')).toBeVisible();
  await mapStep(page, 'synthesize').click();
  await tab(page, 'Model').click();
  await expect(picker('Model')).toHaveText('Point mass');
  for (const [view, label] of [['Task', 'Task'], ['Objective', 'Objective']] as const) {
    await tab(page, view).click();
    await picker(label).click();
    await page.getByRole('menuitemradio', { name: /^Simple reaches/ }).click();
  }
  await expect.poll(async () => (await step('synthesize'))?.base).toMatchObject({ inline: {
    model: { document: 'point-mass' }, task: { document: 'simple-reaches-task' },
    objective: { document: 'simple-reaches-objective' },
  } });
  await run('synthesize');

  await mapStep(page, 'synthesize').click();
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Evaluate');
  await expect.poll(async () => (await step('evaluate'))?.inputs).toEqual({ subject: { step: 'synthesize' } });
  await run('evaluate');
  await tab(page, 'Results').click();
  const results = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  await expect(results.getByText('Synthesized', { exact: true })).toBeVisible();
  await expect(results.getByRole('button', { name: 'Synthesize', exact: true })).toBeVisible();
  await expect(tab(page, 'Scene')).toBeEnabled();

  await mapStep(page, 'simple-reaches').click();
  await tab(page, 'Runs').click();
  await pick(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('combobox', { name: 'Start', exact: true }), 'Synthesize');
  await expect.poll(async () => JSON.stringify(await step('simple-reaches'))).toContain('"step":"synthesize"');
  await run('simple-reaches');
});
