import { expect, test, type Page } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

function submitted(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('the slot header switches, creates, duplicates and renames a step\'s documents in place', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('history-files'));
  const runsPath = `${projectPath}/train-runs`;
  const table = page.getByRole('table', { name: 'Runs' });
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const header = primary.locator('.studio-window-slot__header');
  const rename = header.getByRole('textbox');
  const picker = (label: string) => header.getByRole('button', { name: new RegExp(`^${label} Document `) });
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function rows() {
    const { revision } = await saved();
    const response = await request.get(`${runsPath}?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  async function uses(step: string) {
    const base = (await saved()).project.steps?.[step]?.base;
    return base && 'inline' in base ? (base.inline as { trial_documents: Record<string, { document: string }> }).trial_documents : undefined;
  }
  const status = async (step: string) => (await rows()).rows.find(({ key }) => key.step === step)?.status;
  async function run(step: string) {
    await mapStep(page, step).click();
    await table.getByRole('button').first().click();
    const admission = submitted(page, runsPath);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admission;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0]?.disposition).toBe('admitted');
    await expect.poll(() => status(step), { timeout: 20_000 }).toBe('succeeded');
  }

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await run('train');
  await run('sibling');
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();

  await expect(header.getByRole('heading', { name: 'Model', exact: true })).toBeVisible();
  await expect(header.getByRole('heading')).toHaveText('Model');
  await expect(header.getByRole('button')).toHaveCount(2);
  await expect(picker('Model')).toHaveText('point-mass');
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Model')).toHaveText('point-mass1 of 2');
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Model')).toHaveText('point-mass');

  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(header.getByRole('heading')).toHaveText('Task');
  await expect(picker('Task')).toHaveText('simple-reaches-task');
  const taskMenu = page.getByRole('menu', { name: 'Task Document', exact: true });
  await picker('Task').click();
  await taskMenu.getByRole('menuitem', { name: 'Duplicate', exact: true }).click();
  await expect(rename).toBeFocused();
  const copy = await rename.inputValue();
  expect(copy).not.toBe('simple-reaches-task');
  await page.keyboard.press('Escape');
  await expect(picker('Task')).toHaveText(copy);
  await expect.poll(async () => [(await uses('train'))?.task.document, (await uses('sibling'))?.task.document])
    .toEqual([copy, 'simple-reaches-task']);
  expect(await status('sibling')).toBe('succeeded');

  await picker('Task').press('Enter');
  await expect(taskMenu.getByRole('menuitemradio')).toHaveText(['simple-reaches-task', copy]);
  await taskMenu.getByRole('menuitemradio', { name: 'simple-reaches-task', exact: true }).press('Enter');
  await expect(picker('Task')).toHaveText('simple-reaches-task');
  await expect.poll(async () => (await uses('train'))?.task.document).toBe('simple-reaches-task');

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await expect(picker('Objective')).toHaveText('simple-reaches-objective');
  await picker('Objective').press('Enter');
  await page.keyboard.press('End');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('objective');
  await page.keyboard.type('fresh');
  await page.keyboard.press('Enter');
  await expect(picker('Objective')).toHaveText('fresh');
  await expect.poll(async () => (await uses('train'))?.objective.document).toBe('fresh');
  await picker('Objective').click();
  await page.getByRole('menuitemradio', { name: 'simple-reaches-objective', exact: true }).click();
  await expect.poll(async () => (await uses('train'))?.objective.document).toBe('simple-reaches-objective');

  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await picker('Model').focus();
  await page.keyboard.press('Enter');
  const menu = page.getByRole('menu', { name: 'Model Document', exact: true });
  await expect(menu.getByRole('menuitemradio')).toHaveText(['point-mass']);
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('point-mass');
  await page.keyboard.press('Escape');
  await expect(picker('Model')).toHaveText('point-mass');
  await picker('Model').press('Enter');
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  await page.keyboard.type('arm');
  await page.keyboard.press('Enter');
  await expect(picker('Model')).toHaveText('arm');
  await expect.poll(async () => [(await uses('train'))?.model.document, (await uses('sibling'))?.model.document])
    .toEqual(['arm', 'arm']);
  expect((await rows()).window_map?.problems).toEqual([]);
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(picker('Model')).toHaveText('arm');

  await page.getByRole('tab', { name: 'Scene', exact: true }).click();
  await expect(picker('Model')).toHaveText('arm');

  await header.getByRole('button', { name: 'Split Details', exact: true }).click();
  const slots = page.locator('[data-slot]');
  await expect(slots.getByRole('button', { name: 'Close Slot', exact: true })).toHaveCount(2);
  await page.locator('[data-slot="primary"]').getByRole('button', { name: 'Close Slot', exact: true }).click();
  await expect(slots).toHaveCount(1);
  await expect(header.getByRole('button', { name: 'Split Details', exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});
