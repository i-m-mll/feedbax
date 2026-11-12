import { expect, test, type Locator, type Page } from '@playwright/test';
import type { StudioJobSubmission, StudioProjectDocument, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const region = (page: Page, id: string) => page.locator(`[data-studio-focus-region="${id}"]:visible`);
const command = (page: Page, id: string) => page.locator(`button[data-command-id="${id}"]:visible`).first();

async function tabTo(page: Page, target: Locator, limit = 100) {
  await expect(target).toBeVisible();
  for (let index = 0; index < limit; index += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press('Tab');
  }
  await expect(target).toBeFocused();
}

async function activate(page: Page, target: Locator, key = 'Enter') {
  await tabTo(page, target);
  await page.keyboard.press(key);
}

async function palette(page: Page, id: string, label: string) {
  await page.keyboard.press('Meta+K');
  const dialog = page.getByRole('dialog', { name: 'Command Palette' });
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.type(label);
  const item = dialog.getByRole('option', { name: label, exact: true });
  await expect(item).toHaveAttribute('data-command-id', id);
  await expect(item).toBeEnabled();
  await page.keyboard.press('Enter');
  await expect(dialog).toBeHidden();
}

async function number(page: Page, target: Locator, value: string) {
  await expect(target).toBeEnabled();
  await tabTo(page, target);
  await page.keyboard.press('Meta+A');
  await page.keyboard.type(value);
  await expect(target).toHaveValue(value);
  await page.keyboard.press('Enter');
}

function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('the window commands, tab rail and selection history work without a pointer', async ({ page, request }) => {
  const [model, task, train, successor] = await fixture('files');
  const { snapshot } = await seedManagedProject(request, [model, task, train, successor, { ...train, id: 'third' }]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('F6');
  await expect(region(page, 'top')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'map')).toBeFocused();
  await activate(page, mapStep(page, 'successor'));
  await activate(page, mapStep(page, 'third'));

  await page.keyboard.press('Meta+BracketLeft');
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await palette(page, 'back', 'Back');
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Meta+BracketRight');
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await palette(page, 'forward', 'Forward');
  await expect(mapStep(page, 'third')).toHaveAttribute('aria-pressed', 'true');

  await page.keyboard.press('Control+Meta+1');
  await expect(command(page, 'showMap')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'primary')).toHaveCount(0);
  await palette(page, 'showBoth', 'Show Both');
  await expect(command(page, 'showBoth')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'map')).toBeFocused();
  await page.keyboard.press('Control+Meta+3');
  await expect(command(page, 'showDetails')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'primary')).toBeFocused();
  await expect(mapStep(page, 'third')).toBeHidden();
  await page.keyboard.press('Control+Meta+2');

  await page.keyboard.press('Control+Meta+S');
  await expect(command(page, 'showSteps')).toHaveAccessibleName('Show Steps');
  await expect(region(page, 'steps')).toHaveCount(0);
  await page.keyboard.press('Control+Meta+S');
  await expect(command(page, 'showSteps')).toHaveAccessibleName('Hide Steps');
  await expect(region(page, 'steps')).toBeFocused();

  await page.keyboard.press('F6');
  const tabs = page.getByRole('tablist', { name: 'Details' });
  await expect(tabs).toHaveAttribute('aria-orientation', 'vertical');
  await expect(tabs.getByRole('tab', { name: 'Runs', exact: true })).toBeFocused();
  await page.keyboard.press('ArrowDown');
  const modelTab = tabs.getByRole('tab', { name: 'Model', exact: true });
  await expect(modelTab).toBeFocused();
  await expect(page.getByRole('tooltip')).toHaveText('Model · ⌘2');
  await expect(modelTab).not.toHaveAttribute('title', /.+/);
  await page.keyboard.press('Enter');
  await expect(modelTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel', { name: 'Model', exact: true })).toBeFocused();

  await page.keyboard.press('Meta+0');
  await expect(region(page, 'drawer')).toHaveCount(0);
  await palette(page, 'showDrawer', 'Show Drawer');
  await expect(region(page, 'drawer')).toBeFocused();
  await activate(page, region(page, 'drawer').getByRole('button', { name: 'plant', exact: true }));
  await expect(region(page, 'inspector').getByRole('spinbutton', { name: 'Edit Mass' })).toBeVisible();
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toHaveCount(0);
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'bottom')).toBeFocused();
  await page.keyboard.press('Shift+F6');
  await expect(region(page, 'inspector')).toBeFocused();

  await page.keyboard.press('Meta+Backslash');
  await expect(region(page, 'secondary')).toBeFocused();
  await expect(page.getByRole('tabpanel', { name: 'Scene', exact: true })).toBeVisible();
  await page.keyboard.press('Alt+Meta+Backslash');
  await expect(region(page, 'primary')).toBeFocused();
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  await palette(page, 'switchSlot', 'Switch Slot');
  await expect(region(page, 'secondary')).toBeFocused();
  await palette(page, 'splitDetails', 'Close Slot');
  await expect(region(page, 'secondary')).toHaveCount(0);
  await expect(region(page, 'primary')).toBeFocused();

  await page.keyboard.press('Meta+K');
  const dialog = page.getByRole('dialog', { name: 'Command Palette' });
  await page.keyboard.press('Shift+Tab');
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(region(page, 'primary')).toBeFocused();
});

test('the keyboard authors, runs and stops a strict checkpoint continuation and follows Results', async ({ page, request }) => {
  test.setTimeout(55_000);
  const { snapshot, projectPath } = await seedManagedProject(request, (await fixture('files')).slice(0, 3));
  const runsPath = `${projectPath}/train-runs`;
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('Meta+5');
  const rate = page.getByRole('group', { name: 'Learning Rate', exact: true });
  await activate(page, rate.getByRole('button', { name: 'To Sweep Learning Rate', exact: true }));
  for (const [index, value] of ['0.002', '0.003'].entries()) {
    await page.keyboard.press('Meta+5');
    const add = rate.getByRole('button', { name: 'Add Value Learning Rate', exact: true });
    await expect(add).toBeEnabled();
    await activate(page, add);
    await expect(rate.getByRole('spinbutton')).toHaveCount(index + 2);
    await page.keyboard.press('Meta+5');
    await number(page, rate.getByRole('spinbutton').last(), value);
    await expect(rate.getByRole('spinbutton', { name: `Edit Learning Rate ${value}`, exact: true })).toBeEnabled();
  }
  await page.keyboard.press('Control+Meta+2');
  await activate(page, card.getByRole('button', { name: 'Sweep learning-rate', exact: true }));
  const value = (name: string) => card.getByRole('listitem').getByRole('button', { name: `Select learning-rate ${name}`, exact: true });
  await activate(page, value('0.001'));
  await activate(page, value('0.003'), 'Shift+Enter');
  await expect(value('0.001')).toHaveAttribute('aria-pressed', 'true');
  await expect(value('0.003')).toHaveAttribute('aria-pressed', 'true');
  await expect(value('0.002')).toHaveAttribute('aria-pressed', 'false');
  await expect(card).toContainText('2 of 3');

  const prefix = post(page, runsPath);
  await page.keyboard.press('Meta+R');
  const sourceRun = await prefix;
  expect(sourceRun.status(), await sourceRun.text()).toBe(202);
  const sourceKeys = sourceRun.request().postDataJSON().run_keys as StudioSavedTrainRuns['rows'][number]['key'][];
  expect(sourceKeys).toHaveLength(2);
  await expect(card.getByText('2 Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await expect(card.getByText('1 Not run', { exact: true })).toBeVisible();

  await expect(card.getByRole('button', { name: 'New Step train' })).toHaveAttribute('data-command-id', 'newStep');
  const create = post(page, `${projectPath}/files`);
  await palette(page, 'newStep', 'New Step');
  const created = await create;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: authored } = await created.json() as StudioProjectDocument;
  const child = 'train-continuation';
  const plan = await (await request.get(`${runsPath}?revision=${authored.revision}&window_map=true`)).json() as StudioSavedTrainRuns;
  const selected = plan.window_map!.rows.filter(({ key }) => key.step === 'train' && sourceKeys.some(({ row }) => row === key.row));
  expect(selected.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.003]);
  expect(authored.project.steps?.[child]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: {
      axes: { 'learning-rate': selected.map(({ coordinates }) => coordinates['learning-rate'].value_id) },
    } } },
  });
  expect(plan.rows.filter(({ key }) => key.step === child).map(({ key }) => key.row)).toEqual(sourceKeys.map(({ row }) => row));

  const runs = page.getByRole('region', { name: 'Runs', exact: true });
  await expect(runs.getByRole('combobox', { name: 'Edit Model' })).toBeFocused();
  const batches = runs.getByRole('spinbutton', { name: 'Edit N Batches' });
  await number(page, batches, '1200');
  await expect(batches).toBeEnabled();
  await page.keyboard.press('Meta+1');
  const table = runs.getByRole('table', { name: 'Runs' });
  const row = table.getByRole('row').filter({ has: page.getByRole('cell', { name: '0.001', exact: true }) }).getByRole('button');
  await activate(page, row);
  await expect(row).toHaveAttribute('aria-current', 'true');
  const continuation = post(page, runsPath);
  await palette(page, 'run', 'Run');
  const admitted = await continuation;
  expect(admitted.status(), await admitted.text()).toBe(202);
  expect(admitted.request().postDataJSON().run_keys).toEqual([{ ...sourceKeys[0], step: child }]);
  const job = (await admitted.json() as StudioJobSubmission).items[0].job_id!;
  await expect(runs.getByRole('button', { name: 'Open Run', exact: true })).toBeEnabled();
  await palette(page, 'openRun', 'Open Run');
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  await expect(results).toBeFocused();
  await expect(results.getByRole('img', { name: 'Loss' }).locator('line title').filter({ hasText: job }).first()).toBeAttached({ timeout: 16_000 });
  const stopped = post(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stop = await stopped;
  expect(stop.status(), await stop.text()).toBe(202);
  expect(stop.request().postDataJSON()).toEqual({ job_ids: [job], mode: 'checkpoint' });
  await page.keyboard.press('Meta+1');
  await expect(table.getByText('Stopped', { exact: true })).toBeVisible({ timeout: 12_000 });
  await expect(table.getByText('Not run', { exact: true })).toBeVisible();
  await page.keyboard.press('Meta+7');
  await expect(results).toBeFocused();
  await expect(results.getByRole('img', { name: 'Loss' })).toBeVisible();
});
