import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

interface Task {
  epochs: { id: string; name: string }[];
  signals: { id: string; cells: { id: string; first_epoch: string; last_epoch: string }[] }[];
}

// Catches a Merge that writes more than one cell or loses the span on reload, an absent cell drawn the same as a cell
// holding its absent value (or drawn without the value), a column drag that does not reorder the epochs, and a grid
// that squashes its columns or scrolls its signal names away once five columns do not fit.
test('the epochs grid merges cells, draws an absent cell apart from a zero, reorders columns and scrolls with its names held', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const task = async () => {
    const { project } = (await (await request.get(projectPath)).json()) as StudioProjectSnapshot;
    return (project.documents?.['delayed-reaches'] as { payload: Task } | undefined)?.payload;
  };
  const primary = page.locator('[data-slot="primary"]');
  const grid = primary.getByRole('tabpanel').getByRole('table', { name: 'Epochs', exact: true });
  const row = (name: string) => grid.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const cells = (name: string) => row(name).locator('td[data-cell]');
  const header = (name: string) => grid.locator('th[data-epoch]').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const openTask = async () => {
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Task', exact: true }).click();
  };

  await page.goto(`/?project=${snapshot.project_id}`);
  await openTask();
  await primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Task Document / }).click();
  await page.getByRole('menu', { name: 'Task Document', exact: true }).getByRole('menuitem', { name: 'New', exact: true }).click();
  await page.getByRole('menuitem', { name: /delayed/i }).click();
  await page.keyboard.press('Escape');
  await expect.poll(async () => (await task())?.epochs.map(({ name }) => name)).toEqual(['hold', 'cue', 'move']);

  // target_on is absent in hold, where it reads its absent value 0; its cell over cue and move is given 0.
  await enter(cells('target_on').nth(1).getByRole('textbox'), '0');
  await expect(cells('target_on').nth(1).getByRole('textbox')).toHaveValue('0');
  const absent = cells('target_on').nth(0).getByRole('textbox', { name: 'Absent 0', exact: true });
  await expect(absent).toHaveAttribute('placeholder', '0');
  await expect(absent).toHaveValue('');
  await expect(cells('target_on').nth(1).getByRole('textbox', { name: /^Absent/ })).toHaveCount(0);
  const look = (locator: typeof absent) => locator.evaluate((node) => ({ hatch: getComputedStyle(node).backgroundImage }));
  const absentLook = await look(absent);
  const zeroLook = await look(cells('target_on').nth(1).getByRole('textbox'));
  expect(absentLook.hatch).toContain('gradient');
  expect(zeroLook.hatch).not.toContain('gradient');

  // target_pos is absent in hold and one cell over cue and move: its cells under all three columns merge into one.
  await cells('target_pos').nth(0).click();
  await cells('target_pos').nth(1).click({ modifiers: ['Shift'], position: { x: 4, y: 4 } });
  await page.getByRole('button', { name: 'Cells target_pos', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Merge', exact: true }).click();
  await expect.poll(async () => (await task())?.signals.find(({ id }) => id === 'target_pos')?.cells
    .map(({ first_epoch, last_epoch }) => [first_epoch, last_epoch])).toEqual([['hold', 'move']]);
  await expect(cells('target_pos')).toHaveCount(1);
  await expect(cells('target_pos')).toHaveAttribute('colspan', '3');

  // Dragging cue's handle onto the leading half of hold puts cue first.
  const hold = await header('hold').boundingBox();
  await grid.getByRole('img', { name: 'Move cue', exact: true }).dragTo(header('hold'), {
    targetPosition: { x: 8, y: hold!.height / 2 },
  });
  await expect.poll(async () => (await task())?.epochs.map(({ name }) => name)).toEqual(['cue', 'hold', 'move']);

  for (const name of ['reach', 'settle']) {
    await grid.getByRole('button', { name: 'Add Epoch', exact: true }).click();
    await enter(grid.getByRole('textbox', { name: 'Add Epoch', exact: true }), name);
    await expect(header(name)).toBeVisible();
  }
  await expect.poll(async () => (await task())?.epochs.length).toBe(5);

  // Moving cue out of the merged cell leaves it over hold and move, the cell still stored once.
  await page.reload();
  await openTask();
  await expect(cells('target_pos').locator('xpath=self::*[@colspan="2"]')).toHaveCount(1);
  await expect(cells('target_pos').getByRole('textbox', { name: 'target_pos', exact: true })).toHaveCount(1);
  expect((await task())?.signals.find(({ id }) => id === 'target_pos')?.cells
    .map(({ first_epoch, last_epoch }) => [first_epoch, last_epoch])).toEqual([['hold', 'move']]);
  await expect(grid.locator('th[data-epoch]')).toHaveCount(5);

  await page.setViewportSize({ width: 900, height: 900 });
  const box = grid.locator('xpath=ancestor::*[contains(@class, "studio-table-box")][1]');
  await expect.poll(() => box.evaluate((node) => node.scrollWidth - node.clientWidth)).toBeGreaterThan(0);
  const columnWidth = (await header('hold').boundingBox())!.width;
  const rem = await page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));
  expect(columnWidth).toBeGreaterThanOrEqual(8 * rem - 1);
  const name = row('target_pos').locator('th').first();
  const before = (await name.boundingBox())!.x;
  const settle = (await header('settle').boundingBox())!.x;
  await box.evaluate((node) => { node.scrollLeft = node.scrollWidth; });
  await expect.poll(async () => (await header('settle').boundingBox())!.x).toBeLessThan(settle);
  expect((await name.boundingBox())!.x).toBe(before);
});

// Catches a New from an example that shows the file's key instead of the example's label, an absent cell that takes no
// caret (so only the Inspector could fill it), typed text that is dropped or written as two edits, and a present cell
// emptied of its text that is refused or stored empty instead of becoming absent.
test('a Task from an example shows its label, and typing into an absent cell makes it present', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const saved = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project;
  const spans = async () => ((await saved()).documents?.['delayed-reaches'] as { payload: Task } | undefined)?.payload
    .signals.find(({ id }) => id === 'target_pos')?.cells.map(({ first_epoch, last_epoch }) => [first_epoch, last_epoch]);
  const primary = page.locator('[data-slot="primary"]');
  const picker = primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Task Document / });
  const grid = primary.getByRole('tabpanel').getByRole('table', { name: 'Epochs', exact: true });
  async function command(name: string) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: new RegExp(`^${name}\\b`) }).click();
  }
  const cells = grid.getByRole('row')
    .filter({ has: page.getByRole('textbox', { name: 'Rename target_pos', exact: true }) }).locator('td[data-cell]');

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(picker).toHaveText('Simple reaches');
  await picker.click();
  const menu = page.getByRole('menu', { name: 'Task Document', exact: true });
  await menu.getByRole('menuitem', { name: 'New', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Delayed reaches', exact: true }).click();
  const rename = primary.locator('.studio-window-slot__header').getByRole('textbox');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('Delayed reaches');
  await page.keyboard.press('Escape');
  await expect(picker).toHaveText('Delayed reaches');
  expect((await saved()).documents?.['delayed-reaches']?.label).toBe('Delayed reaches');
  await picker.click();
  await expect(menu.getByRole('menuitemradio')).toHaveText(['Delayed reaches', 'Simple reaches']);
  await page.keyboard.press('Escape');

  const absent = cells.nth(0).getByRole('textbox', { name: /^Absent/ });
  await absent.click();
  await expect(absent).toBeFocused();
  await page.keyboard.type('start');
  await page.keyboard.press('Enter');
  await expect.poll(spans).toEqual([['hold', 'hold'], ['cue', 'move']]);
  await expect(cells.nth(0).getByRole('textbox', { name: 'target_pos', exact: true })).toHaveValue('start');
  await command('Undo');
  await expect.poll(spans).toEqual([['cue', 'move']]);
  await command('Redo');
  await expect.poll(spans).toEqual([['hold', 'hold'], ['cue', 'move']]);

  const held = cells.nth(0).getByRole('textbox', { name: 'target_pos', exact: true });
  await held.fill('');
  await held.press('Enter');
  await expect.poll(spans).toEqual([['cue', 'move']]);
  await expect(cells.nth(0).getByRole('textbox', { name: /^Absent/ })).toBeVisible();
});

// Catches a signal rename that leaves the Model input it fed unfed, and an Outline that does not say what feeds it.
test('renaming a signal in the grid keeps the Model input it feeds', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const grid = page.locator('[data-slot="primary"]').getByRole('tabpanel').getByRole('table', { name: 'Epochs', exact: true });
  await enter(grid.getByRole('textbox', { name: 'Rename target_pos', exact: true }), 'aim');
  await expect(grid.getByRole('textbox', { name: 'Rename aim', exact: true })).toBeVisible();
  const input = page.getByRole('complementary', { name: 'Details Drawer', exact: true })
    .getByRole('tree', { name: 'Model Inputs', exact: true }).getByRole('treeitem').filter({ hasText: 'target_pos' });
  await expect(input).toContainText('aim');
  await expect(input.locator('[data-status-mark]')).toHaveCount(0);
});
