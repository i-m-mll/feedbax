import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches a New that gives anything but a one-epoch task, a variable that is not written once named, a type or value
// that does not stay on its row, a rename that breaks the names reading it, completion that misses the task's
// variables, a name that does not resolve marked as the whole value or with a sentence under the field, and a table
// that a reload does not give back.
test('the Variables table names, types and values a new task\'s variables, and marks a name that does not resolve', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const saved = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project;
  const primary = page.locator('[data-slot="primary"]');
  const header = primary.locator('.studio-window-slot__header');
  const table = primary.getByRole('tabpanel').getByRole('table', { name: 'Variables', exact: true });
  const row = (name: string) => table.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const value = (name: string) => row(name).getByRole('textbox', { name: 'Value', exact: true });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await header.getByRole('button', { name: /^Task Document / }).click();
  await page.getByRole('menu', { name: 'Task Document', exact: true }).getByRole('menuitem', { name: 'New', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Blank', exact: true }).click();
  await page.keyboard.press('Escape');
  const task = async () => {
    const project = await saved();
    const entry = Object.entries(project.documents ?? {}).find(([name]) => /^task(-\d+)?$/.test(name));
    return entry?.[1] as { payload: { epochs: unknown[] } } | undefined;
  };
  await expect.poll(async () => (await task())?.payload.epochs.length).toBe(1);
  await expect(table.getByRole('row')).toHaveCount(2);

  const add = async (name: string, type: string, text: string) => {
    await table.getByRole('button', { name: 'Add Variable', exact: true }).click();
    await enter(table.getByRole('textbox', { name: 'Add Variable', exact: true }), name);
    await expect(row(name)).toBeVisible();
    if (type !== 'Number') {
      await row(name).getByRole('combobox', { name: 'Type', exact: true }).click();
      await page.getByRole('option', { name: type, exact: true }).click();
      await expect(row(name).getByRole('combobox', { name: 'Type', exact: true })).toHaveText(type);
    }
    await enter(value(name), text);
    await expect(value(name)).toHaveValue(text);
  };
  await add('start', 'Point', '[0 m, 0 m]');
  await add('direction', 'Number', '0.5');
  await add('reach', 'Distance', '0.15 m');

  await table.getByRole('button', { name: 'Add Variable', exact: true }).click();
  await enter(table.getByRole('textbox', { name: 'Add Variable', exact: true }), 'target');
  await row('target').getByRole('combobox', { name: 'Type', exact: true }).click();
  await page.getByRole('option', { name: 'Point', exact: true }).click();
  await value('target').fill('start + polar(');
  await value('target').pressSequentially('re');
  await expect(page.getByRole('listbox', { name: 'Completions' }).getByRole('option')).toContainText(['reach']);
  await page.keyboard.press('Tab');
  await page.keyboard.type(', 0.5 rad)');
  await page.keyboard.press('Enter');
  await expect(value('target')).toHaveValue(/polar\(reach, 0\.5 rad\)/);
  await expect(row('target').locator('[data-field-problem], [data-field-term]')).toHaveCount(0);
  await expect(table.getByRole('row')).toHaveCount(6);

  const name = table.getByRole('textbox', { name: 'Rename reach', exact: true });
  await name.click();
  await name.fill('radius');
  await name.press('Enter');
  await expect(value('target')).toHaveValue(/polar\(radius, 0\.5 rad\)/);
  await expect(row('target').locator('[data-field-problem], [data-field-term]')).toHaveCount(0);

  await enter(value('direction'), 'scale * 2');
  const term = row('direction').locator('[data-field-term]');
  await expect(term).toHaveText('scale');
  await expect(term).toHaveAttribute('data-app-tip', /scale/);
  const failure = await page.evaluate(() => {
    const probe = document.createElement('span');
    probe.className = 'text-shell-status-failure';
    document.body.append(probe);
    const color = getComputedStyle(probe).color;
    probe.remove();
    return color;
  });
  await expect(term).toHaveCSS('color', failure);
  await expect(row('direction').locator('[data-field-problem]')).toHaveCount(0);
  await expect(primary.getByRole('alert')).toHaveCount(0);

  const shown = async () => Promise.all(['start', 'direction', 'radius', 'target'].map((name) => value(name).inputValue()));
  await page.keyboard.press('Escape');
  const before = await shown();
  expect(before.slice(0, 3)).toEqual(['[0 m, 0 m]', '0.5', '0.15 m']);
  expect(before[3]).toMatch(/polar\(radius, 0\.5 rad\)/);
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(value('target')).toHaveValue(before[3]);
  expect(await shown()).toEqual(before);
  await expect(row('radius').getByRole('combobox', { name: 'Type', exact: true })).toHaveText('Distance');
});
