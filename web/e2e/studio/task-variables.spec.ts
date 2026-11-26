import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches a New that gives anything but a one-epoch task, a variable that is not written once named, a type or value
// that does not stay on its row, a rename that breaks the names reading it, completion that misses the task's
// variables, a value naming what is not yet declared that is refused rather than stored and marked on its term (an
// edit the user cannot leave half-done), a mark that declaring the name does not clear, Escape during a save or the
// read after it showing the replaced value, and a table that a reload does not give back.
test('the Variables table names, types and values a new task\'s variables, and stores a name that does not resolve with its mark', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const saved = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project;
  const primary = page.locator('[data-slot="primary"]');
  const header = primary.locator('.studio-window-slot__header');
  const table = primary.getByRole('tabpanel').getByRole('table', { name: 'Variables', exact: true });
  const row = (name: string) => table.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const value = (name: string) => row(name).getByRole('textbox', { name: 'Value', exact: true });
  const variables = table.getByRole('row').filter({ has: page.getByRole('textbox', { name: /^Rename / }) });

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
  await expect(variables).toHaveCount(0);

  const add = async (name: string, type: string, text: string) => {
    await table.getByRole('button', { name: 'Add Variable', exact: true }).click();
    await enter(table.getByRole('textbox', { name: 'Add Variable', exact: true }), name);
    await expect(row(name)).toBeVisible();
    if (type !== 'Number') {
      // The value is entered while the Type saves, and is stored after it.
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
  await expect(variables).toHaveCount(4);

  const name = table.getByRole('textbox', { name: 'Rename reach', exact: true });
  await name.click();
  await name.fill('radius');
  await name.press('Enter');
  await expect(value('target')).toHaveValue(/polar\(radius, 0\.5 rad\)/);
  await expect(row('target').locator('[data-field-problem], [data-field-term]')).toHaveCount(0);

  const problemColour = await page.evaluate(() => {
    const probe = document.createElement('span');
    probe.className = 'text-shell-status-failure';
    document.body.append(probe);
    const color = getComputedStyle(probe).color;
    probe.remove();
    return color;
  });
  const term = row('target').locator('[data-field-term]');
  const reopen = async () => {
    await page.reload();
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Task', exact: true }).click();
  };
  await enter(value('target'), 'start + offset');
  await expect(term).toHaveText('offset');
  await expect(term).toHaveCSS('color', problemColour);
  await expect(row('target').locator('[data-field-problem]')).toHaveCount(0);
  await expect(primary.getByRole('alert')).toHaveCount(0);
  await reopen();
  await expect(value('target')).toHaveValue('start + offset');
  await expect(term).toHaveText('offset');
  await expect(term).toHaveCSS('color', problemColour);

  // A value entered while its Type is saving is stored after it. Escape while the committed value saves, and the read
  // of the stored view, leave it shown, not the value it replaces. The value's save waits for the Type's, which settles
  // with its own read; only the reads after the value's are held.
  let holding: 'save' | 'read' | undefined = 'save';
  let release!: () => void;
  let held = new Promise<void>((resolve) => { release = resolve; });
  let sent!: () => void;
  const sending = new Promise<void>((resolve) => { sent = resolve; });
  let releaseType!: () => void;
  const typing = new Promise<void>((resolve) => { releaseType = resolve; });
  await page.route('**/authoring/trials', async (route) => {
    const body = route.request().postData() ?? '';
    const save = holding === 'save' && body.includes('/task/variables/offset/value');
    if (save) sent();
    if (body.includes('"set_type"')) await typing;
    if (save || holding === 'read' && !body.includes('"change"')) await held;
    await route.fallback();
  });
  await add('offset', 'Point', '[0.1 m, 0 m]');
  releaseType();
  await page.keyboard.press('Escape');
  await sending;
  await expect(value('offset')).toHaveValue('[0.1 m, 0 m]');
  const releaseSave = release;
  held = new Promise<void>((resolve) => { release = resolve; });
  holding = 'read';
  releaseSave();
  const stored = async (name: string) => JSON.stringify(((await task())?.payload as unknown as { variables: { name: string; value: unknown }[] })
    .variables.find((variable) => variable.name === name)?.value);
  await expect.poll(() => stored('offset')).toContain('"value":0.1,');
  await expect(value('offset')).toHaveValue('[0.1 m, 0 m]');
  holding = undefined;
  release();
  await expect(row('target').locator('[data-field-problem], [data-field-term]')).toHaveCount(0);

  const shown = async () => Promise.all(['start', 'direction', 'radius', 'offset', 'target'].map((name) => value(name).inputValue()));
  const before = await shown();
  expect(before).toEqual(['[0 m, 0 m]', '0.5', '0.15 m', '[0.1 m, 0 m]', 'start + offset']);
  await reopen();
  await expect(value('target')).toHaveValue('start + offset');
  expect(await shown()).toEqual(before);
  await expect(row('target').locator('[data-field-problem], [data-field-term]')).toHaveCount(0);
  await expect(row('radius').getByRole('combobox', { name: 'Type', exact: true })).toHaveText('Distance');
});

// Catches a Task view that cannot be scrolled from most of its area: a wheel over a table that has nothing more to
// scroll is kept by the table instead of moving the view around it.
test('a wheel over the Variables table scrolls the Task view', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 600 });
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const panel = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Task', exact: true });
  const table = panel.getByRole('table', { name: 'Variables', exact: true });
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  // The shelf takes its height from the view, and gains the Objective's term row once a trial is drawn.
  await expect(shelf.getByRole('group', { name: 'Term 1', exact: true })).toBeVisible();
  const view = await table.evaluateHandle((element) => {
    for (let node = element.parentElement; node; node = node.parentElement) {
      if (/auto|scroll/.test(getComputedStyle(node).overflowY) && node.scrollHeight > node.clientHeight) return node;
    }
    throw new Error('The Task view does not overflow at this height.');
  });
  const top = () => view.evaluate((node) => node.scrollTop);
  const end = await view.evaluate((node) => node.scrollHeight - node.clientHeight);
  await table.getByRole('row').nth(1).hover();
  const start = await top();
  expect(end - start).toBeGreaterThanOrEqual(100);
  await page.mouse.wheel(0, 100);
  await expect.poll(top).toBe(start + 100);
  await page.mouse.wheel(0, -100);
  await expect.poll(top).toBe(start);
});
