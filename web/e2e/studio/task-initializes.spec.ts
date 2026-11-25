import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioTrialAuthoring, TaskDocument } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const position = JSON.stringify({ composites: [], element: {
  kind: 'field', parent: { id: 'plant', kind: 'component' }, path: ['state', 'effector/position'],
} });

// Catches a Model start the Model needs that no row shows or marks, a row for a start nothing needs, a row whose choice
// of variable does not write the link, and an Initializes choice that does not bring the row back when it is undone.
test('a start the Model needs and no variable initializes is a row with its problem and a choice of variables', async ({ page, request }) => {
  test.setTimeout(60_000);
  const files = await fixture('ordinary-files', '2') as { kind: string; id: string; document?: { payload: Record<string, any> } }[];
  for (const file of files) {
    const payload = file.document?.payload;
    if (payload?.interior) payload.interior.components.plant.type = 'reaching_task.AnchoredPointMass';
    if (payload?.variables) {
      payload.links = [];
      payload.variables.find(({ id }: { id: string }) => id === 'start').name = 'origin';
    }
  }
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const links = async () => {
    const { project } = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    const [, task] = Object.entries(project.documents ?? {}).find(([name]) => name.endsWith('-task'))!;
    return (task as unknown as { payload: TaskDocument }).payload.links ?? [];
  };
  const table = page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('table', { name: 'Variables', exact: true });
  const positionName = 'Point mass · effector position';
  const picker = (name: string) => table.getByRole('combobox', { name: `Name ${name}`, exact: true });
  const row = (name: string) => table.getByRole('row')
    .filter({ has: page.getByRole('combobox', { name: `Name ${name}`, exact: true }) });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(row(positionName).locator('[data-field-problem]')).toHaveAttribute('data-app-tip', /start of each trial/);
  await expect(picker('Point mass · effector velocity')).toHaveCount(0);
  // The run's problem is named by the value its row shows, and revealing it puts the keyboard on that row's choice.
  await page.getByRole('button', { name: /^Problems/ }).click();
  await page.getByRole('dialog', { name: 'Problems', exact: true })
    .getByRole('button', { name: new RegExp(`^${positionName} · The Model needs this value`) }).click();
  await expect(picker(positionName)).toBeFocused();
  await page.keyboard.press('Escape');
  await mapStep(page, 'train').click();

  await pick(picker(positionName), 'origin');
  await expect.poll(links).toEqual([{ model_field: position, variable: 'start' }]);
  await expect(picker(positionName)).toHaveCount(0);
  const initializes = table.getByRole('combobox', { name: 'Initializes origin', exact: true });
  await expect(initializes).toHaveAttribute('placeholder', positionName);

  await initializes.click();
  await page.keyboard.type('effector position');
  await page.getByRole('option', { name: 'Point mass · effector · position', exact: true }).click();
  await expect.poll(links).toEqual([]);
  await page.keyboard.press('Escape');
  await expect(row(positionName).locator('[data-field-problem]')).toHaveAttribute('data-app-tip', /start of each trial/);
});

// Catches the five leftover rows of the point mass: a Model start no variable initializes drawn as a row with no name.
test('the shipped Task\'s Variables are its variables and its Sets', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const table = page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('table', { name: 'Variables', exact: true });
  await expect(table.getByRole('rowheader').first()).toBeVisible();
  const names = await table.getByRole('rowheader').evaluateAll((cells) =>
    cells.map((cell) => cell.querySelector('input')?.value ?? cell.textContent ?? ''));
  expect(names.length).toBeGreaterThan(0);
  expect(names.filter((name) => !name.trim())).toEqual([]);
});

// Catches an Initializes list that is flat, offers a state whose unit the variable does not have, or whose type-ahead
// finds a state only at the level the list is open on.
test('Initializes opens on the Model\'s components and descends to the states a variable fits', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const cell = page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('combobox', { name: 'Initializes start', exact: true });
  const list = page.getByRole('listbox');
  await cell.click();
  await expect(list.getByRole('option', { name: 'position', exact: true })).toHaveCount(0);
  await list.getByRole('option', { name: 'Point mass', exact: true }).click();
  await list.getByRole('option', { name: 'effector', exact: true }).click();
  await expect(list.getByRole('option')).toHaveText(['Point mass · effector', 'position']);
  await cell.fill('pos');
  await expect(list.getByRole('option', { name: 'Point mass · effector · position', exact: true })).toContainText('Point mass · effector');
  await expect(list.getByRole('option', { name: /velocity/ })).toHaveCount(0);
});

// Catches a link's problem named by a pointer or a document member no view shows, and a reveal that marks the Task as a
// whole instead of the cell of the variable whose link it is.
test('a link to a field the Model lacks is named and revealed at its variable\'s Initializes', async ({ page, request }) => {
  const response = await request.post('/api/projects', { data: { name: 'missing-initializes' } });
  expect(response.status()).toBe(201);
  const { snapshot } = await response.json() as { snapshot: StudioProjectSnapshot };
  await page.goto(`/?project=${snapshot.project_id}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).press('Enter');
  await page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true }).press('Enter');
  const reason = 'Initializes · start is linked to a field the Model does not have.';
  await page.getByRole('button', { name: 'Problems 1', exact: true }).click();
  await page.getByRole('dialog', { name: 'Problems', exact: true }).getByRole('button', { name: new RegExp(`^${reason}`) }).click();
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const cell = task.getByRole('combobox', { name: 'Initializes start', exact: true });
  await expect(cell).toBeFocused();
  await expect(task.locator('[data-field="task:start:initializes"] [data-field-problem]'))
    .toHaveAttribute('data-app-tip', /start is linked to a field the Model does not have/);
  await expect(task.getByText(reason, { exact: true })).toHaveCount(0);
});

// Catches a link naming no variable that leaves its Model value with no row, or names its problem by the Task's links.
test('a link to a variable the Task lacks is marked and named at the Model value it leaves unstarted', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2') as { kind: string; id: string; document?: { payload: Record<string, any> } }[];
  for (const file of files) {
    const payload = file.document?.payload;
    if (payload?.interior) payload.interior.components.plant.type = 'reaching_task.AnchoredPointMass';
    if (payload?.variables) payload.links = [{ model_field: position, variable: 'ghost' }];
  }
  const { snapshot } = await seedManagedProject(request, files);
  const positionName = 'Point mass · effector position';
  const reason = 'ghost is not a variable of the Task.';
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const row = page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('table', { name: 'Variables', exact: true })
    .getByRole('row').filter({ has: page.getByRole('combobox', { name: `Name ${positionName}`, exact: true }) });
  await expect(row.locator('[data-field-problem]')).toHaveAttribute('data-app-tip', reason);
  await page.getByRole('button', { name: /^Problems \d+$/ }).click();
  await expect(page.getByRole('dialog', { name: 'Problems', exact: true })
    .getByRole('button', { name: new RegExp(`^${positionName} · ${reason.replace('.', '\\.')}`) })).toBeVisible();
});
