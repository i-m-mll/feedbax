import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioTrialAuthoring, TaskDocument } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const position = JSON.stringify({ composites: [], element: {
  kind: 'field', parent: { id: 'plant', kind: 'component' }, path: ['reset', 'effector', 'position'],
} });

// Catches a Model start the Model needs that no row shows or marks, a row whose name does not write the variable and
// its link in one edit, a choice in Initializes that does not re-point the link to the row it is made on, and a Model
// start no variable initializes that is hidden from the table.
test('Initializes chooses the Model values a variable starts, and a start the Model needs is a row to name', async ({ page, request }) => {
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
  const unset = async () => {
    const { revision } = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    const response = await request.post(projectPath + '/authoring/trials', {
      data: { revision, run: { variant: 'default', step: 'train', row: 'default' }, change: null },
    });
    return (await response.json() as StudioTrialAuthoring).problems.filter(({ path = [] }) => path[0] === 'links');
  };
  const table = page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('table', { name: 'Variables', exact: true });
  const named = (name: string) => table.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const start = (name: string) => table.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Name ${name}`, exact: true }) });
  const initializes = (name: string) => named(name).getByRole('combobox', { name: `Initializes ${name}`, exact: true });
  const positionName = 'Point mass · effector position';

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(start(positionName).locator('[data-field-problem]')).toHaveAttribute('data-app-tip', /start of each trial/);
  // The run's problem is named by the value its row shows, and revealing it puts the keyboard in that row's name.
  await page.getByRole('button', { name: /^Problems \d+$/ }).click();
  await page.getByRole('dialog', { name: 'Problems', exact: true })
    .getByRole('button', { name: new RegExp(`^${positionName} · The Model needs this value`) }).click();
  await expect(start(positionName).getByRole('textbox', { name: `Name ${positionName}`, exact: true })).toBeFocused();
  await page.keyboard.press('Escape');
  // The reveal selected the run, which would make the next Task edit that run's exception; the step is chosen again.
  await mapStep(page, 'train').click();
  await expect(start('Point mass · effector velocity')).toBeVisible();
  await expect(start('Point mass · effector velocity').locator('[data-field-problem]')).toHaveCount(0);
  expect(await unset()).toEqual([expect.objectContaining({ path: ['links', position] })]);

  await enter(start(positionName).getByRole('textbox', { name: `Name ${positionName}`, exact: true }), 'start');
  await expect(initializes('start')).toHaveAttribute('placeholder', positionName);
  await expect(start(positionName)).toHaveCount(0);
  await expect.poll(links).toEqual([{ model_field: position, variable: 'start-2' }]);
  expect(await unset()).toEqual([]);

  await initializes('origin').click();
  await page.getByRole('option', { name: positionName, exact: true }).click();
  await expect.poll(links).toEqual([{ model_field: position, variable: 'start' }]);
  await page.keyboard.press('Escape');
  await expect(initializes('start')).toHaveAttribute('placeholder', '');
  await expect(initializes('origin')).toHaveAttribute('placeholder', positionName);
});

// Catches a link's problem named by a pointer or a document member no view shows, and a reveal that marks the Task as a
// whole instead of the cell of the variable whose link it is.
test('a link to a field the Model lacks is named and revealed at its variable\'s Initializes', async ({ page, request }) => {
  const response = await request.post('/api/projects');
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
    .getByRole('row').filter({ has: page.getByRole('textbox', { name: `Name ${positionName}`, exact: true }) });
  await expect(row.locator('[data-field-problem]')).toHaveAttribute('data-app-tip', reason);
  await page.getByRole('button', { name: /^Problems \d+$/ }).click();
  await expect(page.getByRole('dialog', { name: 'Problems', exact: true })
    .getByRole('button', { name: new RegExp(`^${positionName} · ${reason.replace('.', '\\.')}`) })).toBeVisible();
});
