import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, TaskDocument } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type File = { kind: string; id: string; document?: { payload: TaskDocument } };

/** The delayed-reaches files with `hold` and `cue` of fixed lengths, so that three repetitions fit the trial. */
async function files() {
  const all = await fixture('delayed-reaches-files') as File[];
  const task = all.find(({ id }) => id === 'delayed-reaches-task')!.document!.payload;
  const fixed = (steps: number) => ({ kind: 'length', length: { kind: 'literal', value: steps, unit: 'st' }, max_steps: steps }) as const;
  task.epochs = task.epochs.map((epoch) => epoch.id === 'hold' ? { ...epoch, end: fixed(5) }
    : epoch.id === 'cue' ? { ...epoch, end: fixed(10) } : epoch);
  delete task.validation_trials;
  return all;
}

// Catches a Repeat that writes no bracket over the selected columns or none of count 2, a bracket count that does not
// reach the nominal shelf as one instance per repetition, an Item whose Each Repeat does not write the repetition's
// read `[i]`, and a Repeat crossing another that is refused, or stored without a row and a mark of its own.
test('Repeat brackets the selected columns, its count repeats them on the shelf, and Each Repeat reads the repetition', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await files());
  const task = async () => {
    const { project } = (await (await request.get(projectPath)).json()) as StudioProjectSnapshot;
    return (project.documents?.['delayed-reaches-task'] as unknown as { payload: TaskDocument }).payload;
  };
  const primary = page.locator('[data-slot="primary"]');
  const grid = primary.getByRole('table', { name: 'Epochs', exact: true });
  const header = (name: string) => grid.locator('th[data-epoch]').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const bracket = grid.locator('th[data-repeat]');
  const count = bracket.getByRole('textbox');
  const clock = page.locator('footer.studio-window-bottom').getByRole('group', { name: 'Clock', exact: true });
  const epochs = page.getByRole('region', { name: 'Shelf', exact: true }).getByRole('group', { name: 'Epochs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(clock.getByRole('button', { name: /^Trial / })).toHaveText('nominal');
  const shelf = clock.getByRole('button', { name: 'Shelf', exact: true });
  if (await shelf.getAttribute('aria-pressed') !== 'true') await shelf.click();
  await expect(epochs.getByRole('button', { name: 'hold', exact: true })).toHaveCount(1);

  await header('hold').click({ position: { x: 4, y: 4 } });
  await header('cue').click({ modifiers: ['Shift'], position: { x: 4, y: 4 } });
  await header('cue').getByRole('button', { name: 'Epoch cue', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Repeat', exact: true }).click();
  await expect(bracket).toHaveAttribute('colspan', '2');
  await expect(count).toHaveValue('2');
  await expect.poll(async () => (await task()).repeats?.map(({ first_epoch, last_epoch }) => [first_epoch, last_epoch]))
    .toEqual([['hold', 'cue']]);

  await enter(count, '3');
  await expect(epochs.getByRole('button', { name: 'hold', exact: true })).toHaveCount(3);
  await expect(epochs.getByRole('button', { name: 'cue', exact: true })).toHaveCount(3);
  await expect(epochs.getByRole('button', { name: 'move', exact: true })).toHaveCount(1);

  const variables = primary.getByRole('table', { name: 'Variables', exact: true });
  const target = variables.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename target', exact: true }) });
  await target.hover();
  await target.getByRole('button', { name: 'Variable target', exact: true }).click();
  await page.getByRole('menuitemcheckbox', { name: 'List', exact: true }).click();
  await expect(page.getByRole('menu', { name: 'Variable target', exact: true })).toHaveCount(0);
  await expect.poll(async () => (await task()).variables.find(({ id }) => id === 'target')?.count).toMatchObject({ kind: 'literal', value: 2 });

  const goal = grid.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename goal', exact: true }) })
    .locator('td[data-cell]').first().getByRole('textbox');
  await enter(goal, 'target');
  await inspector.getByRole('combobox', { name: 'Item', exact: true }).click();
  await page.getByRole('option', { name: 'Each Repeat', exact: true }).click();
  await expect(goal).toHaveValue('target[i]');
  await expect.poll(async () => (await task()).signals.find(({ id }) => id === 'goal')?.cells[0].value).toMatchObject({
    kind: 'call', function: 'index', args: [{ kind: 'name', reference: 'target' }, { kind: 'name', reference: 'i' }],
  });

  await header('cue').click({ position: { x: 4, y: 4 } });
  await header('move').click({ modifiers: ['Shift'], position: { x: 4, y: 4 } });
  await header('move').getByRole('button', { name: 'Epoch move', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Repeat', exact: true }).click();
  await expect.poll(async () => (await task()).repeats?.map(({ first_epoch, last_epoch }) => [first_epoch, last_epoch]))
    .toEqual([['hold', 'cue'], ['cue', 'move']]);
  await expect(grid.locator('tr').filter({ has: page.locator('th[data-repeat]') })).toHaveCount(2);
  await expect(bracket.getByRole('img', { name: 'Problem', exact: true })).toHaveCount(1);
});
