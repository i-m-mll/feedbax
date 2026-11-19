import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { authorLearningRateSweep, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

test('an optimizer swept with its classes crosses the learning rate, and a member selects its runs and parameters', async ({ page, request }) => {
  test.setTimeout(55_000);
  const files = await fixture('ordinary-files', '2');
  files.at(-1).step.base.inline.settings.optimizer.b1 = 0.8;
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const table = page.getByRole('table', { name: 'Runs' });
  const optimizer = page.getByRole('region', { name: 'Optimizer', exact: true });
  const picker = optimizer.getByRole('combobox', { name: 'Optimizer', exact: true });
  const b1 = optimizer.getByRole('textbox', { name: 'B1', exact: true });

  await authorLearningRateSweep(page.getByRole('region', { name: 'Learning Rate field', exact: true }));
  await optimizer.getByRole('button', { name: 'Optimizer kind: Value', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Sweep', exact: true }).click();
  await expect(optimizer.getByRole('button', { name: 'Optimizer kind: Sweep', exact: true })).toBeVisible();
  await pick(picker, 'SGD');

  await expect(table).toHaveAttribute('aria-rowcount', '6');
  for (const name of ['optimizer', 'learning-rate']) await expect(table.getByRole('columnheader', { name, exact: true })).toBeVisible();
  await expect(table.getByRole('cell', { name: 'SGD', exact: true })).toHaveCount(3);
  await expect(picker).toHaveAttribute('placeholder', 'Adam, SGD');
  await expect(optimizer.getByRole('region', { name: 'Learning Rate field', exact: true })).toBeVisible();
  await expect(b1).toHaveCount(0);

  await optimizer.getByRole('button', { name: 'Select Optimizer Adam', exact: true }).click();
  await expect(b1).toHaveValue('0.8');
  const response = await request.get(projectPath);
  const { project } = await response.json() as StudioProjectSnapshot;
  const members = project.steps!.train.sweep!.axes!.optimizer as { values: Record<string, { patches: Record<string, unknown> }> };
  expect(Object.keys(members.values.sgd.patches)).toEqual(['settings.optimizer.type']);
  expect(members.values.adam.patches['settings.optimizer.b1']).toMatchObject({ value: 0.8 });
});
