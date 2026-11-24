import { expect, test } from './fixtures/studio';
import { enter, mapStep, runProjectFixture, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

// Catches an Evaluate whose Trials take no sweep, or whose sweep is not crossed with the runs it evaluates.
test("typing a list in an Evaluate's Trials sweeps them across its subject's seeds", async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring', 'own-sweep-files'));
  const { snapshot } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'probe').click();
  const card = page.getByRole('article', { name: 'Evaluate probe' });
  const table = page.getByRole('table', { name: 'Runs' });
  await expect(table).toHaveAttribute('aria-rowcount', '2');
  const trials = page.getByRole('region', { name: 'Trials', exact: true });
  await enter(trials.getByRole('textbox', { name: 'Count', exact: true }), '1, 4');
  await expect(table).toHaveAttribute('aria-rowcount', '4');
  await sweepLabel(card, 'trials').click();
  for (const name of ['1', '4']) await expect(page.getByRole('menuitem', { name, exact: true })).toBeVisible();
});

// Catches Add Sweep missing from an Evaluate, or listing none of the fields its own Task defines.
test("Add Sweep on an Evaluate lists its own fields and its Task's", async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring', 'own-sweep-files'));
  const { snapshot } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'probe').click();
  await page.getByRole('article', { name: 'Evaluate probe' }).getByRole('button', { name: 'Add Sweep probe' }).click();
  const menu = page.getByRole('menu', { name: 'Add Sweep' });
  await expect(menu.getByRole('menuitem', { name: 'Seed', exact: true })).toBeVisible();
  await expect(menu.getByRole('menuitem', { name: 'Trials', exact: true })).toBeVisible();
  await menu.getByRole('menuitem', { name: 'Task', exact: true }).click();
  await expect(page.getByRole('menu', { name: 'Task' }).getByRole('menuitem')).not.toHaveCount(0);
});
