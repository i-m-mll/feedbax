import { expect, test } from './fixtures/studio';
import { enter, mapStep, runProjectFixture, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

// Catches an Evaluate whose Trials take no sweep, whose sweep is not crossed with the runs it evaluates, or whose
// Trials editor is lost once its step has an axis of its own.
test("typing a list in an Evaluate's Trials sweeps them across its subject's seeds", async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring', 'own-sweep-files'));
  const { snapshot } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'probe').click();
  const card = page.getByRole('article', { name: 'Evaluate probe' });
  const table = page.getByRole('table', { name: 'Runs' });
  await expect(table).toHaveAttribute('aria-rowcount', '3');
  const trials = page.getByRole('region', { name: 'Trials', exact: true });
  await enter(trials.getByRole('textbox', { name: 'Count', exact: true }), '1, 4');
  await expect(table).toHaveAttribute('aria-rowcount', '5');
  await expect(trials.getByRole('button', { name: 'Count kind: Sweep', exact: true })).toBeVisible();
  await sweepLabel(card, 'trials').click();
  for (const name of ['1', '4']) await expect(page.getByRole('menuitem', { name, exact: true })).toBeVisible();
});

// Catches Add Sweep missing from an Evaluate, listing none of the fields its own Task defines, or not sweeping one.
test("Add Sweep on an Evaluate lists its own fields and its Task's", async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring', 'own-sweep-files'));
  const { snapshot } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'probe').click();
  const card = page.getByRole('article', { name: 'Evaluate probe' });
  await card.getByRole('button', { name: 'Add Sweep probe' }).click();
  const menu = page.getByRole('menu', { name: 'Add Sweep' });
  await expect(menu.getByRole('menuitem', { name: 'Seed', exact: true })).toBeVisible();
  await expect(menu.getByRole('menuitem', { name: 'Trials', exact: true })).toBeVisible();
  await menu.getByRole('menuitem', { name: 'Task', exact: true }).click();
  await page.getByRole('menu', { name: 'Task' }).getByRole('menuitem', { name: 'gain', exact: true }).click();
  await enter(page.locator(':focus'), '0.5, 1');
  await expect(card.getByRole('button', { name: /^Sweep / })).toHaveCount(2);
});
