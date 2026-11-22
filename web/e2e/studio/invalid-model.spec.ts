import { expect, test } from './fixtures/studio';
import { type Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick, reveal } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

// Catches an invalid Model hiding the steps that read it: the Evaluate's fields and runs vanishing, each bad row's
// problem listed once per row, a listed subject drawing no Checkpoint, an update past a subject's training saved,
// and an Objective term's Weight locked while the Model does not build.
test('keeps the steps that read an invalid Model editable and marks them waiting on it', async ({ page, request, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.invalid_model', 'seed', studio.home, 'invalid'));
  const path = '/api/projects/' + project;
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  async function edit(action: () => Promise<unknown>) {
    const saving = page.waitForResponse((response) =>
      response.request().method() === 'PATCH' && new URL(response.url()).pathname === path);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectSnapshot).project;
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded' });
  await mapStep(page, 'conditions').click();
  await tab(page, 'Runs').click();
  await reveal(page, 'Inspector');
  await expect(inspector.getByRole('region', { name: 'Runs', exact: true })).toBeVisible();
  await expect(inspector.getByRole('combobox', { name: 'Checkpoint', exact: true })).toHaveText('Last');
  await expect(inspector.getByRole('combobox', { name: 'Trials', exact: true })).toBeVisible();
  await expect(runs.getByRole('row')).toHaveCount(3);
  await expect(runs.getByText("Can't run", { exact: true })).toHaveCount(2);

  await page.getByRole('button', { name: /^Problems/ }).click();
  const problems = page.getByRole('dialog', { name: 'Problems', exact: true });
  await expect(problems.getByRole('listitem')).toHaveCount(2);
  for (const component of ['feedbax.equinox.Linear', 'feedbax.Conv2d']) {
    await expect(problems.getByRole('listitem').filter({ hasText: component })).toContainText('train · 2 runs');
  }
  await page.keyboard.press('Escape');

  await runs.getByRole('button', { name: /^train · / }).first().click();
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');

  await mapStep(page, 'conditions').click();
  const listed = await edit(() => pick(inspector.getByRole('combobox', { name: 'Add Subject', exact: true }), 'train_b'));
  expect(listed.steps!.conditions.inputs!.subject).toEqual([
    { step: 'train', checkpoint: 'last' }, { step: 'train_b', checkpoint: 'last' },
  ]);
  const kind = inspector.getByRole('combobox', { name: 'Checkpoint', exact: true });
  await expect(kind).toHaveCount(1);
  await pick(kind, 'At');
  const update = inspector.getByRole('textbox', { name: 'Update', exact: true });
  const at = await edit(() => enter(update, '5000'));
  expect(at.steps!.conditions.inputs!.subject).toEqual([
    { step: 'train', checkpoint: { at: 5000 } }, { step: 'train_b', checkpoint: { at: 5000 } },
  ]);
  await update.fill('9000');
  await update.press('Enter');
  await expect(inspector.getByRole('alert').filter({ hasText: 'train_b · 8000 batches' })).toBeVisible();
  const kept = await (await request.get(path)).json() as StudioProjectSnapshot;
  expect(kept.project.steps!.conditions.inputs!.subject).toEqual(at.steps!.conditions.inputs!.subject);

  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  const objective = page.getByRole('tabpanel', { name: 'Objective', exact: true });
  const weight = objective.getByRole('textbox', { name: 'Weight', exact: true }).first();
  const weighted = await edit(() => enter(weight, '3'));
  expect(rootPayload(weighted.documents?.objective)).toMatchObject({
    terms: [expect.objectContaining({ weight: expect.objectContaining({ kind: 'literal', value: 3 }) })],
  });
  await page.reload();
  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true }).first()).toHaveValue('3');
});
