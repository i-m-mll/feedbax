import { expect, test } from './fixtures/studio';
import { type Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick, reveal } from './fixtures/controls';
import { openView } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

// Catches an invalid Model hiding the steps that read it: the Evaluate's fields and runs vanishing, each bad row's
// problem listed once per row, a listed subject drawing no Checkpoint, an update past a subject's training saved,
// an Objective term's Weight locked while the Model does not build, and an input whose producer cannot run explained by
// a sentence (a raw run key) rather than the Cannot Run mark and its link.
test('keeps the steps that read an invalid Model editable and marks them waiting on it', { tag: '@slow' }, async ({ page, request, studio }) => {
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
  await expect(inspector.getByRole('combobox', { name: 'Runs', exact: true })).toBeVisible();
  await expect(inspector.getByRole('combobox', { name: 'Checkpoint', exact: true })).toHaveText('Last');
  await expect(inspector.getByRole('combobox', { name: 'Trials', exact: true })).toBeVisible();
  await expect(runs.getByRole('row')).toHaveCount(3);
  await expect(runs.getByText("Can't run", { exact: true })).toHaveCount(2);
  const subjectRow = inspector.getByRole('list', { name: 'Subject', exact: true }).getByRole('listitem');
  await expect(subjectRow.getByRole('img', { name: "Can't run", exact: true })).toBeVisible();
  await expect(subjectRow.getByRole('button', { name: 'Train', exact: true })).toBeVisible();
  await expect(inspector.getByRole('alert')).toHaveCount(0);

  await page.getByRole('button', { name: /^Problems/ }).click();
  const problems = page.getByRole('dialog', { name: 'Problems', exact: true });
  await expect(problems.getByRole('listitem')).toHaveCount(2);
  await expect(problems.getByRole('heading', { name: 'Reach model', exact: true })).toBeVisible();
  for (const component of ['feedbax.equinox.Linear', 'feedbax.Conv2d']) {
    await expect(problems.getByRole('listitem').filter({ hasText: component })).toContainText('Train · 2 runs');
  }
  await page.keyboard.press('Escape');

  await runs.getByRole('button', { name: /^Train · / }).first().click();
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');

  await mapStep(page, 'conditions').click();
  // Add makes an empty row and writes nothing; the step chosen in that row is the write.
  const writes: string[] = [];
  page.on('request', (sent) => { if (sent.method() === 'PATCH' && new URL(sent.url()).pathname === path) writes.push(sent.url()); });
  const before = (await (await request.get(path)).json() as StudioProjectSnapshot).revision;
  const subject = inspector.getByRole('list', { name: 'Subject', exact: true });
  await inspector.getByRole('button', { name: 'Add Subject', exact: true }).click();
  const empty = subject.getByRole('combobox', { name: 'Subject', exact: true }).last();
  await expect(subject.getByRole('listitem')).toHaveCount(2);
  await expect(empty).toHaveText('—');
  await expect.poll(async () => (await (await request.get(path)).json() as StudioProjectSnapshot).revision).toBe(before);
  expect(writes).toEqual([]);
  const listed = await edit(() => pick(empty, 'Train b'));
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
  // A checkpoint outside the range every subject run has saves, with its reason on the field.
  for (const [value, reason] of [[9000, 'Train b · 1 to 8000'], [0, 'Train · 1 to 8000']] as const) {
    const outside = await edit(async () => {
      await update.fill(String(value));
      await update.press('Enter');
    });
    expect(outside.steps!.conditions.inputs!.subject).toEqual([
      { step: 'train', checkpoint: { at: value } }, { step: 'train_b', checkpoint: { at: value } },
    ]);
    await expect(update).toHaveAccessibleDescription(new RegExp(reason));
    await expect(update.locator('..').getByRole('img', { name: 'Problem', exact: true })).toBeVisible();
  }

  await mapStep(page, 'analyze').click();
  await openView(page, 'analysis');
  await page.getByRole('tabpanel', { name: 'Analysis', exact: true }).getByRole('group', { name: 'Select states', exact: true }).click();
  const states = inspector.getByRole('list', { name: 'States', exact: true }).getByRole('listitem');
  await expect(states.getByRole('img', { name: "Can't run", exact: true })).toBeVisible();
  await states.getByRole('button', { name: /^Conditions/ }).click();
  await expect(mapStep(page, 'conditions')).toHaveAttribute('aria-pressed', 'true');

  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  const objective = page.getByRole('tabpanel', { name: 'Objective', exact: true });
  const weight = objective.getByRole('textbox', { name: 'Weight', exact: true }).first();
  const weighted = await edit(async () => {
    await enter(weight, '3');
    await page.getByRole('menu', { name: 'Apply To', exact: true }).getByRole('menuitem', { name: 'All 2 Steps', exact: true }).click();
  });
  expect(rootPayload(weighted.documents?.objective)).toMatchObject({
    terms: [expect.objectContaining({ weight: expect.objectContaining({ kind: 'literal', value: 3 }) })],
  });
  await page.reload();
  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true }).first()).toHaveValue('3');
});
