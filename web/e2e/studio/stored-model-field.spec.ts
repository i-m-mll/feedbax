import { expect, test } from '@playwright/test';
import type { ModelFieldResponse, StudioProjectDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('reopens a Model duration range and an integer sample in their saved forms', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true }).press('Enter');
  await mapStep(page, 'train').click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const tau = inspector.getByRole('region', { name: 'Tau field', exact: true });
  const tauText = tau.getByRole('textbox', { name: 'Tau', exact: true });
  const hidden = inspector.getByRole('textbox', { name: 'Hidden size', exact: true });
  const rnn = page.getByRole('button', { name: 'Add VanillaRNN', exact: true });
  await rnn.press('Enter');
  await expect(tauText).toHaveValue(/ st$/);
  await enter(tauText, '30 ms');
  await expect(tauText).toHaveValue('0.03 s');
  await enter(tauText, 'linspace(20 ms, 40 ms, 3)');
  await expect(tauText).toHaveValue('linspace(0.02 s, 0.04 s, 3)');
  await enter(tau.getByRole('textbox', { name: 'Count', exact: true }), '4');
  await expect(tauText).toHaveValue('linspace(0.02 s, 0.04 s, 4)');
  await enter(hidden, 'sample(choice(8, 12), 4)');
  await expect(hidden).toHaveValue(/sample\(choice\(8, 12, per=run\), 4\)/);
  const saved = await request.get(projectPath);
  expect(saved.ok()).toBe(true);
  const opened = await saved.json() as StudioProjectSnapshot;
  expect(opened.project.steps?.train.sweep?.axes?.tau).toMatchObject({
    path: expect.stringContaining('"path":["fields","tau"]'), range: { start: 0.02, stop: 0.04, count: 4 },
  });
  expect(opened.project.steps?.train.sweep?.axes?.['hidden-size']).toMatchObject({
    sampler: { count: 4, distribution: { family: 'choice' } },
  });
  await page.reload();
  await model.locator('.react-flow__node').filter({ hasText: 'VanillaRNN' }).locator('header').click();
  await expect(tauText).toHaveValue('linspace(0.02 s, 0.04 s, 4)');
  await expect(hidden).toHaveValue(/sample\(choice\(8, 12, per=run\), 4\)/);
});

test('edits and reopens a stored Model field sweep through project history', async ({ page, request }) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true }).press('Enter');
  await mapStep(page, 'train').click();

  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const hidden = inspector.getByRole('textbox', { name: 'Hidden size', exact: true });
  await page.getByRole('button', { name: 'Add GRU', exact: true }).press('Enter');
  await expect(hidden).toHaveValue('4');
  await enter(hidden, '64');
  await expect(inspector.getByRole('button', { name: 'Hidden size kind: Value' })).toBeVisible();
  await expect(hidden).toHaveValue('64');
  await expect.poll(async () => {
    const { snapshot } = await (await request.get(projectPath + '/open')).json() as StudioProjectDocument;
    return JSON.stringify(snapshot.project.documents?.model?.payload);
  }).toContain('"hidden_size":64');

  const valueMenu = inspector.getByRole('button', { name: 'Hidden size kind: Value' });
  await valueMenu.click();
  await expect(inspector.getByRole('menuitem')).toHaveText(['Value', 'Sweep', 'Formula']);
  await inspector.getByRole('menuitem', { name: 'Sweep', exact: true }).click();
  await expect(hidden).toHaveValue('64,');
  await enter(hidden, '64, 96, 128');
  await expect(inspector.getByRole('textbox', { name: 'Hidden size value 3', exact: true })).toHaveValue('128');
  const saved = async () => (await (await request.get(projectPath + '/open')).json() as StudioProjectDocument).snapshot;
  const axis = (value: StudioProjectSnapshot) => value.project.steps?.train.sweep?.axes?.['hidden-size'];
  await expect.poll(async () => Object.keys((axis(await saved()) as { values: object })?.values ?? {}).length).toBe(3);
  const ids = Object.keys((axis(await saved()) as { values: object }).values);
  await enter(inspector.getByRole('textbox', { name: 'Hidden size value 2', exact: true }), '100');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":100');
  expect(Object.keys((axis(await saved()) as { values: object }).values)).toEqual(ids);

  const select = page.getByRole('button', { name: /Select hidden-size/ }).first();
  await select.click();
  await expect(inspector.getByRole('textbox', { name: 'Hidden size value 2', exact: true })).toHaveValue('100');
  await mapStep(page, 'train').click();
  await model.focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":96');
  await model.focus();
  await page.keyboard.press('Meta+Shift+Z');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":100');

  await page.reload();
  await model.locator('.react-flow__node').filter({ hasText: 'GRU' }).locator('header').click();
  await expect(hidden).toHaveValue('64, 100, 128');
  const invalid = page.waitForResponse((response) => response.url().endsWith('/model/fields') &&
    response.request().postDataJSON()?.edit?.text === 'U(0, 1)');
  await enter(hidden, 'U(0, 1)');
  const { problems } = await (await invalid).json() as ModelFieldResponse;
  expect(problems[0].reason).toContain('Value, Sweep, Formula');
  await expect(inspector.getByRole('alert')).toContainText('Value, Sweep, Formula');
  await hidden.press('Escape');
  await expect(hidden).toHaveValue('64, 100, 128');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});
