import { expect, test } from '@playwright/test';
import type { ModelFieldResponse, StudioProjectDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

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
