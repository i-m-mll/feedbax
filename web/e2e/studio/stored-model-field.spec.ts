import { expect, test } from './fixtures/studio';
import type { ModelFieldResponse, StudioProjectDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { placeFromLibrary } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

test('reopens a Model duration range and an integer sample in their saved forms', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await mapStep(page, 'train').click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const tau = inspector.getByRole('region', { name: 'Tau field', exact: true });
  const tauText = tau.getByRole('textbox', { name: 'Tau', exact: true });
  const hidden = inspector.getByRole('textbox', { name: 'Hidden size', exact: true });
  await placeFromLibrary(page, 'VanillaRNN');
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
  await mapStep(page, 'train').click();
  await expect(model).toBeVisible();
  await model.locator('.react-flow__node').filter({ hasText: 'VanillaRNN' }).locator('header').click();
  await expect(tauText).toHaveValue('linspace(0.02 s, 0.04 s, 4)');
  await expect(hidden).toHaveValue(/sample\(choice\(8, 12, per=run\), 4\)/);
});

test('edits and reopens a stored Model field sweep through project history', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await mapStep(page, 'train').click();

  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const hidden = inspector.getByRole('textbox', { name: 'Hidden size', exact: true });
  await placeFromLibrary(page, 'GRU');
  await expect(hidden).toHaveValue('4');
  await enter(hidden, '64');
  await expect(inspector.getByRole('button', { name: 'Hidden size kind: Value' })).toBeVisible();
  await expect(hidden).toHaveValue('64');
  await expect.poll(async () => {
    const { snapshot } = await (await request.get(projectPath + '/open')).json() as StudioProjectDocument;
    return JSON.stringify(rootPayload(snapshot.project.documents?.model));
  }).toContain('"hidden_size":64');

  const valueMenu = inspector.getByRole('button', { name: 'Hidden size kind: Value' });
  await valueMenu.click();
  await expect(page.getByRole('menuitemradio')).toHaveText(['Value', 'Sweep']);
  await page.getByRole('menuitemradio', { name: 'Sweep', exact: true }).click();
  await expect(hidden).toBeFocused();
  await enter(hidden, '64, 96, 128');
  await expect(inspector.getByText('128', { exact: true })).toBeVisible();
  const saved = async () => (await (await request.get(projectPath + '/open')).json() as StudioProjectDocument).snapshot;
  const axis = (value: StudioProjectSnapshot) => value.project.steps?.train.sweep?.axes?.['hidden-size'];
  await expect.poll(async () => Object.keys((axis(await saved()) as { values: object })?.values ?? {}).length).toBe(3);
  const ids = Object.keys((axis(await saved()) as { values: object }).values);
  await enter(hidden, '64, 100, 128');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":100');
  expect(Object.keys((axis(await saved()) as { values: object }).values)).toEqual(ids);

  const select = page.getByRole('button', { name: /Select Hidden size/ }).first();
  await select.click();
  await expect(hidden).toHaveValue('64, 100, 128');
  await mapStep(page, 'train').click();
  await model.focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":96');
  await model.focus();
  await page.keyboard.press('Meta+Shift+Z');
  await expect.poll(async () => JSON.stringify(axis(await saved()))).toContain('"value":100');
  await expect(hidden).toHaveValue('64, 100, 128');

  await enter(hidden, '100');
  await expect(inspector.getByRole('button', { name: 'Hidden size kind: Value', exact: true })).toBeVisible();
  await expect.poll(async () => axis(await saved())).toBeFalsy();
  await model.focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => Object.keys((axis(await saved()) as { values?: object })?.values ?? {})).toEqual(ids);

  await page.reload();
  await mapStep(page, 'train').click();
  await expect(model).toBeVisible();
  await model.locator('.react-flow__node').filter({ hasText: 'GRU' }).locator('header').click();
  await expect(hidden).toHaveValue('64, 100, 128');
  const invalid = page.waitForResponse((response) => response.url().endsWith('/model/fields') &&
    response.request().postDataJSON()?.edit?.text === 'U(0, 1)');
  await enter(hidden, 'U(0, 1)');
  const { problems } = await (await invalid).json() as ModelFieldResponse;
  expect(problems![0].reason).toContain('Value, Sweep, Formula');
  // The text is stored in the selected run's sweep value, and its Problem is marked there and kept on reopening.
  await expect(hidden).toHaveAttribute('aria-invalid', 'true');
  await expect(hidden).toHaveAccessibleDescription(/Value, Sweep, Formula/);
  await hidden.press('Escape');
  await expect(hidden).toHaveValue('U(0, 1), 100, 128');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.reload();
  await mapStep(page, 'train').click();
  await model.locator('.react-flow__node').filter({ hasText: 'GRU' }).locator('header').click();
  await expect(hidden).toHaveValue('U(0, 1), 100, 128');
  await expect(hidden).toHaveAccessibleDescription(/Value, Sweep, Formula/);
});
