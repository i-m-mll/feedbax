import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { reveal } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) =>
  page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

// Catches: a Synthesize step that New cannot make, or that is authored in a form the Service does not read, so its
// Method and settings are lost.
test('New makes a Synthesize step whose Method and settings are edited in its inspector', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const saved = async () => (await (await request.get(projectPath)).json() as StudioProjectSnapshot).project.steps?.synthesize;
  await page.goto(`/?project=${snapshot.project_id}`);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Synthesize');
  const card = page.getByRole('article', { name: 'Synthesize synthesize', exact: true });
  await expect(card).toContainText('Constant parameters');
  expect(await saved()).toMatchObject({ kind: 'synthesize', base: { inline: {
    model: { document: expect.any(String) }, task: { document: expect.any(String) },
    objective: { document: expect.any(String) }, method: 'feedbax.constant_parameters',
  } } });

  await mapStep(page, 'synthesize').click();
  await tab(page, 'Runs').click();
  await reveal(page, 'Inspector');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('combobox', { name: 'Method', exact: true })).toHaveValue('Constant parameters');
  const value = inspector.getByRole('textbox', { name: 'Value', exact: true });
  await value.fill('2');
  await value.press('Enter');
  await expect.poll(async () => (await saved())?.base).toMatchObject({ inline: { settings: { value: 2 } } });
});
