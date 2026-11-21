import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches a Train field that goes read-only while an edit saves (its own or another's), and an edit made then that is lost.
test('Train fields stay editable while an edit saves, and an edit made then saves after it', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const save = await hold(page, projectPath, (body) => !!body && typeof body === 'object' && ('patches' in body || 'files' in body));
  const train = async () => JSON.stringify(((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps?.train);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const b1 = page.getByRole('region', { name: 'B1 field', exact: true }).getByRole('textbox', { name: 'B1', exact: true });
  const b2 = page.getByRole('region', { name: 'B2 field', exact: true }).getByRole('textbox', { name: 'B2', exact: true });
  await expect(b1).toBeEditable();

  await enter(b1, '0.71');
  await save.sent;
  await expect(b1).toBeEditable();
  await expect(b2).toBeEditable();
  await enter(b2, '0.973');
  expect(await train()).not.toContain('0.71');

  save.release();
  await expect.poll(train).toContain('0.973');
  expect(await train()).toContain('0.71');
  await expect(b1).toHaveValue('0.71');
  await expect(b2).toHaveValue('0.973');
});
