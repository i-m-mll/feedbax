import { expect, test } from './fixtures/studio';
import type { StudioProjectDocument, StudioProjectExamples } from '../../src/generated/studioContracts';
import { mapStep } from './fixtures/canonicalTrain';
import { testProjectName } from './fixtures/studioKeyboard';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches a Task in the picker marked with another Task's problem, or left unmarked when it leaves the Model's
// target input unfed.
test("the Task picker marks only the Task that leaves the Model's input unfed", async ({ page, request }) => {
  const { examples } = await (await request.get('/api/projects/examples')).json() as StudioProjectExamples;
  const created = await request.post('/api/projects', { data: { name: testProjectName(), example: examples[0].id } });
  expect(created.status()).toBe(201);
  const { snapshot } = await created.json() as StudioProjectDocument;
  const shipped = snapshot.project.documents?.['simple-reaches-task'];
  if (!shipped || !('payload' in shipped)) throw new Error('The example has no root simple-reaches Task');
  const payload = shipped.payload as { signals: Record<string, unknown> };
  const { target_pos: _, ...signals } = payload.signals;
  const document = { ...shipped, label: 'Untargeted', payload: { ...payload, id: 'untargeted', signals } };
  const added = await request.post(`/api/projects/${snapshot.project_id}/files`, {
    data: { revision: snapshot.revision, file: { kind: 'document', id: 'untargeted', document } },
  });
  expect(added.status(), await added.text()).toBe(201);

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'simple-reaches').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const picker = page.locator('.studio-window-slot__header').getByRole('button', { name: /^Task Document / });
  await expect(picker).toHaveText('Simple reaches', { timeout: 30_000 });
  await picker.click();
  const items = page.getByRole('menu', { name: 'Task Document', exact: true }).getByRole('menuitemradio');
  await expect(items.filter({ hasText: 'Untargeted' }).getByRole('img', { name: 'Leaves target_pos unfed' })).toHaveCount(1);
  await expect(items.getByRole('img', { name: /unfed$/ })).toHaveCount(1);
});
