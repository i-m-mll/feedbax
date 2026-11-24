import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { ModelDocument, ModelEditResponse } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

async function openModel(page: Page, request: Parameters<typeof seedManagedProject>[0]) {
  const files = await fixture('ordinary-files', '2');
  const saved = files.find((file: { id: string }) => file.id === 'point-mass').document.payload as ModelDocument;
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.locator('.react-flow__node').first()).toBeVisible();
  const edit = () => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/edit`);
  return { model, errors, edit, saved: Object.keys(saved.interior.components ?? {}).sort() };
}

test('a Shift-drag box selects the cards it covers', async ({ page, request }) => {
  const { model, errors } = await openModel(page, request);
  const cards = model.locator('.react-flow__node');
  const boxes = await cards.evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().toJSON() as DOMRect));
  const left = Math.min(...boxes.map((box) => box.left)) - 20;
  const top = Math.min(...boxes.map((box) => box.top)) - 20;
  const right = Math.max(...boxes.map((box) => box.right)) + 20;
  const bottom = Math.max(...boxes.map((box) => box.bottom)) + 20;
  await page.keyboard.down('Shift');
  await page.mouse.move(left, top);
  await page.mouse.down();
  await page.mouse.move(right, bottom, { steps: 12 });
  await page.mouse.up();
  await page.keyboard.up('Shift');
  await expect(model.locator('.react-flow__node.selected').first()).toBeVisible();
  await expect(model.getByText('Model failed to render')).toHaveCount(0);
  expect(errors).toEqual([]);
});

type Applied = Extract<ModelEditResponse['result'], { status: 'applied' }>;
const components = (result: ModelEditResponse['result']) =>
  Object.keys(((result as Applied).snapshot.project.documents!['point-mass'] as unknown as { payload: ModelDocument }).payload.interior.components ?? {}).sort();

test('Delete removes the wired cards Duplicate just made', async ({ page, request }) => {
  const { model, errors, edit, saved } = await openModel(page, request);
  await model.locator('.react-flow__node[data-id="cell"] header').click();
  await page.keyboard.press('Meta+A');
  const duplicated = edit();
  await page.keyboard.press('Meta+D');
  const duplicate = (await (await duplicated).json() as ModelEditResponse).result;
  expect(duplicate.status).toBe('applied');
  const removed = edit();
  await page.keyboard.press('Backspace');
  const remove = (await (await removed).json() as ModelEditResponse).result;
  expect(remove.status, remove.status === 'refused' ? remove.problem.detail : '').toBe('applied');
  expect(components(duplicate)).toHaveLength(2 * saved.length);
  expect(components(remove)).toEqual(saved);
  expect(errors).toEqual([]);
});
