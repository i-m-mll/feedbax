import { expect, test } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, StudioProjectDocument } from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('authors and reopens an addressed Model through the project history', async ({ page, request }) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const mutation = (kind: 'edit' | 'undo') => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/${kind}`);
  async function apply(action: () => Promise<unknown>, kind: 'edit' | 'undo' = 'edit') {
    const [response] = await Promise.all([mutation(kind), action()]);
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    expect(result.status, result.status === 'refused' ? result.problem.detail : '').toBe('applied');
    if (result.status !== 'applied') throw new Error(result.problem.detail);
    return result;
  }
  function stored(project: StudioProjectDocument['snapshot']) {
    return project.project.documents?.model.payload as unknown as ModelDocument;
  }
  const card = (id: string) => model.locator(`.react-flow__node[data-id="${id}"]`);
  await page.goto(url);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true }).press('Enter');
  await expect(model.locator('[data-boundary="true"]')).toHaveCount(2);
  await mapStep(page, 'train').click();
  await expect(inspector.getByRole('textbox', { name: 'Step', exact: true })).toBeEnabled();
  const start = await request.get(`${projectPath}/open`);
  const initial = (await start.json() as StudioProjectDocument).snapshot;

  const step = await apply(() => enter(inspector.getByRole('textbox', { name: 'Step', exact: true }), '0.02 s'));
  expect(stored(step.snapshot).step_seconds).toBe(0.02);
  const gru = drawer.getByRole('button', { name: 'Add GRU', exact: true });
  const added = await apply(() => gru.press('Enter'));
  const component = added.changed[0].element;
  if (component.kind !== 'component') throw new Error('Add did not return a component address');
  const cell = card(component.id);
  await expect(cell).toBeVisible();
  await expect(cell.getByText('GRU', { exact: true }).first()).toBeVisible();
  await model.getByRole('button', { name: 'Fit View', exact: true }).click();
  const initialPosition = await cell.evaluate((node) => node.style.transform);
  const moved = await apply(async () => {
    const bounds = await cell.locator('header').boundingBox();
    if (!bounds) throw new Error('The cell header is not measurable');
    const x = bounds.x + bounds.width / 2, y = bounds.y + bounds.height / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 45, y + 35, { steps: 8 });
    await page.mouse.up();
  });
  expect(moved.snapshot.semantic_root).toBe(added.snapshot.semantic_root);
  await expect.poll(() => cell.evaluate((node) => node.style.transform)).not.toBe(initialPosition);
  const handle = (direction: 'input' | 'output', name: string) => cell.locator(`[data-handleid="${direction}:${name}"]`);
  const reviewed = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/model/connections`);
  const wired = await apply(async () => {
    await handle('output', 'hidden').hover();
    await page.mouse.down();
    await handle('input', 'hidden').hover();
    expect((await reviewed).ok()).toBe(true);
    await page.mouse.up();
  });
  const wire = wired.changed[0].element;
  if (wire.kind !== 'wire') throw new Error('Connect did not return a wire address');
  expect(stored(wired.snapshot).interior.wires?.[wire.id].attachments).toMatchObject([
    { kind: 'delay', delay: { value: 1, unit: 'st' }, initial: 0 },
  ]);
  await model.getByRole('button', { name: 'Delay', exact: true }).click();
  const delayed = await apply(() => enter(inspector.getByRole('textbox', { name: 'Delay', exact: true }), '0.06 s'));
  expect(stored(delayed.snapshot).interior.wires?.[wire.id].attachments?.[0]).toMatchObject({ delay: { value: 0.06, unit: 's' } });
  await expect(inspector.getByText('3 st')).toBeVisible();
  const linked = await apply(async () => {
    await inspector.getByRole('button', { name: 'Pick Initial', exact: true }).click();
    await handle('output', 'hidden').click();
  });
  expect(stored(linked.snapshot).interior.wires?.[wire.id].attachments?.[0]).toMatchObject({
    initial: { kind: 'name', reference: expect.stringContaining(component.id) },
  });
  await model.focus();
  const undone = await apply(() => page.keyboard.press('Meta+Z'), 'undo');
  expect(stored(undone.snapshot).interior.wires?.[wire.id].attachments?.[0]).toMatchObject({ initial: 0 });
  await model.focus();
  const redone = await apply(() => page.keyboard.press('Meta+Shift+Z'), 'undo');
  expect(stored(redone.snapshot).interior.wires?.[wire.id].attachments?.[0]).toMatchObject({
    initial: { kind: 'name', reference: expect.stringContaining(component.id) },
  });

  const before = await cell.evaluate((node) => node.style.transform);
  const group = await apply(async () => {
    await cell.locator('header').click();
    await page.keyboard.press('Meta+G');
  });
  const groupAddress = group.changed.find(({ element }) => element.kind === 'component');
  if (groupAddress?.element.kind !== 'component') throw new Error('Group did not return a component address');
  await card(groupAddress.element.id).locator('header').dblclick();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await page.reload();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await expect(cell).toHaveCount(1);
  expect(await cell.evaluate((node) => node.style.transform)).toBe(before);
  expect(initial.project.documents?.model.schema).toBe('feedbax.spec.model');
});
