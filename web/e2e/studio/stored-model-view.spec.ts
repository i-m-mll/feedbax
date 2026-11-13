import { expect, test } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, StudioProjectDocument } from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('authors and reopens an addressed Model through the project history', async ({ page, request }) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
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
  await model.getByRole('button', { name: 'Split Details', exact: true }).click();
  const secondary = page.locator('[data-slot="secondary"]');
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const secondaryModel = secondary.getByRole('tabpanel', { name: 'Model', exact: true });
  await secondaryModel.getByRole('button', { name: 'Fit View', exact: true }).click();
  const otherHidden = secondaryModel.locator(`.react-flow__node[data-id="${component.id}"] [data-handleid="output:hidden"]`);
  await model.focus();
  const linked = await apply(async () => {
    await inspector.getByRole('button', { name: 'Pick Initial', exact: true }).click();
    await expect(otherHidden.locator('..')).toHaveAttribute('data-verdict', 'allowed');
    await otherHidden.click();
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
  await page.getByRole('button', { name: 'Close Slot', exact: true }).click();
  const group = await apply(async () => {
    await cell.locator('header').click();
    await page.keyboard.press('Meta+G');
  });
  const groupAddress = group.changed.find(({ element }) => element.kind === 'component' && element.id !== component.id);
  if (groupAddress?.element.kind !== 'component') throw new Error('Group did not return a component address');
  await card(groupAddress.element.id).locator('header').dblclick();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await page.reload();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await expect(cell).toHaveCount(1);
  expect(await cell.evaluate((node) => node.style.transform)).toBe(before);
  expect(initial.project.documents?.model.schema).toBe('feedbax.spec.model');
});

test('edits and reopens an authored Evaluate attachment without changing the Model document', async ({ page, request }, info) => {
  test.setTimeout(45_000);
  const seeded = await seedManagedProject(request, []);
  const { projectPath } = seeded;
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', info.config.metadata.home, seeded.snapshot.project_id);
  const opened = await request.get(`${projectPath}/open`);
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await page.goto(url);
  await mapStep(page, 'baseline').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await model.locator('[data-id="wrapper"] header').dblclick();
  await model.locator('[data-id="controller"] header').dblclick();
  await drawer.getByRole('button', { name: 'Outline', exact: true }).click();
  await drawer.getByRole('button', { name: 'Cell.output → Readout.input', exact: true }).click();
  const [first] = await Promise.all([
    page.waitForResponse((response) => response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath),
    inspector.getByRole('button', { name: 'Add Intervention', exact: true }).click(),
  ]);
  expect(first.ok(), await first.text()).toBe(true);
  await expect(inspector.getByRole('spinbutton', { name: 'Factor', exact: true })).toHaveValue('1');
  const [changed] = await Promise.all([
    page.waitForResponse((response) => response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath),
    enter(inspector.getByRole('spinbutton', { name: 'Factor', exact: true }), '0.5'),
  ]);
  expect(changed.ok(), await changed.text()).toBe(true);
  const saved = await changed.json() as StudioProjectDocument['snapshot'];
  expect(saved.project.documents).toEqual(snapshot.project.documents);
  expect(saved.project.steps?.baseline).toMatchObject({ model_attachments: expect.any(Object) });
  await page.reload();
  await mapStep(page, 'baseline').click();
  await expect(model.getByRole('button', { name: 'Scale', exact: true })).toBeVisible();
  await model.getByRole('button', { name: 'Scale', exact: true }).click();
  await expect(inspector.getByRole('spinbutton', { name: 'Factor', exact: true })).toHaveValue('0.5');
});
