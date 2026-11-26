import { expect, test } from './fixtures/studio';
import { type Page } from '@playwright/test';
import { modelDocumentSchemaId, type StoredModelDocument as ModelDocument, type ModelEditResponse, type StudioProjectDocument } from '../../src/generated/studioContracts';
import { chooseSweepValue, enter, mapStep, runProjectFixture, seedManagedProject, mapCard } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { modelEdit, outlineRow, pick, placeFromLibrary } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

/** The first attachment on the wire `wire` of `model`. */
const firstAttachment = (model: ModelDocument, wire: string) => Object.values(model.interior.wires?.[wire]?.attachments ?? {})[0];

test.use({ viewport: { width: 1440, height: 900 } });

async function addIntervention(page: Page, projectPath: string, selection: { rows: string[]; axes?: object }) {
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  const button = page.getByRole('menuitem', { name: 'Add Intervention', exact: true });
  const [edited] = await Promise.all([
    page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === `${projectPath}/model/edit`),
    button.click(),
  ]);
  const body = edited.request().postDataJSON();
  expect({ rows: body.rows, ...(body.axes ? { axes: body.axes } : {}) }).toEqual(selection);
  expect(edited.ok(), await edited.text()).toBe(true);
  const { result } = await edited.json() as ModelEditResponse;
  expect(result.status, result.status === 'refused' ? result.problem.detail : '').toBe('applied');
}

test('authors and reopens an addressed Model through the project history', { tag: '@slow' }, async ({ page, request }) => {
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
    if (result.status !== 'applied') throw new Error(JSON.stringify(result));
    return result;
  }
  function stored(project: StudioProjectDocument['snapshot']) {
    return rootPayload(project.project.documents?.model) as unknown as ModelDocument;
  }
  const card = (id: string) => model.locator(`.react-flow__node[data-id="${id}"]`);
  await page.goto(url);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await expect(model.locator('[data-boundary="true"]')).toHaveCount(2);
  await mapStep(page, 'train').click();
  await expect(inspector.getByRole('textbox', { name: 'Step', exact: true })).toBeEnabled();
  const start = await request.get(`${projectPath}/open`);
  const initial = (await start.json() as StudioProjectDocument).snapshot;

  await apply(() => enter(inspector.getByRole('textbox', { name: 'Step', exact: true }), '0.02 s'));
  await expect(inspector.getByRole('textbox', { name: 'Step', exact: true })).toHaveValue('0.02 s');
  await expect(model.getByRole('button', { name: 'Lock', exact: true })).toBeEnabled();
  const added = await apply(() => placeFromLibrary(page, 'GRU'));
  const component = added.changed[0].element;
  if (component.kind !== 'component') throw new Error('Add did not return a component address');
  const cell = card(component.id);
  await expect(cell).toBeVisible();
  await expect(cell.getByText('GRU', { exact: true }).first()).toBeVisible();
  await model.getByRole('button', { name: 'Fit View', exact: true }).click();
  const initialPosition = await cell.evaluate((node) => node.style.transform);
  // Topology edits wait for the plan of the saved revision after each commit.
  const editableCard = () => expect(cell).toHaveClass(/\bdraggable\b/);
  await editableCard();
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
  await editableCard();
  const wired = await apply(async () => {
    await handle('output', 'hidden').hover();
    await page.mouse.down();
    await handle('input', 'hidden').hover();
    expect((await reviewed).ok()).toBe(true);
    await page.mouse.up();
  });
  const wire = wired.changed[0].element;
  if (wire.kind !== 'wire') throw new Error('Connect did not return a wire address');
  expect(Object.values(stored(wired.snapshot).interior.wires?.[wire.id].attachments ?? {})).toMatchObject([
    { kind: 'delay' },
  ]);
  const delay = model.getByRole('button', { name: 'Attachment Delay', exact: true });
  await expect(delay).toHaveAccessibleDescription(/1 st · .+hidden → .+hidden/);
  await delay.click();
  await expect(inspector.getByRole('button', { name: 'Delay kind: Value' })).toBeVisible();
  await expect(inspector.getByRole('button', { name: 'Initial kind: Value' })).toBeVisible();
  await apply(() => enter(inspector.getByRole('textbox', { name: 'Delay', exact: true }), '0.06 s'));
  await expect(delay).toHaveAccessibleDescription(/^0\.06 s · /);
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
  expect(firstAttachment(stored(linked.snapshot), wire.id)).toMatchObject({
    initial_source: { kind: 'wire', source: { owner: 'component', component: component.id, port: 'hidden' } },
  });
  await model.focus();
  const undone = await apply(() => page.keyboard.press('Meta+Z'), 'undo');
  const undoneDelay = firstAttachment(stored(undone.snapshot), wire.id) as { initial_source?: unknown };
  expect(undoneDelay.initial_source ?? { kind: 'value', value: 0 }).toEqual({ kind: 'value', value: 0 });
  await model.focus();
  const redone = await apply(() => page.keyboard.press('Meta+Shift+Z'), 'undo');
  expect(firstAttachment(stored(redone.snapshot), wire.id)).toMatchObject({
    initial_source: { kind: 'wire', source: { owner: 'component', component: component.id, port: 'hidden' } },
  });

  const before = await cell.evaluate((node) => node.style.transform);
  const wirePath = model.locator(`.react-flow__edge[data-id="${wire.id}"] .react-flow__edge-path`);
  await expect(wirePath).toHaveAttribute('d', /^M/);
  const unflippedPath = (await wirePath.getAttribute('d'))!;
  await page.locator('[data-slot="secondary"]').getByRole('button', { name: 'Close Slot', exact: true }).click();
  await cell.locator('header').click();
  await page.keyboard.press('Meta+K');
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  const filter = palette.getByRole('combobox');
  await filter.fill('Fit View');
  await expect(palette.getByRole('option', { name: 'Fit View', exact: true })).toHaveCount(1);
  await filter.fill('Flip');
  await expect(palette.getByRole('option', { name: 'Flip', exact: true })).toBeVisible();
  await filter.press('Escape');
  await cell.locator('header').click({ button: 'right' });
  const flipped = await apply(() => page.getByRole('menuitem', { name: 'Flip', exact: true }).press('Enter'));
  expect(flipped.snapshot.semantic_root).toBe(redone.snapshot.semantic_root);
  await expect(handle('output', 'hidden').locator('..')).toHaveAttribute('data-side', 'left');
  await expect(wirePath).not.toHaveAttribute('d', unflippedPath);
  await expect(model.getByRole('button', { name: 'Collapse All', exact: true })).toBeEnabled();
  const collapsed = await apply(() => model.getByRole('button', { name: 'Collapse All', exact: true }).click());
  expect(collapsed.snapshot.semantic_root).toBe(flipped.snapshot.semantic_root);
  await expect(handle('output', 'hidden').locator('..')).toHaveAttribute('data-collapsed', 'true');
  await model.focus();
  const expanded = await apply(() => page.keyboard.press('Meta+Z'), 'undo');
  expect(expanded.snapshot.semantic_root).toBe(collapsed.snapshot.semantic_root);
  await expect(handle('output', 'hidden').locator('..')).toHaveAttribute('data-collapsed', 'false');
  const group = await apply(async () => {
    await cell.locator('header').click();
    await page.keyboard.press('Meta+G');
  });
  const groupAddress = group.changed.find(({ element }) => element.kind === 'component' && element.id !== component.id);
  if (groupAddress?.element.kind !== 'component') throw new Error('Group did not return a component address');
  const composite = card(groupAddress.element.id);
  await expect(composite).toBeVisible();
  await expect(composite).toHaveAttribute('tabindex', '0');
  await composite.focus();
  await expect(composite).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await model.focus();
  await page.keyboard.press('Meta+K');
  await filter.fill('Leave');
  await expect(palette.getByRole('option', { name: /^Leave\b/ })).toBeVisible();
  await filter.press('Escape');
  await model.focus();
  await page.keyboard.press('Meta+ArrowUp');
  await expect(model.getByRole('navigation', { name: 'Model Path' })).toBeHidden();
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await card(groupAddress.element.id).locator('header').dblclick();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Group')).toBeVisible();
  await expect(cell).toHaveCount(1);
  await expect(handle('output', 'hidden').locator('..')).toHaveAttribute('data-side', 'left');
  expect(await cell.evaluate((node) => node.style.transform)).toBe(before);
  expect(initial.project.documents?.model.schema).toBe(modelDocumentSchemaId);
});

test('an Evaluate attachment for the step is stored in a document based on the subject\'s, and reopens', { tag: '@slow' }, async ({ page, request, studio }, info) => {
  const seeded = await seedManagedProject(request, []);
  const { projectPath } = seeded;
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, seeded.configuration.project, 'sweep');
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
  const edited = () => page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === `${projectPath}/model/edit`);
  await outlineRow(page, 'Cell.output → Readout.input').click();
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  const added = await modelEdit(page, projectPath,
    () => page.getByRole('menuitem', { name: 'Add Intervention', exact: true }).click(), 'Selected Readers');
  expect(added.ok(), await added.text()).toBe(true);
  expect(added.request().postDataJSON()).toMatchObject({ step: 'baseline', command: { operation: { op: 'set_attachment' } } });
  const { result } = await added.json() as { result: { created: string[] } };
  const [variant] = result.created;
  expect(variant).toBeDefined();
  // The view keeps its path in the document the edit made.
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Controller')).toBeVisible();
  await model.getByRole('button', { name: 'Attachment Scale', exact: true }).click();
  await expect(inspector.getByRole('textbox', { name: 'Factor', exact: true })).toHaveValue('1');
  const [changed] = await Promise.all([edited(), enter(inspector.getByRole('textbox', { name: 'Factor', exact: true }), '0.5')]);
  expect(changed.ok(), await changed.text()).toBe(true);
  const saved = (await changed.json() as { result: { snapshot: StudioProjectDocument['snapshot'] } }).result.snapshot;
  const { [variant]: based, ...others } = saved.project.documents ?? {};
  expect(others).toEqual(snapshot.project.documents);
  expect(based).toMatchObject({ base: { document: expect.any(String) } });
  expect((saved.project.steps?.baseline?.base as { inline: { trial_documents: unknown } }).inline.trial_documents)
    .toEqual({ model: { document: variant } });
  await page.reload();
  await mapStep(page, 'baseline').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(model.getByRole('navigation', { name: 'Model Path' }).getByText('› Controller')).toBeVisible();
  await expect(model.getByRole('button', { name: 'Attachment Scale', exact: true })).toBeVisible();
  await model.getByRole('button', { name: 'Attachment Scale', exact: true }).click();
  await expect(inspector.getByRole('textbox', { name: 'Factor', exact: true })).toHaveValue('0.5');
  const evaluation = mapCard(page, 'baseline');
  await chooseSweepValue(evaluation, 'Seed', 'Control');
  await outlineRow(page, 'Cell.output → Readout.input').click();
  await addIntervention(page, projectPath, { rows: ['seed=control'], axes: { seed: ['control'] } });
  await mapStep(page, 'baseline').click();
  await evaluation.locator('.studio-map-run-mark').and(evaluation.getByRole('button', { name: /^Select Comparison/ })).click();
  await outlineRow(page, 'Cell.output → Readout.input').click();
  await addIntervention(page, projectPath, { rows: ['seed=comparison'] });
  await expect(model.getByRole('button', { name: 'Attachment Scale', exact: true })).toHaveCount(2);
});

test('persists an attachment in the selected project variant', { tag: '@slow' }, async ({ page, request, studio }, info) => {
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, []);
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, configuration.project, 'variant');
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await pick(page.getByRole('combobox', { name: 'Variant', exact: true }), 'treated');
  await mapStep(page, 'baseline').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await model.locator('[data-id="wrapper"] header').dblclick();
  await model.locator('[data-id="controller"] header').dblclick();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await outlineRow(page, 'Cell.output → Readout.input').click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  const [added] = await Promise.all([
    page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === `${projectPath}/model/edit`),
    page.getByRole('menuitem', { name: 'Add Intervention', exact: true }).click(),
  ]);
  expect(added.ok(), await added.text()).toBe(true);
  expect(added.request().postDataJSON()).toMatchObject({ step: 'baseline', variant: 'treated' });
  const { result } = await added.json() as ModelEditResponse;
  expect(result.status, result.status === 'refused' ? result.problem.detail : '').toBe('applied');
  await expect(model.getByRole('button', { name: 'Attachment Scale', exact: true })).toBeVisible();
  await page.reload();
  await pick(page.getByRole('combobox', { name: 'Variant', exact: true }), 'treated');
  await mapStep(page, 'baseline').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(model.getByRole('button', { name: 'Attachment Scale', exact: true })).toBeVisible();
});
