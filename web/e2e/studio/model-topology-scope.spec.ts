import { expect, test, type Locator } from '@playwright/test';
import type {
  GraphSpec, StudioDocumentAuthoring, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('places and connects a Model component for exactly the lit steps', async ({ page, request }) => {
  test.setTimeout(48_000);
  const files = await fixture('files');
  const project = await seedManagedProject(request, files.slice(0, 3));
  const { projectPath } = project;
  let original = project.snapshot;
  for (const id of ['lit', 'unlit']) {
    const response = await request.post(`${projectPath}/files`, {
      data: { revision: original.revision, file: { kind: 'step', id, step: original.project.steps.train } },
    });
    expect(response.status(), await response.text()).toBe(201);
    original = (await response.json() as StudioProjectDocument).snapshot;
  }
  const initialModel = original.project.documents.model;
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const gru = model.locator('.react-flow__node[data-id="gru"]');
  const url = `/?project=${encodeURIComponent(original.project_id)}`;
  const button = (container: Locator, name: string) => container.getByRole('button', { name, exact: true });
  const undo = async (key: string) => {
    await model.focus();
    return edit(() => page.keyboard.press(key));
  };
  async function edit(action: () => Promise<unknown>) {
    const saved = page.waitForResponse((response) => response.request().method() === 'PATCH' &&
      new URL(response.url()).pathname === projectPath);
    await action();
    const response = await saved;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return await response.json() as StudioProjectSnapshot;
  }
  async function requests(revision: string) {
    const response = await request.get(`${projectPath}/train-runs`, { params: { revision } });
    expect(response.ok(), await response.text()).toBe(true);
    const { rows } = await response.json() as StudioSavedTrainRuns;
    return Object.fromEntries(rows.map(({ key, request_id }) => [key.step, request_id]));
  }
  async function graph(step: string, revision: string) {
    const response = await request.post(`${projectPath}/authoring/document`, { data: {
      revision, run: { variant: 'default', step, row: 'default' }, field: 'graph',
    } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioDocumentAuthoring).payload as unknown as GraphSpec;
  }
  await page.goto(url);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(button(drawer, 'Add GRU')).toBeEnabled();
  const initialRequests = await requests(original.revision);
  expect(Object.keys(initialRequests).sort()).toEqual(['lit', 'train', 'unlit']);
  await mapStep(page, 'unlit').click({ modifiers: ['Meta'] });
  await expect(model.getByRole('status').filter({ hasText: '2 of 3' })).toBeVisible();

  await model.locator('.react-flow__pane').dblclick({ position: { x: 24, y: 24 } });
  const chooser = page.getByRole('dialog', { name: 'Add Component', exact: true });
  const added = await edit(() => button(chooser, 'Add GRU').press('Enter'));
  await expect(gru).toBeVisible();
  const positions = added.presentation?.steps;
  const point = positions?.train.model_positions?.model.gru;
  expect(point).toMatchObject({ x: expect.any(Number), y: expect.any(Number) });
  expect(positions?.lit.model_positions?.model.gru).toEqual(point);
  expect(positions?.unlit).toBeUndefined();
  expect(added.presentation?.documents).toEqual(original.presentation?.documents);
  expect(added.project.documents.model).toEqual(initialModel);

  await button(drawer, 'Connect Ports').click();
  const connect = page.getByRole('dialog', { name: 'Connect Ports', exact: true });
  await connect.getByRole('combobox', { name: 'Source Port' }).selectOption({ label: 'gru.hidden' });
  await connect.getByRole('combobox', { name: 'Target Port' }).selectOption({ label: 'gru.hidden' });
  const wired = await edit(() => button(connect, 'Connect Ports').press('Enter'));
  await button(drawer, 'Expose Output').click();
  const expose = page.getByRole('dialog', { name: 'Expose Output', exact: true });
  await expose.getByRole('combobox', { name: 'Source Port' }).selectOption({ label: 'gru.hidden' });
  const exposed = await edit(() => button(expose, 'Expose Output').press('Enter'));
  expect(exposed.project.documents.model).toEqual(initialModel);
  expect(exposed.project.steps.unlit).toEqual(original.project.steps.unlit);
  expect(exposed.project.steps.train.document_overrides?.model).toEqual(exposed.project.steps.lit.document_overrides?.model);
  for (const step of ['train', 'lit']) {
    const effective = await graph(step, exposed.revision);
    expect(effective.nodes?.gru.type).toBe('feedbax.GRU');
    expect(effective.wires).toContainEqual(expect.objectContaining({
      source_node: 'gru', source_port: 'hidden', target_node: 'gru', target_port: 'hidden',
      temporality: 'recurrent',
    }));
    expect(effective.output_bindings?.hidden).toEqual(['gru', 'hidden']);
  }
  expect(await graph('unlit', exposed.revision)).toEqual(initialModel.payload);
  const exposedRequests = await requests(exposed.revision);
  expect(exposedRequests.unlit).toBe(initialRequests.unlit);
  expect(exposedRequests.train).not.toBe(initialRequests.train);
  expect(exposedRequests.lit).not.toBe(initialRequests.lit);

  expect((await undo('Meta+Z')).project).toEqual(wired.project);
  expect((await undo('Meta+Z')).project).toEqual(added.project);
  const undone = await undo('Meta+Z');
  expect(undone.project).toEqual(original.project);
  expect(undone.presentation).toEqual(original.presentation);
  expect(await requests(undone.revision)).toEqual(initialRequests);
  await expect(gru).toHaveCount(0);

  await undo('Meta+Shift+Z');
  await undo('Meta+Shift+Z');
  const redone = await undo('Meta+Shift+Z');
  expect(redone.project).toEqual(exposed.project);
  expect(redone.presentation).toEqual(exposed.presentation);
  const transform = await gru.evaluate((node) => (node as HTMLElement).style.transform);
  await page.reload();
  await mapStep(page, 'train').click();
  await expect(gru).toBeVisible();
  expect(await gru.evaluate((node) => (node as HTMLElement).style.transform)).toBe(transform);
  const reopened = await request.get(projectPath);
  expect(reopened.ok()).toBe(true);
  const restored = await reopened.json() as StudioProjectSnapshot;
  expect(restored.project).toEqual(redone.project);
  expect(restored.presentation).toEqual(redone.presentation);
  expect(await requests(restored.revision)).toEqual(exposedRequests);
  await mapStep(page, 'unlit').click();
  await expect(gru).toHaveCount(0);
  await mapStep(page, 'lit').click();
  await expect(gru).toBeVisible();
});
