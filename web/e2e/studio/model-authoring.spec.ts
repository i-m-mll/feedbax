import { expect, test, type Locator } from '@playwright/test';
import type { GraphSpec, StudioProjectDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('authors the saved reaching Model by keyboard and drawer', async ({ page, request }) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const projectId = snapshot.project_id;
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const pageUrl = `/?project=${encodeURIComponent(projectId)}`;

  function mutation(method: 'PATCH' | 'POST', path: string) {
    return page.waitForResponse((response) =>
      response.request().method() === method && new URL(response.url()).pathname === path);
  }
  async function edit(action: () => Promise<unknown>) {
    const saving = mutation('PATCH', projectPath);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return await response.json() as StudioProjectSnapshot;
  }
  const graph = (saved: StudioProjectSnapshot) => saved.project.documents?.model.payload as unknown as GraphSpec;
  const field = (name: string) => inspector.getByRole('spinbutton', { name: `Edit ${name}`, exact: true });
  const add = (container: Locator, name: string) => container.getByRole('button', { name: `Add ${name}`, exact: true });

  await page.goto(pageUrl);
  await page.getByRole('button', { name: 'New Step', exact: true }).press('Enter');
  const newStep = page.getByRole('dialog', { name: 'New Step', exact: true });
  await newStep.getByRole('button', { name: 'Train', exact: true }).press('Enter');
  await expect(model).toBeFocused();
  await expect(model.locator('.react-flow__node')).toHaveCount(0);

  await edit(() => add(drawer, 'GRU').click());
  await expect(model.locator('.react-flow__node.selected')).toHaveCount(1);
  await edit(() => enter(field('Input Size'), '2'));
  await edit(() => enter(field('Hidden Size'), '100'));

  await model.focus();
  await page.keyboard.press('Shift+A');
  const chooser = page.getByRole('dialog', { name: 'Add Component', exact: true });
  const search = chooser.getByRole('textbox', { name: 'Components', exact: true });
  await expect(search).toBeFocused();
  await search.press('Shift+A');
  await expect(search).toHaveValue('A');
  await search.fill('linear');
  await edit(() => add(chooser, 'Linear').press('Enter'));
  await expect(chooser).toBeHidden();
  await edit(() => enter(field('Input Size'), '100'));
  await edit(() => enter(field('Output Size'), '2'));
  const position = page.waitForResponse((response) => response.request().method() === 'PUT' &&
    new URL(response.url()).pathname === `${projectPath}/presentation`);
  await model.locator('.react-flow__pane').dblclick({ position: { x: 24, y: 24 } });
  await edit(() => add(chooser, 'PointMass').click());
  const positioned = await position;
  expect(positioned.ok(), await positioned.text()).toBe(true);
  const { presentation } = await positioned.json() as StudioProjectSnapshot;
  const placed = presentation?.documents?.model.positions?.pointMass;
  expect(placed).toMatchObject({ x: expect.any(Number), y: expect.any(Number) });
  await expect(model.locator('.react-flow__node')).toHaveCount(3);

  async function connect(source: string, target: string, keyboard = false) {
    if (keyboard) {
      await model.focus();
      await page.keyboard.press('Meta+K');
      const palette = page.getByRole('dialog', { name: 'Command Palette' });
      await palette.getByRole('combobox').fill('Connect Ports');
      await expect(palette.getByRole('option', { name: 'Connect Ports', exact: true })).toHaveAttribute('data-command-id', 'connectPorts');
      await page.keyboard.press('Enter');
    } else {
      await drawer.getByRole('button', { name: 'Connect Ports', exact: true }).click();
    }
    const dialog = page.getByRole('dialog', { name: 'Connect Ports', exact: true });
    await dialog.getByRole('combobox', { name: 'Source Port' }).selectOption({ label: source });
    await dialog.getByRole('combobox', { name: 'Target Port' }).selectOption({ label: target });
    const submit = dialog.getByRole('button', { name: 'Connect Ports', exact: true });
    await edit(() => keyboard ? submit.press('Enter') : submit.click());
  }

  await connect('gru.hidden', 'linear.input', true);
  await connect('gru.hidden', 'gru.hidden');
  await connect('linear.output', 'pointMass.force');
  await drawer.getByRole('button', { name: 'Expose Output', exact: true }).click();
  const output = page.getByRole('dialog', { name: 'Expose Output', exact: true });
  await output.getByRole('combobox', { name: 'Source Port' }).selectOption({ label: 'pointMass.effector' });
  const complete = await edit(() => output.getByRole('button', { name: 'Expose Output', exact: true }).click());
  const authored = graph(complete);
  expect(authored.nodes).toMatchObject({
    gru: { type: 'feedbax.GRU', revision: expect.any(Number), params: { input_size: 2, hidden_size: 100 } },
    linear: { type: 'feedbax.Linear', revision: expect.any(Number), params: { input_size: 100, output_size: 2 } },
    pointMass: { type: 'feedbax.PointMass', revision: expect.any(Number), params: { mass: 1, dt: 0.01 } },
  });
  expect(authored.wires).toMatchObject([
    { source_node: 'gru', source_port: 'hidden', target_node: 'linear', target_port: 'input' },
    { source_node: 'gru', source_port: 'hidden', target_node: 'gru', target_port: 'hidden',
      temporality: 'recurrent', recurrent_initializer: { kind: 'zeros', scope: 'trial', shape: [100], dtype: 'float32' } },
    { source_node: 'linear', source_port: 'output', target_node: 'pointMass', target_port: 'force' },
  ]);
  expect(authored.output_ports).toEqual(['effector']);
  expect(authored.output_bindings).toEqual({ effector: ['pointMass', 'effector'] });
  expect(complete.project.documents?.task).toMatchObject({ schema: 'feedbax.spec.task' });

  await model.focus();
  const undone = await edit(() => page.keyboard.press('Meta+Z'));
  expect(graph(undone).output_bindings).toBeUndefined();
  expect(graph(undone).output_ports).toBeUndefined();
  expect(graph(undone).wires).toEqual(authored.wires);
  const redone = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(graph(redone)).toEqual(authored);

  await page.reload();
  await expect(model.locator('.react-flow__node')).toHaveCount(3);
  await expect(model.locator('.react-flow__edge')).toHaveCount(3);
  const reopened = await request.get(projectPath);
  expect(reopened.ok()).toBe(true);
  const restored = await reopened.json() as StudioProjectSnapshot;
  expect(graph(restored)).toEqual(authored);
  expect(restored.presentation?.documents?.model.positions?.pointMass).toEqual(placed);

  const peer = await request.post(`${projectPath}/files`, {
    data: { revision: redone.revision, file: { kind: 'step', id: 'peer', step: redone.project.steps.train } },
  });
  expect(peer.status(), await peer.text()).toBe(201);
  const { snapshot: shared } = await peer.json() as StudioProjectDocument;
  await expect(mapStep(page, 'peer')).toBeVisible();
  await expect(add(drawer, 'GRU')).toBeEnabled();
  await mapStep(page, 'peer').click({ modifiers: ['Meta'] });
  await expect(add(drawer, 'GRU')).toBeDisabled();
  const refused = await request.post(`${projectPath}/authoring/document`, { data: {
    revision: shared.revision, run: { variant: 'default', step: 'train', row: 'default' }, field: 'graph',
    change: { lit_steps: ['train'], operation: { kind: 'addComponent', component: 'feedbax.GRU', revision: authored.nodes?.gru.revision } },
  } });
  expect(refused.status()).toBe(422);
  expect(await refused.text()).toContain('all Model users');
  expect(graph(await (await request.get(projectPath)).json() as StudioProjectSnapshot)).toEqual(authored);
});
