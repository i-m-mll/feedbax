import { expect, test, type Locator } from '@playwright/test';
import type {
  GraphSpec, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('authors and reopens the reaching Model from an empty Train, then reaches ordinary Run', async ({ page, request }) => {
  test.setTimeout(55_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  expect(snapshot.project.steps ?? {}).toEqual({});
  expect(snapshot.project.documents ?? {}).toEqual({});
  const projectId = snapshot.project_id;
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const pageUrl = `/?project=${encodeURIComponent(projectId)}`;
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });

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

  const card = page.getByRole('article', { name: 'Train train', exact: true });
  const problems = page.getByRole('dialog', { name: 'Problems', exact: true });
  await card.getByRole('button', { name: "1 Can't run", exact: true }).press('Enter');
  await problems.getByRole('button', { name: 'Components · Model has no components.', exact: true }).press('Enter');
  await expect(problems).toBeHidden();
  await expect(model.getByText('Components · Model has no components.', { exact: true })).toBeFocused();
  await expect(card.locator('.studio-map-run-mark')).toHaveAttribute('aria-pressed', 'true');
  await expect(add(drawer, 'GRU')).toBeDisabled();
  await mapStep(page, 'train').click();
  await expect(add(drawer, 'GRU')).toBeEnabled();

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

  await tab('Task').click();
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  await edit(() => enter(task.getByRole('spinbutton', { name: 'Edit Eval Reach Length', exact: true }), '0.6'));
  await tab('Training').click();
  const training = page.getByRole('tabpanel', { name: 'Training', exact: true });
  const shortened = await edit(() => enter(training.getByRole('spinbutton', { name: 'Edit N Batches', exact: true }), '2'));
  expect(shortened.project.steps?.train?.base).toMatchObject({
    inline: { training_config: { n_batches: 2 }, method_payload: { payload: { target_total: 2 } } },
  });
  await tab('Model').click();
  await page.reload();
  await expect(model.locator('.react-flow__node')).toHaveCount(3);
  await expect(model.locator('.react-flow__edge')).toHaveCount(3);
  const reopened = await request.get(`${projectPath}/open`);
  expect(reopened.ok()).toBe(true);
  const { snapshot: restored } = await reopened.json() as StudioProjectDocument;
  expect(restored.revision).toBe(shortened.revision);
  expect(graph(restored)).toEqual(authored);
  expect(restored.project.documents?.task).toMatchObject({ payload: { params: { eval_reach_length: 0.6 } } });
  expect(restored.presentation?.documents?.model.positions?.pointMass).toEqual(placed);

  await mapStep(page, 'train').click();
  await tab('Runs').click();
  const runs = page.getByRole('region', { name: 'Runs', exact: true });
  const run = runs.getByRole('button', { name: 'Run', exact: true });
  await expect(run).toBeVisible();
  const planPath = `${projectPath}/train-runs`;
  const read = await request.get(planPath, { params: { revision: restored.revision, window_map: true } });
  expect(read.ok(), await read.text()).toBe(true);
  const plan = await read.json() as StudioSavedTrainRuns;
  const key = { variant: 'default', step: 'train', row: 'default' };
  expect(plan.rows).toMatchObject([{ key, attempts: [] }]);
  await test.info().attach('first-Train saved Plan', {
    body: JSON.stringify({ rows: plan.rows, problems: plan.window_map?.problems }),
    contentType: 'application/json',
  });

  test.fail(true, '2d24abe · First Train still needs managed task bindings and an authored loss term.');
  await expect(run).toBeEnabled();
  const admitted = mutation('POST', planPath);
  await run.click();
  const admission = await admitted;
  expect(admission.status(), await admission.text()).toBe(202);
  expect((await admission.json() as StudioJobSubmission).items).toEqual([
    expect.objectContaining({ run_key: JSON.stringify([key.variant, key.step, key.row]), disposition: 'admitted' }),
  ]);
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
});

test('rejects topology edits when shared Model users are narrowed', async ({ page, request }) => {
  const [model, task, train] = await fixture('files');
  const { snapshot, projectPath } = await seedManagedProject(request, [model, task, train, { ...train, id: 'peer' }]);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const add = page.getByRole('complementary', { name: 'Details Drawer', exact: true })
    .getByRole('button', { name: 'Add GRU', exact: true });
  await expect(add).toBeEnabled();
  await mapStep(page, 'peer').click({ modifiers: ['Meta'] });
  await expect(add).toBeDisabled();
  const graph = snapshot.project.documents?.model.payload as unknown as GraphSpec;
  const gru = Object.values(graph.nodes ?? {}).find((node) => node.type === 'feedbax.GRU');
  expect(gru).toBeDefined();
  const refused = await request.post(`${projectPath}/authoring/document`, { data: {
    revision: snapshot.revision, run: { variant: 'default', step: 'train', row: 'default' }, field: 'graph',
    change: { lit_steps: ['train'], operation: { kind: 'addComponent', component: 'feedbax.GRU', revision: gru?.revision } },
  } });
  expect(refused.status()).toBe(422);
  expect(await refused.text()).toContain('all Model users');
  expect((await (await request.get(projectPath)).json() as StudioProjectSnapshot).revision).toBe(snapshot.revision);
});
