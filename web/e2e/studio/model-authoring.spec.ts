import { expect, test, type Locator } from '@playwright/test';
import type {
  GraphSpec, StudioJobEventHistory, StudioJobSnapshot, StudioJobSubmission, StudioProjectDocument,
  StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('authors the saved reaching Model by keyboard and drawer', async ({ page, request }) => {
  test.setTimeout(60_000);
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
  const train = (saved: StudioProjectSnapshot) => {
    const base = saved.project.steps.train.base;
    if (!('inline' in base)) throw new Error('Expected the authored Train definition');
    return base.inline;
  };
  const view = (name: string) => page.getByRole('tabpanel', { name, exact: true });
  const show = (name: string) => page.getByRole('tab', { name, exact: true }).click();
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

  await show('Task');
  const task = view('Task');
  await task.getByRole('combobox', { name: 'Model Input' }).selectOption({ label: 'gru.input' });
  await task.getByRole('spinbutton', { name: 'Target X (m)', exact: true }).fill('0.1');
  await task.getByRole('spinbutton', { name: 'Target Y (m)', exact: true }).fill('-0.05');
  const bound = await edit(() => task.getByRole('button', { name: 'Bind Target', exact: true }).press('Enter'));
  expect(train(bound).task_binding_spec).toMatchObject({
    exposed_data: [{
      id: 'reach_target', path: 'inputs.effector_target.pos', expected_shape: ['time', 2],
      units: 'm', frame: 'world',
      value_spec: { value_form: 'literal', value: [0.1, -0.05] },
    }],
    bindings: [{
      id: 'task:reach_target->gru:input', source_data_id: 'reach_target',
      target_node_id: 'gru', target_port: 'input', role: 'model_input',
    }],
  });
  await task.focus();
  const unbound = await edit(() => page.keyboard.press('Meta+Z'));
  expect(train(unbound).task_binding_spec).toEqual(train(restored).task_binding_spec);
  const rebound = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(train(rebound).task_binding_spec).toEqual(train(bound).task_binding_spec);

  await show('Objective');
  const objective = view('Objective');
  const loss = objective.getByRole('combobox', { name: 'Edit Final-state objective', exact: true });
  const withLoss = await edit(() => loss.selectOption({ label: 'Endpoint position → Reach target' }));
  expect(train(withLoss).objective).toMatchObject({
    kind: 'loss_term',
    loss: {
      type: 'target_state', selector: 'path:pointMass.effector.pos',
      target_selector: 'task_data:reach_target', time_agg: { mode: 'final' },
    },
  });
  await objective.focus();
  const withoutLoss = await edit(() => page.keyboard.press('Meta+Z'));
  expect(train(withoutLoss).objective).toEqual(train(restored).objective);
  const restoredLoss = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(train(restoredLoss).objective).toEqual(train(withLoss).objective);

  await show('Training');
  const configured = await edit(() => enter(view('Training').getByRole('spinbutton', { name: 'Edit N Batches' }), '2'));
  expect(configured.project.documents).toEqual(restored.project.documents);
  expect(train(configured).training_config).toMatchObject({ n_batches: 2, n_reach_steps: 200 });
  const runPath = `${projectPath}/train-runs`;
  async function plan() {
    const response = await request.get(runPath, { params: { revision: configured.revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  const reviewed = await plan();
  expect(reviewed.window_map?.problems).toEqual([]);
  expect(reviewed.rows).toEqual([expect.objectContaining({
    key: { variant: 'default', step: 'train', row: 'default' }, request_id: expect.any(String), attempts: [],
  })]);
  await page.reload();
  await show('Runs');
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  const admission = mutation('POST', runPath);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await admission;
  expect(accepted.status(), await accepted.text()).toBe(202);
  const { items } = await accepted.json() as StudioJobSubmission;
  expect(items).toEqual([expect.objectContaining({
    run_key: '["default","train","default"]', disposition: 'admitted', job_id: expect.any(String),
  })]);
  const job = items[0].job_id;
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 24_000 });
  const finished = await plan();
  expect(finished.rows).toEqual([expect.objectContaining({
    request_id: reviewed.rows[0].request_id,
    attempts: [expect.objectContaining({
      job_id: job, status: 'succeeded', progress: { completed_updates: 2, total_updates: 2 },
      successful_receipt: expect.objectContaining({ manifest_sha256: expect.any(String) }),
    })],
  })]);
  const jobs = await request.get('/api/jobs', { params: { project: projectId } });
  expect(jobs.ok(), await jobs.text()).toBe(true);
  const { cursor } = await jobs.json() as StudioJobSnapshot;
  const history = await request.get('/api/events/history', { params: { project: projectId, through: cursor } });
  expect(history.ok(), await history.text()).toBe(true);
  const { events } = await history.json() as StudioJobEventHistory;
  const interval = events.filter((event) => event.kind === 'progress' && event.job_id === job)
    .flatMap((event) => event.data.metric_interval ? [event.data.metric_interval] : [])
    .at(-1) as { end_inclusive: number; metrics: { train_loss: { last: number } } } | undefined;
  expect(interval?.end_inclusive).toBe(2);
  expect(Number.isFinite(interval?.metrics.train_loss.last)).toBe(true);

  await show('Model');
  const peer = await request.post(`${projectPath}/files`, {
    data: { revision: configured.revision, file: { kind: 'step', id: 'peer', step: configured.project.steps.train } },
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
