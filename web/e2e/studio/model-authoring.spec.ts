import { expect, test, type Locator, type Page } from '@playwright/test';
import type {
  ModelDocument, StudioJobEventHistory, StudioJobSnapshot, StudioJobSubmission, StudioProjectDocument, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'retain-on-failure', viewport: { width: 1440, height: 900 } });

type ModelWire = NonNullable<ModelDocument['interior']['wires']>[string];

function matchesFields(actual: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== 'object') return actual === expected;
  if (actual === null || typeof actual !== 'object') return false;
  return Object.entries(expected).every(([key, value]) =>
    matchesFields((actual as Record<string, unknown>)[key], value));
}

async function connect(page: Page, source: Locator, target: Locator, path: string,
  operation: Record<string, unknown>, matchesWire: (wire: ModelWire) => boolean) {
  await expect(source).toHaveAttribute('aria-disabled', 'false');
  await expect(target).toHaveAttribute('aria-disabled', 'false');
  const saved = page.waitForResponse((response) => {
    const request = response.request();
    if (request.method() !== 'POST' || new URL(response.url()).pathname !== path + '/model/edit') return false;
    const actual = request.postDataJSON()?.command?.operation;
    return actual?.graph?.document === 'model' && actual.graph.element?.kind === 'graph' &&
      !actual.graph.composites.length && matchesFields(actual, operation);
  });
  await source.hover();
  await page.mouse.down();
  await target.hover();
  await page.mouse.up();
  const response = await saved;
  expect(response.ok(), await response.text()).toBe(true);
  const { result } = await response.json();
  expect(result.status, result.problem?.detail).toBe('applied');
  const model = result.snapshot.project.documents.model.payload as ModelDocument;
  expect(Object.values(model.interior.wires ?? {}).filter(matchesWire)).toHaveLength(1);
}

test('authors a managed Train in an empty folder and reopens its result', async ({ page, request }) => {
  test.setTimeout(58_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  expect(snapshot.project.steps).toEqual({});
  expect(snapshot.project.documents).toEqual({});
  const url = '/?project=' + encodeURIComponent(snapshot.project_id);
  await page.goto(url);
  const create = page.getByRole('button', { name: 'New Step', exact: true });
  await expect(create).toBeEnabled();
  await create.press('Enter');
  await page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true }).press('Enter');
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  const panel = (name: string) => page.getByRole('tabpanel', { name, exact: true });
  const model = panel('Model');
  const task = panel('Task');
  const objective = panel('Objective');
  const runs = panel('Runs');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(model).toBeFocused();
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await card.getByRole('button', { name: "1 Can't run", exact: true }).press('Enter');
  const problems = page.getByRole('dialog', { name: 'Problems', exact: true });
  await problems.getByRole('button', { name: 'Components · Model has no components.', exact: true }).press('Enter');
  await expect(problems).toBeHidden();
  await expect(model.getByText('Components · Model has no components.', { exact: true })).toBeFocused();
  await mapStep(page, 'train').click();
  await tab('Task').click();
  await expect(task.getByRole('region', { name: 'Signals' })).toContainText('Reach Target');
  const taskTarget = task.getByRole('spinbutton', { name: 'Target x (m)', exact: true });
  const taskSteps = task.getByRole('spinbutton', { name: /N Steps/i });
  const saved = async () => {
    const response = await request.get(projectPath + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return ((await response.json()) as StudioProjectDocument).snapshot;
  };
  const patch = () => page.waitForResponse((response) => response.request().method() === 'PATCH' &&
    new URL(response.url()).pathname === projectPath);
  const edit = async (field: Locator, value: string) => {
    const [saved] = await Promise.all([patch(), enter(field, value)]);
    expect(saved.ok(), await saved.text()).toBe(true);
    await expect(field).toBeEnabled();
  };
  await edit(taskTarget, '0.22');
  await edit(taskSteps, '12');
  await task.focus();
  await Promise.all([patch(), page.keyboard.press('Meta+Z')]);
  await expect(taskSteps).toHaveValue('80');
  await task.focus();
  await Promise.all([patch(), page.keyboard.press('Meta+Shift+Z')]);
  await expect(taskSteps).toHaveValue('12');
  await expect.poll(async () => (await saved()).project.documents?.task?.payload).toMatchObject({
    type: 'feedbax.task.reaching.v1', params: { x: 0.22, n_steps: 12 },
  });

  await tab('Model').click();
  const component = (name: string) => model.locator('.react-flow__node').filter({ hasText: name });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  async function add(name: string, category: string) {
    const library = drawer.getByRole('region', { name: category, exact: true });
    const button = library.getByRole('button', { name: 'Add ' + name, exact: true });
    await button.click();
    const place = inspector.getByRole('button', { name: 'Add ' + name, exact: true });
    await expect(place).toBeEnabled();
    const edited = page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === projectPath + '/model/edit');
    await place.click();
    const response = await edited;
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json();
    expect(result.status, result.problem?.detail).toBe('applied');
    const card = component(name);
    await expect(card).toBeVisible();
    await card.locator('header').click();
    return card;
  }
  const cell = await add('GRU', 'Runtime');
  await expect(inspector.getByRole('textbox', { name: 'Input size', exact: true })).toBeEnabled();
  await enter(inspector.getByRole('textbox', { name: 'Input size', exact: true }), '2');
  await expect.poll(async () => Object.values(((await saved()).project.documents?.model?.payload as unknown as ModelDocument)
    .interior.components ?? {})).toEqual(expect.arrayContaining([expect.objectContaining({
      type: 'feedbax.GRU', fields: expect.objectContaining({ input_size: 2 }),
    })]));
  const linear = await add('Linear', 'Runtime');
  await enter(inspector.getByRole('textbox', { name: 'Input size', exact: true }), '4');
  await expect.poll(async () => Object.values(((await saved()).project.documents?.model?.payload as unknown as ModelDocument)
    .interior.components ?? {}).find(({ type }) => type === 'feedbax.Linear')).toMatchObject({ fields: { input_size: 4 } });
  await enter(inspector.getByRole('textbox', { name: 'Output size', exact: true }), '2');
  await expect.poll(async () => Object.values(((await saved()).project.documents?.model?.payload as unknown as ModelDocument)
    .interior.components ?? {})).toEqual(expect.arrayContaining([expect.objectContaining({
      type: 'feedbax.Linear', fields: expect.objectContaining({ input_size: 4, output_size: 2 }),
    })]));
  const plant = await add('PointMass', 'Mechanics');
  const cellId = (await cell.getAttribute('data-id'))!;
  const linearId = (await linear.getAttribute('data-id'))!;
  const plantId = (await plant.getAttribute('data-id'))!;
  const endpoint = (id: string, port: string) => ({ owner: 'component', component: id, port } as const);
  const inputToCell = endpoint(cellId, 'input');
  const gruOutput = endpoint(cellId, 'output');
  const linearInput = endpoint(linearId, 'input');
  const linearOutput = endpoint(linearId, 'output');
  const plantForce = endpoint(plantId, 'force');
  await model.getByRole('button', { name: 'Fit View', exact: true }).click();
  await connect(page, model.locator('[aria-label="reach_target"].react-flow__handle'),
    cell.locator('[data-handleid="input:input"]'), projectPath,
    { op: 'connect_task_signal', signal: 'reach_target', target: inputToCell },
    (wire) => wire.source.owner === 'inputs' && matchesFields(wire.target, inputToCell));
  await connect(page, cell.locator('[data-handleid="output:output"]'),
    linear.locator('[data-handleid="input:input"]'), projectPath,
    { op: 'connect', source: gruOutput, target: linearInput },
    (wire) => matchesFields(wire.source, gruOutput) && matchesFields(wire.target, linearInput));
  await connect(page, linear.locator('[data-handleid="output:output"]'),
    plant.locator('[data-handleid="input:force"]'), projectPath,
    { op: 'connect', source: linearOutput, target: plantForce },
    (wire) => matchesFields(wire.source, linearOutput) && matchesFields(wire.target, plantForce));
  const output = model.locator('[data-id="@outputs"]');
  await output.getByRole('button', { name: 'Add Port', exact: true }).click();
  const portDialog = page.getByRole('dialog', { name: 'Add Port', exact: true });
  await enter(portDialog.getByRole('textbox', { name: 'Name', exact: true }), 'Command');
  await expect(portDialog).toBeHidden();
  const savedModel = (await saved()).project.documents?.model?.payload as unknown as ModelDocument;
  const command = savedModel.ports?.outputs?.find(({ name }) => name === 'Command');
  expect(command).toBeDefined();
  const commandOutput = { owner: 'outputs', port: command!.id };
  await connect(page, linear.locator('[data-handleid="output:output"]'),
    output.locator('[aria-label="Command"].react-flow__handle'), projectPath,
    { op: 'connect', source: linearOutput, target: commandOutput },
    (wire) => matchesFields(wire.source, linearOutput) && matchesFields(wire.target, commandOutput));

  await expect(output.locator('[aria-label="Command"].react-flow__handle')).toBeVisible();
  await tab('Objective').click();
  await objective.getByRole('combobox', { name: 'Measure source', exact: true }).selectOption({ label: 'Command' });
  await objective.getByRole('button', { name: 'Add Term', exact: true }).click();
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(Command)');
  await enter(objective.getByRole('textbox', { name: 'Weight', exact: true }), '2');
  await expect.poll(async () => (await saved()).project.documents?.objective?.payload).toMatchObject({
    terms: [expect.objectContaining({ weight: expect.objectContaining({ kind: 'literal', value: 2 }) })],
  });
  await objective.focus();
  await Promise.all([patch(), page.keyboard.press('Meta+Z')]);
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('1');
  await objective.focus();
  await Promise.all([patch(), page.keyboard.press('Meta+Shift+Z')]);
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('2');
  await tab('Training').click();
  await edit(panel('Training').getByRole('textbox', { name: 'N Batches', exact: true }), '2');
  await edit(panel('Training').getByRole('textbox', { name: 'Batch Size', exact: true }), '4');
  await expect.poll(async () => (await saved()).project.steps?.train.base).toMatchObject({
    inline: { training_config: { n_batches: 2, batch_size: 4 } },
  });

  const reopened = await saved();
  const reopenedModel = reopened.project.documents?.model?.payload as unknown as ModelDocument;
  expect(Object.values(reopenedModel.interior.wires ?? {})).toEqual(expect.arrayContaining([
    expect.objectContaining({ target: inputToCell }),
    expect.objectContaining({ source: gruOutput, target: linearInput }),
    expect.objectContaining({ source: linearOutput, target: plantForce }),
    expect.objectContaining({ source: linearOutput, target: commandOutput }),
  ]));
  const runsPath = projectPath + '/train-runs';
  const reviewed = async () => {
    const response = await request.get(runsPath + '?window_map=true&revision=' + reopened.revision);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const before = await reviewed();
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toHaveLength(1);
  await page.reload();
  await mapStep(page, 'train').click();
  await tab('Objective').click();
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(Command)');
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('2');
  await tab('Runs').click();
  await runs.getByRole('table', { name: 'Runs', exact: true }).getByRole('button').first().click();
  const submitted = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === runsPath);
  await runs.getByRole('button', { name: 'Run', exact: true }).click();
  const response = await submitted;
  expect(response.status(), await response.text()).toBe(202);
  const { items } = await response.json() as StudioJobSubmission;
  expect(items).toEqual([expect.objectContaining({ disposition: 'admitted', job_id: expect.any(String) })]);
  await expect.poll(async () => (await reviewed()).rows[0]?.attempts[0]?.status, { timeout: 20_000 }).toBe('succeeded');
  const [completed] = (await reviewed()).rows;
  expect(completed.request_id).toBe(before.rows[0].request_id);
  expect(completed.attempts).toEqual([expect.objectContaining({
    job_id: items[0].job_id, status: 'succeeded', progress: { completed_updates: 2, total_updates: 2 },
    successful_receipt: expect.objectContaining({ manifest_sha256: expect.any(String) }),
  })]);
  const jobs = await request.get('/api/jobs', { params: { project: snapshot.project_id } });
  expect(jobs.ok(), await jobs.text()).toBe(true);
  const { cursor } = await jobs.json() as StudioJobSnapshot;
  const history = await request.get('/api/events/history', { params: { project: snapshot.project_id, through: cursor } });
  expect(history.ok(), await history.text()).toBe(true);
  const { events } = await history.json() as StudioJobEventHistory;
  const last = events.filter((event) => event.kind === 'progress' && event.job_id === items[0].job_id)
    .flatMap((event) => event.data.metric_interval ? [event.data.metric_interval] : [])
    .at(-1) as { end_inclusive: number; metrics: { train_loss: { last: number } } } | undefined;
  expect(last?.end_inclusive).toBe(2);
  expect(Number.isFinite(last?.metrics.train_loss.last)).toBe(true);
  await page.reload();
  await mapStep(page, 'train').click();
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  console.info('First managed Train', JSON.stringify({
    revision: reopened.revision, request: completed.request_id, attempt: completed.attempts[0], trainLoss: last?.metrics.train_loss.last,
  }));
});
