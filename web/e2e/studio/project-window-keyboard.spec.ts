import { expect, test, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, ModelFieldResponse, ObjectiveDocument, PythonTaskSelection, StudioActivityJob, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainKey, StudioSavedTrainRuns, WorkspaceReplayTrial } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const region = (page: Page, id: string) => page.locator(`[data-studio-focus-region="${id}"]:visible`);
const command = (page: Page, id: string) => page.locator(`button[data-command-id="${id}"]:visible`).first();

function alternateModelFiles(files: Awaited<ReturnType<typeof fixture>>, id: string) {
  const [model, task, objective, train] = structuredClone(files);
  function rehome(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(rehome);
    if (typeof value === 'string') {
      return value.replaceAll(`"document":"${model.id}"`, `"document":"${id}-model"`);
    }
    if (value === null || typeof value !== 'object') return value;
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key,
      key === 'document' && entry === model.id ? `${id}-model` : rehome(entry),
    ]));
  }
  const trial = train.step.base.inline.trial_documents;
  trial.model_name = `${id}-model`;
  trial.model = { document: `${id}-model` };
  trial.task = { document: `${id}-task` };
  trial.objective = { document: `${id}-objective` };
  train.id = id;
  return [{ ...model, id: `${id}-model` },
    { ...task, id: `${id}-task`, document: rehome(task.document) },
    { ...objective, id: `${id}-objective`, document: rehome(objective.document) }, train];
}

async function tabTo(page: Page, target: Locator, limit = 100) {
  await expect(target).toBeVisible();
  for (let index = 0; index < limit; index += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press('Tab');
  }
  await expect(target).toBeFocused();
}

async function activate(page: Page, target: Locator, key = 'Enter') {
  await expect(target).toBeEnabled();
  await tabTo(page, target);
  await page.keyboard.press(key);
}

async function palette(page: Page, id: string, label: string) {
  await page.keyboard.press('Meta+K');
  const dialog = page.getByRole('dialog', { name: 'Command Palette' });
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.type(label);
  const item = dialog.getByRole('option', { name: label, exact: true });
  await expect(item).toHaveAttribute('data-command-id', id);
  await expect(item).toBeEnabled();
  await page.keyboard.press('Enter');
  await expect(dialog).toBeHidden();
}

async function number(page: Page, target: Locator, value: string) {
  await expect(target).toBeEditable();
  await tabTo(page, target);
  // A saved edit can refresh the field projection while focus moves.
  await expect(target).toBeEditable();
  await expect(target).toBeFocused();
  await page.keyboard.press('Meta+A');
  await page.keyboard.type(value);
  await expect(target).toHaveValue(value);
  await page.keyboard.press('Enter');
}

function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

async function readSaved(request: APIRequestContext, projectPath: string) {
  const response = await request.get(projectPath);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioProjectSnapshot;
}

test('an empty saved project starts its first Train by pointer', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  expect(snapshot.project.steps ?? {}).toEqual({});
  expect(snapshot.project.documents ?? {}).toEqual({});
  await page.goto(`/?project=${snapshot.project_id}`);

  const add = page.getByRole('button', { name: 'New Step', exact: true });
  await expect(add).toBeEnabled();
  await expect(add).toHaveText('');
  await add.click();
  const chooser = page.getByRole('dialog', { name: 'New Step', exact: true });
  const train = chooser.getByRole('button', { name: 'Train', exact: true });
  await expect(chooser.getByRole('button')).toHaveCount(1);
  await expect(train).toBeFocused();
  const creation = page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/files` &&
    response.request().postDataJSON().file.kind === 'step');
  await train.click();
  const created = await creation;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: saved } = await created.json() as StudioProjectDocument;
  expect(Object.keys(saved.project.steps ?? {})).toEqual(['train']);
  expect(saved.project.steps?.train).toMatchObject({
    kind: 'train', base: { inline: { trial_documents: {
      model_name: 'model', model: { document: 'model' }, python_task: { document: 'task' },
      objective: { document: 'objective' },
    } } },
  });
  expect(saved.project.documents).toMatchObject({
    model: { schema: 'feedbax.spec.model', payload: { ports: {}, interior: {} } },
    task: { schema: 'feedbax.spec.python_task', payload: { type: 'feedbax.task.reaching.v1', params: {} } },
    objective: { schema: 'feedbax.spec.objective_document', payload: { terms: [] } },
  });

  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(chooser).toBeHidden();
  await expect(model).toBeFocused();
  await expect(model.locator('.react-flow')).toBeVisible();
  await expect(model.locator('[data-boundary="true"]')).toHaveCount(2);
  await expect(model.getByRole('button', { name: 'model', exact: true })).toBeVisible();
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await expect(card.getByRole('button', { name: "1 Can't run", exact: true })).toBeVisible();

  async function readPlan(revision: string) {
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  const plan = await readPlan(saved.revision);
  expect(plan.rows).toHaveLength(1);
  const [run] = plan.rows;
  expect(run).toMatchObject({ key: { step: 'train', row: 'default' }, status: 'cannot_run', attempts: [] });
  expect(plan.window_map?.rows.map(({ key }) => key)).toEqual([run.key]);
  expect(plan.window_map?.problems).toMatchObject([{ path: '/trial_documents/model/interior/components' }]);
  const refused = await request.post(`${projectPath}/train-runs`, {
    data: { revision: saved.revision, run_keys: [run.key], token: crypto.randomUUID() },
  });
  expect(refused.status(), await refused.text()).toBe(202);
  expect((await refused.json() as StudioJobSubmission).items).toMatchObject([
    { disposition: 'rejected', reason: '/trial_documents/model/interior/components: Model has no components.' },
  ]);

  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  await tab('Training').click();
  await expect(page.getByRole('textbox', { name: 'N Batches', exact: true })).toBeEnabled();
  await tab('Task').click();
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const reach = task.getByRole('spinbutton', { name: 'Target x (m)', exact: true });
  await expect(reach).toHaveValue('0.15');
  await expect(task.getByRole('alert')).toHaveCount(0);
  await enter(reach, '0.6');
  await expect.poll(async () => (await readSaved(request, projectPath)).project.documents?.task?.payload)
    .toMatchObject({ params: { x: 0.6 } });
  const current = await readSaved(request, projectPath);
  expect(current.revision).not.toBe(saved.revision);

  await page.reload();
  await expect(page).toHaveURL(new RegExp(`[?&]project=${snapshot.project_id}(?:&|$)`));
  await mapStep(page, 'train').click();
  await tab('Model').click();
  await expect(model.locator('.react-flow')).toBeVisible();
  await expect(model.getByRole('button', { name: 'model', exact: true })).toBeVisible();
  await expect(card.getByRole('button', { name: "1 Can't run", exact: true })).toBeVisible();
  await tab('Task').click();
  await expect(reach).toHaveValue('0.6');
  const persisted = await readSaved(request, projectPath);
  expect(persisted).toMatchObject({ project_id: snapshot.project_id, revision: current.revision, project: current.project });
  const reopened = await readPlan(persisted.revision);
  expect(reopened.rows).toMatchObject([{ key: run.key, status: 'cannot_run', attempts: [] }]);
  expect(reopened.window_map?.problems).toMatchObject([{ path: '/trial_documents/model/interior/components' }]);
});

test('authors, checkpoint-resumes and evaluates a subset from one fresh managed project', async ({ page, request }, info) => {
  test.setTimeout(100_000);
  const totalUpdates = 6000;
  const home = String(info.config.metadata.home);
  const creation = post(page, '/api/projects');
  await page.goto('/');
  const created = await creation;
  expect(created.status()).toBe(201);
  const { snapshot } = await created.json() as StudioProjectDocument;
  expect(snapshot.project.steps).toEqual({});
  expect(snapshot.project.documents).toEqual({});
  const projectPath = '/api/projects/' + snapshot.project_id;
  const runsPath = projectPath + '/train-runs';
  const savedProject = () => readSaved(request, projectPath);
  const plan = async (revision: string) => {
    const response = await request.get(runsPath, { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const setNumber = async (target: Locator, value: number, projectValue: (project: StudioProjectSnapshot['project']) => unknown) => {
    await number(page, target, String(value));
    await expect.poll(async () => projectValue((await savedProject()).project)).toBe(value);
  };
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const objective = page.getByRole('tabpanel', { name: 'Objective', exact: true });
  const training = page.getByRole('tabpanel', { name: 'Training', exact: true });
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const taskX = task.getByRole('spinbutton', { name: 'Target x (m)', exact: true });
  const nSteps = task.getByRole('spinbutton', { name: /N Steps/i });
  const weight = objective.getByRole('textbox', { name: 'Weight', exact: true });
  const nBatches = training.getByRole('textbox', { name: 'N Batches', exact: true });
  const batchSize = training.getByRole('textbox', { name: 'Batch Size', exact: true });
  const taskValues = (project: StudioProjectSnapshot['project']) => (
    project.documents?.task?.payload as PythonTaskSelection | undefined
  )?.params;
  const terms = (project: StudioProjectSnapshot['project']) =>
    (project.documents?.objective?.payload as unknown as ObjectiveDocument | undefined)?.terms ?? [];
  const config = (project: StudioProjectSnapshot['project']) => {
    const base = project.steps?.train?.base;
    return base && 'inline' in base
      ? base.inline.training_config as { n_batches?: number; batch_size?: number }
      : undefined;
  };

  await expect(page.getByRole('button', { name: 'New Step', exact: true })).toBeEnabled();
  await palette(page, 'newStep', 'New Step');
  const chooser = page.getByRole('dialog', { name: 'New Step', exact: true });
  await expect(chooser.getByRole('button', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(model).toBeFocused();
  await page.keyboard.press('Meta+3');
  await setNumber(taskX, 0.22, (project) => taskValues(project)?.x);
  await setNumber(nSteps, 12, (project) => taskValues(project)?.n_steps);

  await page.keyboard.press('Meta+2');
  const component = (name: string) => model.locator('.react-flow__node').filter({
    has: page.getByRole('article', { name, exact: true }),
  });
  async function add(name: string, category: string) {
    await expect(model).toBeFocused();
    await expect(model.getByRole('button', { name: 'Lock', exact: true })).toBeEnabled();
    await page.keyboard.press('Shift+A');
    const add = page.getByRole('dialog', { name: 'Add Component', exact: true });
    await expect(add.getByRole('textbox', { name: 'Components', exact: true })).toBeFocused();
    await page.keyboard.type(name);
    await activate(page, add.getByRole('region', { name: category, exact: true })
      .getByRole('button', { name: 'Add ' + name, exact: true }));
    await expect(add).toBeHidden();
    await expect(component(name)).toBeVisible();
    return component(name);
  }
  const linear = await add('Linear', 'Runtime');
  const linearId = await linear.getAttribute('data-id');
  expect(linearId).toBeTruthy();
  function modelField(name: string, field: string, value: number) {
    return setNumber(inspector.getByRole('textbox', { name, exact: true }), value, (project) => {
      const model = project.documents?.model?.payload as unknown as ModelDocument | undefined;
      const component = model?.interior.components?.[linearId!];
      return component?.kind === 'leaf' ? component.fields?.[field] : undefined;
    });
  }
  await modelField('Input size', 'input_size', 2);
  await modelField('Output size', 'output_size', 2);
  await page.keyboard.press('Meta+2');
  const plant = await add('PointMass', 'Mechanics');
  const plantId = await plant.getAttribute('data-id');
  expect(plantId).toBeTruthy();
  const output = model.locator('[data-id="@outputs"]');
  await page.keyboard.press('Meta+2');
  await expect(model).toBeFocused();
  await activate(page, output.getByRole('button', { name: 'Add Port', exact: true }));
  const portDialog = page.getByRole('dialog', { name: 'Add Port', exact: true });
  await number(page, portDialog.getByRole('textbox', { name: 'Name', exact: true }), 'Command');
  await expect(portDialog).toBeHidden();

  async function connect(source: Locator, target: Locator) {
    const { revision } = await savedProject();
    await page.keyboard.press('Meta+2');
    await expect(model).toBeFocused();
    await expect(source).toBeVisible();
    await expect(source).toHaveAttribute('aria-disabled', 'false');
    await expect(source).toHaveAttribute('tabindex', '0');
    for (let hop = 0; hop < 20 && !await source.evaluate((handle) => handle === document.activeElement); hop += 1) {
      await page.keyboard.press('Tab');
      expect(await model.evaluate((panel) => panel.contains(document.activeElement))).toBe(true);
    }
    await expect(source).toBeFocused();
    const reviewed = post(page, projectPath + '/model/connections');
    await page.keyboard.press('Enter');
    const review = await reviewed;
    expect(review.ok(), await review.text()).toBe(true);
    expect(review.request().postDataJSON()).toMatchObject({ revision });
    await expect(source).toHaveAttribute('aria-disabled', 'true');
    await expect(target).toHaveAttribute('aria-disabled', 'false');
    await expect(source).toBeFocused();
    for (let count = 0; count < 8 && !await target.evaluate((node) => node === document.activeElement); count += 1) {
      await page.keyboard.press('ArrowRight');
    }
    await expect(target).toBeFocused();
    const edit = post(page, projectPath + '/model/edit');
    await page.keyboard.press('Enter');
    const response = await edit;
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json();
    expect(result.status, result.problem?.detail).toBe('applied');
  }
  const source = linear.locator('[data-handleid="output:output"]');
  await connect(model.locator('[data-handleid="output:@task:reach_target"]'), linear.locator('[data-handleid="input:input"]'));
  await connect(source, plant.locator('[data-handleid="input:force"]'));
  await connect(source, output.locator('[aria-label="Port Command"].react-flow__handle'));

  await page.keyboard.press('Meta+4');
  const measure = objective.getByRole('combobox', { name: 'Measure Source', exact: true });
  await expect(measure.locator('option:checked')).toHaveText('Command');
  await activate(page, objective.getByRole('button', { name: 'Add Term', exact: true }));
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(Command)');
  await setNumber(weight, 2, (project) => {
    const authored = terms(project)[0]?.weight;
    return authored && 'value' in authored ? authored.value : undefined;
  });
  await page.keyboard.press('Meta+5');
  await setNumber(nBatches, totalUpdates, (project) => config(project)?.n_batches);
  await expect(nBatches).toBeEditable();
  await setNumber(batchSize, 4, (project) => config(project)?.batch_size);

  const saved = await savedProject();
  expect(saved.project_id).toBe(snapshot.project_id);
  expect(saved.revision).not.toBe(snapshot.revision);
  expect(saved.project.steps?.train?.base).toMatchObject({ inline: {
    trial_documents: { model: { document: 'model' }, python_task: { document: 'task' }, objective: { document: 'objective' } },
    training_config: { n_batches: totalUpdates, batch_size: 4 },
  } });
  const savedModel = saved.project.documents?.model?.payload as unknown as ModelDocument;
  expect(savedModel.interior.components).toMatchObject({
    [linearId!]: { kind: 'leaf', type: 'feedbax.Linear', fields: { input_size: 2, output_size: 2 } },
    [plantId!]: { kind: 'leaf', type: 'feedbax.PointMass' },
  });
  const reachPort = savedModel.ports.inputs?.find(({ name }) => name === 'reach_target');
  const commandPort = savedModel.ports.outputs?.find(({ name }) => name === 'Command');
  expect(reachPort?.id).toBeTruthy();
  expect(commandPort?.id).toBeTruthy();
  const linearOutput = { owner: 'component', component: linearId, port: 'output' };
  const wires = Object.values(savedModel.interior.wires ?? {});
  expect(wires).toHaveLength(3);
  expect(wires).toEqual(expect.arrayContaining([
    expect.objectContaining({
      source: { owner: 'inputs', port: reachPort!.id },
      target: { owner: 'component', component: linearId, port: 'input' },
    }),
    expect.objectContaining({
      source: linearOutput, target: { owner: 'component', component: plantId, port: 'force' },
    }),
    expect.objectContaining({ source: linearOutput, target: { owner: 'outputs', port: commandPort!.id } }),
  ]));
  expect(saved.project.documents?.task?.payload).toMatchObject({ params: { x: 0.22, n_steps: 12 } });
  const [term] = terms(saved.project);
  expect(terms(saved.project)).toHaveLength(1);
  expect(term).toMatchObject({
    id: expect.any(String),
    measure: { function: 'sqnorm', args: [{ kind: 'name', reference: expect.any(String) }] },
    weight: { value: 2 },
  });
  const measureArgument = 'args' in term.measure ? term.measure.args?.[0] : undefined;
  expect(JSON.parse(measureArgument && 'reference' in measureArgument ? measureArgument.reference : 'null')).toEqual({
    document: 'model', composites: [], element: { kind: 'port', end: { owner: 'outputs', port: commandPort!.id } },
  });
  const before = await plan(saved.revision);
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toMatchObject([{ key: { variant: 'default', step: 'train', row: 'default' },
    request_id: expect.any(String), status: 'not_run', attempts: [] }]);
  async function reopen() {
    await page.reload();
    await expect(page).toHaveURL(new RegExp(`[?&]project=${snapshot.project_id}(?:&|$)`));
    await activate(page, mapStep(page, 'train'));
    await expect(runs).toBeVisible();
  }
  await reopen();
  expect(await savedProject()).toMatchObject({ project_id: saved.project_id, revision: saved.revision, project: saved.project });
  expect((await plan(saved.revision)).rows).toEqual(before.rows);
  await page.keyboard.press('Meta+4');
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(Command)');
  await expect(weight).toHaveValue('2');
  await page.keyboard.press('Meta+5');
  await expect(nBatches).toHaveValue(String(totalUpdates));
  await expect(batchSize).toHaveValue('4');
  const train = page.getByRole('article', { name: 'Train train', exact: true });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activate(page, rate.getByRole('menuitem', { name: 'Sweep', exact: true }));
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate', exact: true });
  await expect(rateText).toHaveValue('0.001,');
  await number(page, rateText, '0.001, 0.002, 0.003');
  await expect(rate.getByRole('textbox', { name: 'Learning Rate value 3', exact: true })).toHaveValue('0.003');
  const swept = await savedProject();
  const sourceRows = (await plan(swept.revision)).window_map!.rows.filter(({ key }) => key.step === 'train');
  expect(sourceRows.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.002, 0.003]);
  const [selected] = sourceRows;
  await page.keyboard.press('Control+Meta+2');
  await activate(page, train.getByRole('button', { name: 'Sweep learning-rate', exact: true }));
  const slow = train.getByRole('listitem').getByRole('button', { name: 'Select learning-rate 0.001', exact: true });
  await activate(page, slow);
  await expect(slow).toHaveAttribute('aria-pressed', 'true');
  await expect(train).toContainText('1 of 3');

  async function submit(keys: StudioSavedTrainKey[], revision: string) {
    const response = post(page, runsPath);
    await palette(page, 'run', 'Run');
    const admitted = await response;
    expect(admitted.status(), await admitted.text()).toBe(202);
    expect(admitted.request().postDataJSON()).toMatchObject({ revision, run_keys: keys });
    const submission: StudioJobSubmission = await admitted.json();
    expect(submission.project_id).toBe(saved.project_id);
    expect(submission.items).toMatchObject(keys.map(() => ({ disposition: 'admitted', job_id: expect.any(String) })));
    return submission.items;
  }

  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  await page.keyboard.press('Meta+1');
  const trainTable = runs.getByRole('table', { name: 'Runs', exact: true });
  const producerButton = trainTable.getByRole('row').filter({
    has: page.getByRole('cell', { name: '0.001', exact: true }),
  }).getByRole('button');
  await activate(page, producerButton);
  const [initial] = await submit([selected.key], swept.revision);
  await expect(runs.getByRole('button', { name: 'Open Run', exact: true })).toBeEnabled();
  await palette(page, 'openRun', 'Open Run');
  const loss = results.getByRole('img', { name: 'Loss', exact: true });
  await expect(loss.locator(`line[data-attempt="${initial.job_id}"] title`).first()).toBeAttached({ timeout: 16_000 });
  const stopping = post(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stop = await stopping;
  expect(stop.status(), await stop.text()).toBe(202);
  expect(stop.request().postDataJSON()).toEqual({ job_ids: [initial.job_id], mode: 'checkpoint' });

  async function readSelected(revision: string) {
    const rows = (await plan(revision)).rows;
    return rows.find(({ key }) => key.step === selected.key.step && key.row === selected.key.row)!;
  }
  await expect.poll(async () => (await readSelected(swept.revision)).status, { timeout: 12_000 }).toBe('stopped');
  const witness = 'web.e2e.studio.fixtures.managed_lifecycle';
  const observe = async (action: string, ...args: string[]) =>
    JSON.parse(await runProjectFixture(witness, action, home, saved.project_id, initial.job_id!, ...args));
  const paused = await observe('restart');
  expect(paused.checkpoint.completed_updates).toBeGreaterThan(0);
  expect(paused.checkpoint.completed_updates).toBeLessThan(totalUpdates);
  expect(paused.new_host).not.toBe(paused.old_host);

  await page.reload();
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const resumed = post(page, `/api/jobs/${initial.job_id}/resume`);
  await activity.getByRole('button', { name: 'Resume', exact: true }).click();
  const response = await resumed;
  expect(response.status(), await response.text()).toBe(202);
  const continued: StudioActivityJob = await response.json();
  const producer = (await readSelected(swept.revision)).request_id;
  expect(continued).toMatchObject({ project_id: saved.project_id, request_id: producer, continues: initial.job_id, run_key: initial.run_key });
  expect(continued.job_id).not.toBe(initial.job_id);
  const binding = await observe('resume', continued.job_id);
  expect(binding).toEqual({ host: paused.new_host, checkpoint: paused.checkpoint });
  await expect.poll(async () => (await readSelected(swept.revision)).status, { timeout: 22_000 }).toBe('succeeded');
  const trained = await readSelected(swept.revision);
  expect(trained.result).toMatchObject({
    job_id: continued.job_id,
    request_id: producer,
    progress: { completed_updates: totalUpdates, total_updates: totalUpdates },
    receipt: { manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
  });
  expect(trained.attempts.map(({ job_id }) => job_id)).toEqual([initial.job_id, continued.job_id]);

  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await page.keyboard.press('Control+Meta+2');
  await activate(page, mapStep(page, 'train'));
  const sweep = train.getByRole('button', { name: 'Sweep learning-rate', exact: true });
  if (!(await slow.isVisible())) await activate(page, sweep);
  await activate(page, slow);
  await expect(slow).toHaveAttribute('aria-pressed', 'true');
  await expect(train).toContainText('1 of 3');
  const creating = post(page, projectPath + '/files');
  await activate(page, train.getByRole('button', { name: 'New Step train', exact: true }));
  const choice = page.getByRole('dialog', { name: 'New Step', exact: true });
  await activate(page, choice.getByRole('button', { name: 'Evaluate', exact: true }));
  const createdEvaluate = await creating;
  expect(createdEvaluate.status(), await createdEvaluate.text()).toBe(201);
  const { snapshot: evaluated }: StudioProjectDocument = await createdEvaluate.json();
  expect(evaluated.project.steps.evaluate.inputs?.subject).toEqual({
    step: 'train', checkpoint: 'last', where: { axes: { 'learning-rate': [selected.coordinates['learning-rate'].value_id] } },
  });
  await expect(page.getByRole('region', { name: 'Inputs', exact: true })).toContainText('1 of 3');
  const projection = await plan(evaluated.revision);
  const [pending] = projection.window_map!.evaluations!;
  expect(projection.window_map?.evaluations).toHaveLength(1);
  const [evaluation] = await submit([pending.key], evaluated.revision);
  const evaluatedRow = async () => (await plan(evaluated.revision)).window_map!.evaluations![0];
  await expect.poll(async () => (await evaluatedRow()).status, { timeout: 18_000 }).toBe('succeeded');
  const result = await evaluatedRow();
  expect(result.result).toMatchObject({
    job_id: evaluation.job_id,
    receipt: { manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
    checkpoint_source: { job_id: continued.job_id, request_id: producer, completed_updates: totalUpdates },
  });
  const [trial] = result.trials;
  expect(trial).toBeTruthy();
  expect(result.trials).toHaveLength(1);

  await page.reload();
  await activate(page, mapStep(page, 'evaluate'));
  await page.keyboard.press('Meta+1');
  const label = projection.window_map!.rows.find(({ key }) => key.step === pending.key.step && key.row === pending.key.row)!.label;
  await activate(page, runs.getByRole('table', { name: 'Runs', exact: true }).getByRole('button', { name: label, exact: true }));
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  await expect(runs.getByRole('region', { name: 'Made From', exact: true })).toContainText('1 of 3');
  expect(await evaluatedRow()).toEqual(result);
  await info.attach('managed-checkpoint-and-evaluate', {
    body: JSON.stringify({ project: saved.project_id, paused, continued, result }),
    contentType: 'application/json',
  });
  const sceneTab = page.getByRole('tab', { name: 'Scene', exact: true });
  await expect(sceneTab).toHaveAttribute('aria-keyshortcuts', 'Meta+6');
  const [replayed] = await Promise.all([
    post(page, projectPath + '/evaluate-replay'),
    page.keyboard.press('Meta+6'),
  ]);
  await expect(sceneTab).toHaveAttribute('aria-selected', 'true');
  expect(replayed.ok(), await replayed.text()).toBe(true);
  expect(replayed.request().postDataJSON()).toMatchObject({ revision: evaluated.revision, key: result.key, job_id: evaluation.job_id, trial });
  const replay: WorkspaceReplayTrial = await replayed.json();
  const observed = await observe('result', continued.job_id, evaluated.revision, evaluation.job_id!, trial);
  expect(observed.receipt).toEqual(result.result!.receipt);
  expect(replay.manifest_refs.producer_manifest).toMatchObject({
    id: observed.receipt.manifest_id, metadata: { manifest_sha256: observed.receipt.manifest_sha256 },
  });
  expect(replay.identity.stable_id).toBe(trial);
  expect(replay.tracks![0].samples).toEqual(observed.samples);
  expect(observed.samples.length).toBeGreaterThan(1);
  expect(observed.samples.every((sample: number[]) => sample.length === 2 && sample.every(Number.isFinite))).toBe(true);
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toBeVisible();
  await info.attach('authenticated-scene-replay', {
    body: JSON.stringify({ replay, observed }),
    contentType: 'application/json',
  });
});

test('Model edits follow the lit users of a saved shared document', async ({ page, request }) => {
  test.setTimeout(45_000);
  const [model, task, objective, train] = await fixture('managed-files');
  const alpha = { ...train, id: 'alpha' };
  const beta = { ...train, id: 'beta' };
  const gamma = { kind: 'step', id: 'gamma', step: { kind: 'train', base: { step: 'alpha' } } };
  const { snapshot, projectPath } = await seedManagedProject(request, [
    model, task, objective, alpha, beta, gamma, ...alternateModelFiles([model, task, objective, train], 'other'),
  ]);

  async function readPlan(revision: string) {
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    const plan = await response.json() as StudioSavedTrainRuns;
    expect(plan.rows.map(({ key }) => key.step).sort()).toEqual(['alpha', 'beta', 'gamma', 'other']);
    return Object.fromEntries(plan.rows.map((row) => [row.key.step, row]));
  }

  async function readMass(revision: string, run: StudioSavedTrainRuns['rows'][number]['key']) {
    const response = await request.post(`${projectPath}/model/fields`, { data: { revision, run,
      owner: { document: model.id, composites: [], element: { kind: 'component', id: 'plant' } },
    } });
    expect(response.ok(), await response.text()).toBe(true);
    const field = (await response.json() as ModelFieldResponse).fields?.find(({ label }) => label === 'Mass');
    return field?.effective ?? field?.text;
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'alpha')).toBeVisible();
  await page.keyboard.press('Meta+2');
  const modelView = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const count = modelView.locator('header').getByRole('status').filter({ hasText: /^\s*· \d+ of \d+\s*$/ });
  const card = (step: string) => page.getByRole('article', { name: `Train ${step}`, exact: true });
  await expect(modelView).toBeFocused();
  await expect(count).toHaveText('· 3 of 3');
  await expect(modelView.locator('header').first()).toContainText(model.id);
  for (const step of ['alpha', 'beta', 'gamma']) await expect(card(step)).toHaveAttribute('data-document-scope', 'lit');
  await expect(card('other')).toHaveAttribute('data-document-scope', 'faint');
  await expect(mapStep(page, 'alpha')).toHaveAttribute('aria-current', 'step');
  const initial = await readPlan(snapshot.revision);

  const component = modelView.getByRole('article', { name: 'Point mass', exact: true }).locator('header');
  await component.click();
  const inspector = region(page, 'inspector');
  const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
  const documentValue = (saved: StudioProjectSnapshot) => saved.project.documents?.[model.id].payload;
  const valueAt = (mass: number) => ({ interior: { components: { plant: { fields: { mass } } } } });

  async function savedModelWrite(writing: ReturnType<typeof post>) {
    const response = await writing;
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    expect(result.status).toBe('applied');
    if (result.status !== 'applied') throw new Error(result.problem.detail);
    return result.snapshot;
  }

  await expect(mass).toHaveValue('1');
  const sharedWrite = post(page, `${projectPath}/model/edit`);
  await enter(mass, '2');
  const shared = await savedModelWrite(sharedWrite);
  expect(documentValue(shared)).toMatchObject(valueAt(2));
  const afterShared = await readPlan(shared.revision);
  for (const step of ['alpha', 'beta', 'gamma']) expect(afterShared[step].request_id).not.toEqual(initial[step].request_id);
  expect(afterShared.other.request_id).toEqual(initial.other.request_id);

  await expect(count).toHaveText('· 3 of 3');
  await activate(page, mapStep(page, 'gamma'), 'Shift+Enter');
  await activate(page, mapStep(page, 'alpha'), 'Shift+Enter');
  await expect(count).toHaveText('· 2 of 3');
  await expect(card('gamma')).toHaveAttribute('data-document-scope', 'plain');
  await expect(card('alpha')).toHaveAttribute('data-document-scope', 'lit');
  await expect(mapStep(page, 'alpha')).toHaveAttribute('aria-current', 'step');

  await page.keyboard.press('Meta+1');
  await expect(page.getByRole('tabpanel', { name: 'Runs', exact: true })).toBeFocused();
  await expect(card('beta')).not.toHaveAttribute('data-document-scope');
  await page.keyboard.press('Meta+2');
  await expect(modelView).toBeFocused();
  await expect(count).toHaveText('· 2 of 3');
  await expect(mass).toHaveValue('2');

  const rejected = post(page, `${projectPath}/model/fields`);
  await enter(mass, '3');
  const refusal = await (await rejected).json() as ModelFieldResponse;
  expect(refusal.problems?.[0].reason).toBe('Select all steps that use this Model to change its document value.');
  await expect(inspector.getByRole('alert')).toContainText(refusal.problems![0].reason);
  expect((await readSaved(request, projectPath)).revision).toBe(shared.revision);
  for (const step of ['alpha', 'beta', 'gamma']) expect(await readMass(shared.revision, afterShared[step].key)).toBe('2');
  await mass.press('Escape');

  await activate(page, mapStep(page, 'gamma'), 'Shift+Enter');
  await expect(count).toHaveText('· 3 of 3');
  const restoring = post(page, `${projectPath}/model/edit`);
  await enter(mass, '4');
  const restored = await savedModelWrite(restoring);
  expect(documentValue(restored)).toMatchObject(valueAt(4));
  const afterRestore = await readPlan(restored.revision);
  for (const step of ['alpha', 'beta', 'gamma']) expect(afterRestore[step].request_id).not.toEqual(afterShared[step].request_id);
  expect(afterRestore.other.request_id).toEqual(afterShared.other.request_id);

  const undoing = post(page, `${projectPath}/model/undo`);
  await modelView.focus();
  await page.keyboard.press('Meta+Z');
  const beforeReload = await savedModelWrite(undoing);
  expect(documentValue(beforeReload)).toMatchObject(valueAt(2));
  await expect(mass).toHaveValue('2');

  await page.reload();
  await expect(count).toHaveText('· 3 of 3');
  await component.click();
  await expect(mass).toHaveValue('2');
  const reopened = await readSaved(request, projectPath);
  expect(reopened.revision).toBe(beforeReload.revision);
  const plan = await readPlan(reopened.revision);
  for (const step of ['alpha', 'beta', 'gamma', 'other']) expect(plan[step].request_id).toEqual(afterShared[step].request_id);
});

test('the window commands, tab rail and selection history work without a pointer', async ({ page, request }) => {
  const [model, task, objective, train, successor] = await fixture('managed-files');
  const { snapshot } = await seedManagedProject(request, [
    model, task, objective, train, successor, ...alternateModelFiles([model, task, objective, train], 'third'),
  ]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('F6');
  await expect(region(page, 'top')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'map')).toBeFocused();
  await activate(page, mapStep(page, 'train'));
  await activate(page, mapStep(page, 'successor'));
  await activate(page, mapStep(page, 'third'));

  await page.keyboard.press('Meta+BracketLeft');
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await palette(page, 'back', 'Back');
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Meta+BracketRight');
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await palette(page, 'forward', 'Forward');
  await expect(mapStep(page, 'third')).toHaveAttribute('aria-pressed', 'true');

  await page.keyboard.press('Control+Meta+1');
  await expect(command(page, 'showMap')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'primary')).toHaveCount(0);
  await palette(page, 'showBoth', 'Show Both');
  await expect(command(page, 'showBoth')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'map')).toBeFocused();
  await page.keyboard.press('Control+Meta+3');
  await expect(command(page, 'showDetails')).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'primary')).toBeFocused();
  await expect(mapStep(page, 'third')).toBeHidden();
  await page.keyboard.press('Control+Meta+2');

  await page.keyboard.press('Control+Meta+S');
  await expect(command(page, 'showSteps')).toHaveAccessibleName('Show Steps');
  await expect(region(page, 'steps')).toHaveCount(0);
  await page.keyboard.press('Control+Meta+S');
  await expect(command(page, 'showSteps')).toHaveAccessibleName('Hide Steps');
  await expect(region(page, 'steps')).toBeFocused();

  await page.keyboard.press('F6');
  const tabs = page.getByRole('tablist', { name: 'Details' });
  await expect(tabs).toHaveAttribute('aria-orientation', 'vertical');
  await expect(tabs.getByRole('tab', { name: 'Runs', exact: true })).toBeFocused();
  await page.keyboard.press('ArrowDown');
  const modelTab = tabs.getByRole('tab', { name: 'Model', exact: true });
  await expect(modelTab).toBeFocused();
  await expect(modelTab).toHaveAttribute('tabindex', '0');
  await expect(tabs.getByRole('tab', { name: 'Runs', exact: true })).toHaveAttribute('tabindex', '-1');
  await expect(page.getByRole('tooltip')).toHaveText('Model · ⌘2');
  await expect(modelTab).not.toHaveAttribute('title', /.+/);
  await page.keyboard.press('Enter');
  await expect(modelTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel', { name: 'Model', exact: true })).toBeFocused();

  await page.keyboard.press('Meta+0');
  await expect(region(page, 'drawer')).toHaveCount(0);
  await palette(page, 'showDrawer', 'Show Drawer');
  await expect(region(page, 'drawer')).toBeFocused();
  const plant = region(page, 'drawer').getByRole('button', { name: 'Point mass', exact: true });
  await activate(page, plant);
  await expect(plant).toHaveAttribute('aria-pressed', 'true');
  await expect(region(page, 'inspector').getByRole('spinbutton', { name: 'Edit Mass', exact: true })).toHaveValue('1');
  await expect(plant).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toHaveCount(0);
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'bottom')).toBeFocused();
  await page.keyboard.press('Shift+F6');
  await expect(region(page, 'inspector')).toBeFocused();

  await page.keyboard.press('Meta+Backslash');
  await expect(region(page, 'secondary')).toBeFocused();
  await expect(page.getByRole('tabpanel', { name: 'Scene', exact: true })).toBeVisible();
  await page.keyboard.press('Alt+Meta+Backslash');
  await expect(region(page, 'primary')).toBeFocused();
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  await palette(page, 'switchSlot', 'Switch Slot');
  await expect(region(page, 'secondary')).toBeFocused();
  await palette(page, 'splitDetails', 'Close Slot');
  await expect(region(page, 'secondary')).toHaveCount(0);
  await expect(region(page, 'primary')).toBeFocused();

  await page.keyboard.press('Meta+K');
  const dialog = page.getByRole('dialog', { name: 'Command Palette' });
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(region(page, 'primary')).toBeFocused();

  await palette(page, 'activity', 'Activity');
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const closeActivity = activity.getByRole('button', { name: 'Close', exact: true });
  await expect(closeActivity).toBeFocused();
  await page.keyboard.press('Meta+K');
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await expect(dialog.locator('[data-command-id="run"], [data-command-id="stop"]')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(closeActivity).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(activity).toBeHidden();
});

test('the keyboard authors, runs and stops a strict checkpoint continuation and follows Results', async ({ page, request }) => {
  test.setTimeout(55_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const runsPath = `${projectPath}/train-runs`;
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('Meta+5');
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate', exact: true });
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activate(page, rate.getByRole('menuitem', { name: 'Sweep', exact: true }));
  await expect(rateText).toHaveValue('0.001,');
  await number(page, rateText, '0.001, 0.002, 0.003');
  await expect(rate.getByRole('textbox', { name: 'Learning Rate value 3', exact: true })).toHaveValue('0.003');
  const savedSweep = await request.get(projectPath);
  expect(savedSweep.ok()).toBe(true);
  const swept = await savedSweep.json() as StudioProjectSnapshot;
  expect(swept.revision).not.toBe(snapshot.revision);
  await page.reload();
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('Control+Meta+2');
  await activate(page, card.getByRole('button', { name: 'Sweep learning-rate', exact: true }));
  const value = (name: string) => card.getByRole('listitem').getByRole('button', { name: `Select learning-rate ${name}`, exact: true });
  await activate(page, value('0.001'));
  await activate(page, value('0.003'), 'Shift+Enter');
  await expect(value('0.001')).toHaveAttribute('aria-pressed', 'true');
  await expect(value('0.003')).toHaveAttribute('aria-pressed', 'true');
  await expect(value('0.002')).toHaveAttribute('aria-pressed', 'false');
  await expect(card).toContainText('2 of 3');
  const marks = card.getByRole('grid', { name: 'Runs' });
  await tabTo(page, marks.getByRole('rowheader').first().getByRole('button'));
  await page.keyboard.press('ArrowRight');
  await expect(marks.getByRole('gridcell').first().getByRole('button')).toBeFocused();

  const prefix = post(page, runsPath);
  await page.keyboard.press('Meta+R');
  const sourceRun = await prefix;
  expect(sourceRun.status(), await sourceRun.text()).toBe(202);
  const sourceKeys = sourceRun.request().postDataJSON().run_keys as StudioSavedTrainRuns['rows'][number]['key'][];
  expect(sourceKeys).toHaveLength(2);
  const producer = (await sourceRun.json() as StudioJobSubmission).items[0].job_id;
  await expect(card.getByText('2 Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await expect(card.getByText('1 Not run', { exact: true })).toBeVisible();

  await expect(card.getByRole('button', { name: 'New Step train' })).toHaveAttribute('data-command-id', 'newStep');
  const create = post(page, `${projectPath}/files`);
  await palette(page, 'newStep', 'New Step');
  const train = page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true });
  await expect(train).toBeFocused();
  await page.keyboard.press('Enter');
  const created = await create;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: authored } = await created.json() as StudioProjectDocument;
  const child = 'train-continuation';
  const plan = await (await request.get(`${runsPath}?revision=${authored.revision}&window_map=true`)).json() as StudioSavedTrainRuns;
  const selected = plan.window_map!.rows.filter(({ key }) => key.step === 'train' && sourceKeys.some(({ row }) => row === key.row));
  expect(selected.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.003]);
  expect(authored.project.steps?.[child]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: {
      axes: { 'learning-rate': selected.map(({ coordinates }) => coordinates['learning-rate'].value_id) },
    } } },
  });
  expect(plan.rows.filter(({ key }) => key.step === child).map(({ key }) => key.row)).toEqual(sourceKeys.map(({ row }) => row));

  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await expect(drawer.getByRole('combobox', { name: 'Edit Training method' })).toBeFocused();
  const batches = drawer.getByRole('textbox', { name: 'N Batches' });
  const childEdit = page.waitForResponse((response) => response.request().method() === 'PATCH' &&
    new URL(response.url()).pathname === projectPath);
  await number(page, batches, '1200');
  const changed = await childEdit;
  expect(changed.ok(), await changed.text()).toBe(true);
  const reviewed = await changed.json() as StudioProjectSnapshot;
  expect(reviewed.project.steps[child].deltas).toEqual(expect.arrayContaining([
    expect.objectContaining({ patches: expect.arrayContaining([
      expect.objectContaining({ path: 'training_config.n_batches', value: 1200 }),
    ]) }),
  ]));
  await page.keyboard.press('Meta+1');
  const table = runs.getByRole('table', { name: 'Runs' });
  const row = table.getByRole('row').filter({ has: page.getByRole('cell', { name: '0.001', exact: true }) }).getByRole('button');
  await activate(page, row);
  await expect(row).toHaveAttribute('aria-current', 'true');
  await expect(runs.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  const continuation = post(page, runsPath);
  await palette(page, 'run', 'Run');
  const admitted = await continuation;
  expect(admitted.status(), await admitted.text()).toBe(202);
  expect(admitted.request().postDataJSON().run_keys).toEqual([{ ...sourceKeys[0], step: child }]);
  const job = (await admitted.json() as StudioJobSubmission).items[0].job_id!;
  await expect(runs.getByRole('button', { name: 'Open Run', exact: true })).toBeEnabled();
  await palette(page, 'openRun', 'Open Run');
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  await expect(results).toBeFocused();
  await expect(results.getByRole('img', { name: 'Loss' }).locator('line title').filter({ hasText: job }).first()).toBeAttached({ timeout: 16_000 });
  const stopped = post(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stop = await stopped;
  expect(stop.status(), await stop.text()).toBe(202);
  expect(stop.request().postDataJSON()).toEqual({ job_ids: [job], mode: 'checkpoint' });
  await page.keyboard.press('Meta+1');
  await expect(table.getByText('Stopped', { exact: true })).toBeVisible({ timeout: 12_000 });
  await expect(table.getByText('Not run', { exact: true })).toBeVisible();
  await page.keyboard.press('Meta+7');
  await expect(results).toBeFocused();
  await expect(results.getByRole('img', { name: 'Loss' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(results.getByRole('img', { name: 'Loss' })).toBeHidden();
  await page.keyboard.press('Meta+BracketLeft');
  await expect(results.getByRole('img', { name: 'Loss' })).toBeVisible();

  const saved = await (await request.get(projectPath)).json() as StudioProjectDocument['snapshot'];
  const observed = await (await request.get(`${runsPath}?revision=${saved.revision}`)).json() as StudioSavedTrainRuns;
  const stoppedRow = observed.rows.find(({ key }) => key.step === child && key.row === sourceKeys[0].row);
  expect(stoppedRow?.attempts.at(-1)).toMatchObject({
    job_id: job, status: 'stopped', checkpoint_source: { job_id: producer },
  });
  console.info('Keyboard Train witness', JSON.stringify({
    project: saved.project_id, sourceKeys, continuation: stoppedRow?.key, producer, stoppedJob: job,
  }));
});
