import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, ObjectiveDocument, PythonTaskSelection, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../../src/generated/studioContracts';
import { mapStep } from './canonicalTrain';
import { activate, number, palette, post, readSaved } from './studioKeyboard';

export async function authorFirstTrain(
  page: Page, request: APIRequestContext, totalUpdates: number,
  forceComparison?: { targetY: number; activation: 'softmax' },
) {
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
  await expect(page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(model).toBeFocused();
  await page.keyboard.press('Meta+3');
  await setNumber(taskX, 0.22, (project) => taskValues(project)?.x);
  if (forceComparison) {
    await setNumber(task.getByRole('spinbutton', { name: 'Target y (m)', exact: true }),
      forceComparison.targetY, (project) => taskValues(project)?.y);
  }
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
  function linearField(project: StudioProjectSnapshot['project'], field: string) {
    const model = project.documents?.model?.payload as unknown as ModelDocument | undefined;
    const component = model?.interior.components?.[linearId!];
    return component?.kind === 'leaf' ? component.fields?.[field] : undefined;
  }
  function setLinearSize(name: string, field: string) {
    return setNumber(inspector.getByRole('textbox', { name, exact: true }), 2, (project) => linearField(project, field));
  }
  await setLinearSize('Input size', 'input_size');
  await setLinearSize('Output size', 'output_size');
  if (forceComparison) {
    const activation = inspector.getByRole('combobox', { name: 'Activation', exact: true });
    // Repeated initials cycle the options. The inspector saves and disables the field between keys.
    for (const value of ['sigmoid', forceComparison.activation]) {
      await activate(page, activation, 's');
      await expect(activation).toHaveValue(value);
      await expect.poll(async () => linearField((await savedProject()).project, 'activation')).toBe(value);
    }
  }
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
    method: 'feedbax.supervised_gradient',
    settings: { optimizer: { type: 'adam', params: { learning_rate: 0.001 } } },
    training_config: { n_batches: totalUpdates, batch_size: 4 },
  } });
  const savedModel = saved.project.documents?.model?.payload as unknown as ModelDocument;
  expect(savedModel.interior.components).toMatchObject({
    [linearId!]: { kind: 'leaf', type: 'feedbax.Linear', fields: {
      input_size: 2, output_size: 2, ...(forceComparison && { activation: forceComparison.activation }),
    } },
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
  expect(saved.project.documents?.task?.payload).toMatchObject({
    type: 'feedbax.Reaching',
    params: { x: 0.22, n_steps: 12, ...(forceComparison && { y: forceComparison.targetY }) },
  });
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
  return { saved, projectPath, savedProject, plan, plantId: plantId! };
}
