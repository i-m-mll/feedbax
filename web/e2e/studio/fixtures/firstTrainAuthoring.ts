import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import {
  firstTrainTask, type ModelDocument, type ObjectiveDocument, type StudioProjectSnapshot, type StudioSavedTrainRuns,
  type StudioTrialAuthoring, type TaskDocument,
} from '../../../src/generated/studioContracts';
import { mapStep } from './canonicalTrain';
import { pick } from './controls';
import { activate, activateMenuItem, number, palette, post, readSaved, startProject } from './studioKeyboard';
import { rootPayload } from './rootPayload';

/** The Task document a first Train creates: simple reaches, named by its label. */
const taskDocument = 'simple-reaches';

export async function authorFirstTrain(
  page: Page, request: APIRequestContext, totalUpdates: number,
  forceComparison?: { activation: 'softmax' },
) {
  const { snapshot } = await startProject(page);
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
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const weight = objective.getByRole('textbox', { name: 'Weight', exact: true });
  const nBatches = inspector.getByRole('textbox', { name: 'Batches', exact: true });
  const batchSize = inspector.getByRole('textbox', { name: 'Batch Size', exact: true });
  const terms = (project: StudioProjectSnapshot['project']) =>
    (rootPayload(project.documents?.objective) as unknown as ObjectiveDocument | undefined)?.terms ?? [];
  /** A new Train step reads its training config from the `training` document. */
  const config = (project: StudioProjectSnapshot['project']) => (
    rootPayload(project.documents?.training) as { training_config?: { n_batches?: number; batch_size?: number } } | undefined
  )?.training_config;

  await expect(page.getByRole('button', { name: 'New Step', exact: true })).toBeEnabled();
  await palette(page, 'newStep', 'New Step');
  await expect(page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(model).toBeFocused();
  await page.keyboard.press('Meta+3');
  await expect(task).toBeVisible();

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
    const model = rootPayload(project.documents?.model) as unknown as ModelDocument | undefined;
    const component = model?.interior.components?.[linearId!];
    return component?.kind === 'leaf' ? component.fields?.[field] : undefined;
  }
  await expect(inspector.getByRole('textbox', { name: 'Input size', exact: true })).toHaveCount(0);
  await setNumber(inspector.getByRole('textbox', { name: 'Output size', exact: true }), 2,
    (project) => linearField(project, 'output_size'));
  if (forceComparison) {
    const activation = inspector.getByRole('combobox', { name: 'Activation', exact: true });
    await pick(activation, new RegExp(`^${forceComparison.activation}$`, 'i'));
    await expect(activation).toHaveText(new RegExp(forceComparison.activation, 'i'));
    await expect.poll(async () => linearField((await savedProject()).project, 'activation')).toBe(forceComparison.activation);
  }
  await page.keyboard.press('Meta+2');
  const plant = await add('PointMass', 'Mechanics');
  const plantId = await plant.getAttribute('data-id');
  expect(plantId).toBeTruthy();
  const output = model.locator('[data-id="@outputs"]');

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
  await connect(model.locator('[data-handleid="output:@task:target_pos"]'), linear.locator('[data-handleid="input:input"]'));
  await connect(source, plant.locator('[data-handleid="input:force"]'));
  await connect(source, output.locator('[data-handleid="input:__new_port__"]'));

  await page.keyboard.press('Meta+4');
  await activate(page, objective.getByRole('button', { name: 'Add Term', exact: true }));
  await activateMenuItem(page, page.getByRole('menu', { name: 'Add Term', exact: true }).getByRole('menuitem', { name: 'Output', exact: true }));
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(output)');
  await setNumber(weight, 2, (project) => {
    const authored = terms(project)[0]?.weight;
    return authored && 'value' in authored ? authored.value : undefined;
  });
  await page.keyboard.press('Meta+1');
  await setNumber(nBatches, totalUpdates, (project) => config(project)?.n_batches);
  await expect(nBatches).toBeEditable();
  await setNumber(batchSize, 4, (project) => config(project)?.batch_size);

  // Simple reaches starts nothing until `start` is chosen to initialize this Model's point mass.
  const shipped = rootPayload(firstTrainTask) as unknown as TaskDocument;
  const plantLink = JSON.stringify({ composites: [], element: {
    kind: 'field', parent: { id: plantId, kind: 'component' }, path: ['state', 'effector/position'],
  } });
  const problems = async () => {
    const { revision } = await savedProject();
    const response = await request.post(projectPath + '/authoring/trials', {
      data: { revision, run: { variant: 'default', step: 'train', row: 'default' }, change: null },
    });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioTrialAuthoring).problems;
  };
  const links = async () => (rootPayload((await savedProject()).project.documents?.[taskDocument]) as unknown as TaskDocument).links;
  expect(await links()).toEqual([]);
  await page.keyboard.press('Meta+3');
  const initializes = task.getByRole('combobox', { name: 'Initializes start', exact: true });
  const toggle = async (component: string) => {
    await initializes.click();
    await page.keyboard.type('effector position');
    await page.getByRole('option', { name: `${component} · effector · position`, exact: true }).click();
    await page.keyboard.press('Escape');
  };
  await toggle('PointMass');
  await expect.poll(links).toEqual([{ model_field: plantLink, variable: 'start' }]);
  expect(await problems()).toEqual([]);
  await page.keyboard.press('Meta+1');
  // A comparison of forces reads one reach: simple reaches' others point every way, and their mean along the reach is
  // flat. The Task view does not edit validation trials yet.
  const [reach] = shipped.validation_trials!;
  if (forceComparison) {
    const narrowed = await request.patch(projectPath, { data: { revision: (await savedProject()).revision, files: [
      { address: { kind: 'document', id: taskDocument }, patches: [{ op: 'replace', path: '/payload/validation_trials', value: [reach] }] },
    ] } });
    expect(narrowed.ok(), await narrowed.text()).toBe(true);
  }

  const saved = await savedProject();
  expect(saved.project_id).toBe(snapshot.project_id);
  expect(saved.revision).not.toBe(snapshot.revision);
  expect(saved.project.steps?.train?.base).toMatchObject({ inline: {
    trial_documents: { model: { document: 'model' }, task: { document: taskDocument }, objective: { document: 'objective' } },
    method: { document: 'training', pointer: '/method' },
    settings: { document: 'training', pointer: '/settings' },
    training_config: { document: 'training', pointer: '/training_config' },
  } });
  expect(rootPayload(saved.project.documents?.training)).toMatchObject({
    method: 'feedbax.supervised_gradient',
    settings: { optimizer: { type: 'adam', learning_rate: { type: 'constant_schedule', value: 0.001 } } },
    training_config: { n_batches: totalUpdates, batch_size: 4 },
  });
  const savedModel = rootPayload(saved.project.documents?.model) as unknown as ModelDocument;
  expect(savedModel.interior.components).toMatchObject({
    [linearId!]: { kind: 'leaf', type: 'feedbax.Linear', fields: {
      output_size: 2, ...(forceComparison && { activation: forceComparison.activation }),
    } },
    [plantId!]: { kind: 'leaf', type: 'feedbax.PointMass' },
  });
  expect(linearField(saved.project, 'input_size')).toBeUndefined();
  const reachPort = savedModel.ports.inputs?.find(({ name }) => name === 'target_pos');
  const commandPort = savedModel.ports.outputs?.find(({ name }) => name === 'output');
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
  expect(rootPayload(saved.project.documents?.[taskDocument])).toEqual({
    ...shipped, links: [{ model_field: plantLink, variable: 'start' }],
    ...(forceComparison && { validation_trials: [reach] }),
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
    composites: [], element: { kind: 'port', end: { owner: 'outputs', port: commandPort!.id } },
  });
  const before = await plan(saved.revision);
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toMatchObject([{ key: { variant: 'default', step: 'train', row: 'default' },
    request_id: expect.any(String), status: 'not_run', attempts: [] }]);
  async function reopen() {
    await page.reload();
    await expect(page).toHaveURL(new RegExp(`[?&]project=${snapshot.project_id}(?:&|$)`));
    await activate(page, mapStep(page, 'train'));
    // The window comes back as it was left, on Runs.
    await expect(runs).toBeVisible();
  }
  await reopen();
  expect(await savedProject()).toMatchObject({ project_id: saved.project_id, revision: saved.revision, project: saved.project });
  expect((await plan(saved.revision)).rows).toEqual(before.rows);
  await page.keyboard.press('Meta+4');
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(output)');
  await expect(weight).toHaveValue('2');
  await page.keyboard.press('Meta+1');
  await expect(nBatches).toHaveValue(String(totalUpdates));
  await expect(batchSize).toHaveValue('4');
  return { saved, projectPath, savedProject, plan, plantId: plantId!, task: rootPayload(saved.project.documents?.[taskDocument]) as unknown as TaskDocument, reach };
}
