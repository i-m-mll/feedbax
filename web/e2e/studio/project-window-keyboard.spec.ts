import { expect, test, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, StudioDocumentAuthoring, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const region = (page: Page, id: string) => page.locator(`[data-studio-focus-region="${id}"]:visible`);
const command = (page: Page, id: string) => page.locator(`button[data-command-id="${id}"]:visible`).first();

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
  await page.keyboard.press('Meta+A');
  await page.keyboard.type(value);
  await expect(target).toHaveValue(value);
  await page.keyboard.press('Enter');
}

function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
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
  const edit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  await enter(reach, '0.6');
  const edited = await edit;
  expect(edited.ok(), await edited.text()).toBe(true);
  const current = await edited.json() as StudioProjectSnapshot;
  await tab('Model').click();

  await page.reload();
  await expect(model.locator('.react-flow')).toBeVisible();
  await expect(model.getByRole('button', { name: 'model', exact: true })).toBeVisible();
  await expect(card.getByRole('button', { name: "1 Can't run", exact: true })).toBeVisible();
  const persisted = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  expect(persisted).toMatchObject({ revision: current.revision, project: current.project });
  expect(persisted.project.documents?.task).toMatchObject({ payload: { params: { x: 0.6 } } });
  const reopened = await readPlan(persisted.revision);
  expect(reopened.rows).toMatchObject([{ key: run.key, status: 'cannot_run', attempts: [] }]);
  expect(reopened.window_map?.problems).toMatchObject([{ path: '/trial_documents/model/interior/components' }]);
});

test('the keyboard authors a managed Train from an empty folder and runs it after reopening', async ({ page, request }) => {
  test.setTimeout(58_000);
  const creation = post(page, '/api/projects');
  await page.goto('/');
  const created = await creation;
  expect(created.status()).toBe(201);
  const { snapshot } = await created.json() as StudioProjectDocument;
  expect(snapshot.project.steps).toEqual({});
  expect(snapshot.project.documents).toEqual({});
  const projectPath = '/api/projects/' + snapshot.project_id;
  const runsPath = projectPath + '/train-runs';
  const readSaved = async () => {
    const response = await request.get(projectPath + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  };
  const plan = async (revision: string) => {
    const response = await request.get(runsPath, { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const patch = () => page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  const setNumber = async (target: Locator, value: string) => {
    await expect(target).toBeEditable();
    const [saved] = await Promise.all([patch(), number(page, target, value)]);
    expect(saved.ok(), await saved.text()).toBe(true);
  };
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const objective = page.getByRole('tabpanel', { name: 'Objective', exact: true });
  const training = page.getByRole('tabpanel', { name: 'Training', exact: true });
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });

  await expect(page.getByRole('button', { name: 'New Step', exact: true })).toBeEnabled();
  await palette(page, 'newStep', 'New Step');
  const chooser = page.getByRole('dialog', { name: 'New Step', exact: true });
  await expect(chooser.getByRole('button', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(model).toBeFocused();
  await page.keyboard.press('Meta+3');
  await setNumber(task.getByRole('spinbutton', { name: 'Target x (m)', exact: true }), '0.22');
  await setNumber(task.getByRole('spinbutton', { name: /N Steps/i }), '12');

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
  async function modelField(name: string, field: string, value: string) {
    await number(page, inspector.getByRole('textbox', { name, exact: true }), value);
    await expect.poll(async () => {
      const model = (await readSaved()).project.documents?.model?.payload as unknown as ModelDocument;
      const linear = Object.values(model.interior.components ?? {}).find(({ name }) => name === 'Linear');
      return linear?.kind === 'leaf' ? linear.fields?.[field] : undefined;
    }).toBe(Number(value));
  }
  await modelField('Input size', 'input_size', '2');
  await modelField('Output size', 'output_size', '2');
  await page.keyboard.press('Meta+2');
  const plant = await add('PointMass', 'Mechanics');
  const output = model.locator('[data-id="@outputs"]');
  await page.keyboard.press('Meta+2');
  await expect(model).toBeFocused();
  await activate(page, output.getByRole('button', { name: 'Add Port', exact: true }));
  const portDialog = page.getByRole('dialog', { name: 'Add Port', exact: true });
  await number(page, portDialog.getByRole('textbox', { name: 'Name', exact: true }), 'Command');
  await expect(portDialog).toBeHidden();

  async function connect(source: Locator, target: Locator) {
    await expect(source).toHaveAttribute('aria-disabled', 'false');
    await page.keyboard.press('Meta+2');
    await expect(model).toBeFocused();
    await tabTo(page, source);
    const reviewed = post(page, projectPath + '/model/connections');
    await page.keyboard.press('Enter');
    expect((await reviewed).ok()).toBe(true);
    await expect(source).toHaveAttribute('aria-disabled', 'true');
    await expect(target).toHaveAttribute('aria-disabled', 'false');
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
  await connect(model.locator('[aria-label="Port reach_target"].react-flow__handle'), linear.locator('[data-handleid="input:input"]'));
  await connect(source, plant.locator('[data-handleid="input:force"]'));
  await connect(source, output.locator('[aria-label="Port Command"].react-flow__handle'));

  await page.keyboard.press('Meta+4');
  const measure = objective.getByRole('combobox', { name: 'Measure Source', exact: true });
  await expect(measure.locator('option:checked')).toHaveText('Command');
  await activate(page, objective.getByRole('button', { name: 'Add Term', exact: true }));
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(Command)');
  await setNumber(objective.getByRole('textbox', { name: 'Weight', exact: true }), '2');
  await page.keyboard.press('Meta+5');
  await setNumber(training.getByRole('textbox', { name: 'N Batches', exact: true }), '2');
  await setNumber(training.getByRole('textbox', { name: 'Batch Size', exact: true }), '4');

  const saved = await readSaved();
  const savedModel = saved.project.documents?.model?.payload as unknown as ModelDocument;
  expect(Object.values(savedModel.interior.wires ?? {})).toHaveLength(3);
  expect(saved.project.documents?.task?.payload).toMatchObject({ params: { x: 0.22, n_steps: 12 } });
  expect(saved.project.documents?.objective?.payload).toMatchObject({
    terms: [{ measure: { function: 'sqnorm' }, weight: { value: 2 } }],
  });
  const before = await plan(saved.revision);
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toMatchObject([{ status: 'not_run', attempts: [] }]);
  await page.reload();
  await expect(objective.or(training)).toBeVisible();
  await page.keyboard.press('Meta+4');
  await expect(objective.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('2');
  await page.keyboard.press('Meta+1');
  await expect(runs.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  const posted = post(page, runsPath);
  await page.keyboard.press('Meta+R');
  const admission = await posted;
  expect(admission.status(), await admission.text()).toBe(202);
  expect(admission.request().postDataJSON()).toMatchObject({ revision: saved.revision, run_keys: [before.rows[0].key] });
  const { items } = await admission.json() as StudioJobSubmission;
  expect(items).toMatchObject([{ disposition: 'admitted', job_id: expect.any(String) }]);
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });
  const [completed] = (await plan(saved.revision)).rows;
  expect(completed.request_id).toBe(before.rows[0].request_id);
  expect(completed.result).toMatchObject({
    job_id: items[0].job_id,
    progress: { completed_updates: 2 },
    receipt: { manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
  });
  console.info('Keyboard first managed Train', JSON.stringify({ revision: saved.revision, result: completed.result }));
});

test('Model edits follow the lit users of a saved shared document', async ({ page, request }) => {
  test.setTimeout(45_000);
  const [model, task, train] = await fixture('files');
  const alpha = { ...train, id: 'alpha' };
  const beta = { ...train, id: 'beta' };
  const gamma = { kind: 'step', id: 'gamma', step: { kind: 'train', base: { step: 'alpha' } } };
  const otherModel = { ...model, id: 'other-model' };
  const other = { ...train, id: 'other', step: { ...train.step, base: { inline: {
    ...train.step.base.inline, graph: { document: 'other-model' },
  } } } };
  const { snapshot, projectPath } = await seedManagedProject(request, [model, task, otherModel, alpha, beta, gamma, other]);
  const massPath = '/nodes/plant/params/mass';

  async function readPlan(revision: string) {
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    const plan = await response.json() as StudioSavedTrainRuns;
    expect(plan.rows.map(({ key }) => key.step).sort()).toEqual(['alpha', 'beta', 'gamma', 'other']);
    return Object.fromEntries(plan.rows.map((row) => [row.key.step, row]));
  }

  async function definition(revision: string, run: StudioSavedTrainRuns['rows'][number]['key']) {
    const response = await request.post(`${projectPath}/authoring/document`, {
      data: { revision, run, field: 'graph' },
    });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioDocumentAuthoring;
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'alpha')).toBeVisible();
  await page.keyboard.press('Meta+2');
  const modelView = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const count = modelView.locator('header').getByRole('status').filter({ hasText: /^\s*· \d+ of \d+\s*$/ });
  const card = (step: string) => page.getByRole('article', { name: `Train ${step}`, exact: true });
  await expect(modelView).toBeFocused();
  await expect(count).toHaveText('· 3 of 3');
  await expect(modelView.getByRole('combobox', { name: 'Edit Model' }).locator('option:checked')).toHaveText('model');
  for (const step of ['alpha', 'beta', 'gamma']) await expect(card(step)).toHaveAttribute('data-document-scope', 'lit');
  await expect(card('other')).toHaveAttribute('data-document-scope', 'faint');
  await expect(mapStep(page, 'alpha')).toHaveAttribute('aria-current', 'step');
  const initial = await readPlan(snapshot.revision);

  const component = page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('button', { name: 'plant', exact: true });
  await component.click();
  const mass = region(page, 'inspector').getByRole('spinbutton', { name: 'Edit Mass', exact: true });
  const write = () => page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  async function edit(value: string) {
    const writing = write();
    await enter(mass, value);
    const response = await writing;
    expect(response.ok(), await response.text()).toBe(true);
    return { written: await response.json() as StudioProjectSnapshot, body: response.request().postDataJSON() };
  }
  const documentValue = (saved: StudioProjectSnapshot) => saved.project.documents?.model.payload;
  const valueAt = (mass: number) => ({ nodes: { plant: { params: { mass } } } });
  const changes = (value: number) => ({ model: { [massPath]: value } });

  await expect(mass).toHaveValue('1');
  const shared = await edit('2');
  expect(shared.body.address).toEqual({ kind: 'document', id: 'model' });
  expect(documentValue(shared.written)).toMatchObject(valueAt(2));
  const afterShared = await readPlan(shared.written.revision);
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

  const narrowed = await edit('3');
  expect(narrowed.body.files.map(({ address }: { address: { kind: string; id: string } }) => address)).toEqual([
    { kind: 'step', id: 'alpha' }, { kind: 'step', id: 'beta' },
  ]);
  expect(documentValue(narrowed.written)).toMatchObject(valueAt(2));
  expect(narrowed.written.project.steps?.alpha.document_overrides).toEqual(changes(3));
  expect(narrowed.written.project.steps?.beta.document_overrides).toEqual(changes(3));
  const afterNarrowed = await readPlan(narrowed.written.revision);
  for (const step of ['alpha', 'beta']) expect(afterNarrowed[step].request_id).not.toEqual(afterShared[step].request_id);
  for (const step of ['gamma', 'other']) expect(afterNarrowed[step].request_id).toEqual(afterShared[step].request_id);
  const alphaLocal = await definition(narrowed.written.revision, afterNarrowed.alpha.key);
  const betaLocal = await definition(narrowed.written.revision, afterNarrowed.beta.key);
  const excluded = await definition(narrowed.written.revision, afterNarrowed.gamma.key);
  expect(alphaLocal.payload).toMatchObject(valueAt(3));
  expect(betaLocal.payload).toMatchObject(valueAt(3));
  expect(excluded.payload).toMatchObject(valueAt(2));
  expect(alphaLocal.sources[massPath]).toMatchObject({ kind: 'step', name: 'alpha' });
  expect(betaLocal.sources[massPath]).toMatchObject({ kind: 'step', name: 'beta' });
  expect(excluded.sources[massPath]).toMatchObject({ kind: 'document', name: 'model' });
  await expect(mass).toHaveValue('3');

  await expect(count).toHaveText('· 2 of 3');
  await activate(page, mapStep(page, 'gamma'), 'Shift+Enter');
  await expect(count).toHaveText('· 3 of 3');
  const restored = await edit('4');
  expect(restored.body.files).toHaveLength(3);
  expect(documentValue(restored.written)).toMatchObject(valueAt(4));
  expect(restored.written.project.steps?.alpha.document_overrides).toEqual({});
  expect(restored.written.project.steps?.beta.document_overrides).toEqual({});
  const afterRestore = await readPlan(restored.written.revision);
  for (const step of ['alpha', 'beta', 'gamma']) expect(afterRestore[step].request_id).not.toEqual(afterNarrowed[step].request_id);
  expect(afterRestore.other.request_id).toEqual(afterNarrowed.other.request_id);

  const undoing = write();
  await page.keyboard.press('Meta+Z');
  const undone = await undoing;
  expect(undone.ok(), await undone.text()).toBe(true);
  expect(undone.request().postDataJSON().files).toHaveLength(3);
  const beforeReload = await undone.json() as StudioProjectSnapshot;
  expect(documentValue(beforeReload)).toMatchObject(valueAt(2));
  expect(beforeReload.project.steps?.alpha.document_overrides).toEqual(changes(3));
  expect(beforeReload.project.steps?.beta.document_overrides).toEqual(changes(3));
  await expect(mass).toHaveValue('3');

  await page.reload();
  await expect(count).toHaveText('· 3 of 3');
  await component.click();
  await expect(mass).toHaveValue('3');
  const reopened = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  expect(reopened.revision).toBe(beforeReload.revision);
  const plan = await readPlan(reopened.revision);
  for (const step of ['alpha', 'beta', 'gamma', 'other']) expect(plan[step].request_id).toEqual(afterNarrowed[step].request_id);
});

test('the window commands, tab rail and selection history work without a pointer', async ({ page, request }) => {
  const [model, task, train, successor] = await fixture('files');
  const thirdModel = { ...model, id: 'third-model' };
  const third = { ...train, id: 'third', step: { ...train.step, base: { inline: {
    ...train.step.base.inline, graph: { document: thirdModel.id },
  } } } };
  const { snapshot } = await seedManagedProject(request, [model, task, train, successor, thirdModel, third]);
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
  await activate(page, region(page, 'drawer').getByRole('button', { name: 'plant', exact: true }));
  await expect(region(page, 'inspector').getByRole('spinbutton', { name: 'Edit Mass' })).toBeVisible();
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
  if (process.env.FEEDBAX_STUDIO_EVALUATION_CANDIDATE === '1') {
    const train = page.getByRole('dialog', { name: 'New Step', exact: true }).getByRole('button', { name: 'Train', exact: true });
    await expect(train).toBeFocused();
    await page.keyboard.press('Enter');
  }
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
