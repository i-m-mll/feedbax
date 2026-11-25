import { expect, firstJobTimeout, test } from './fixtures/studio';
import { type APIRequestContext, type Locator, type Page } from '@playwright/test';
import { firstTrainTask, type ModelDocument, type ModelEditResponse, type ModelFieldResponse, type StudioActivityJob, type StudioJobSubmission, type StudioProjectDocument, type StudioProjectSnapshot, type StudioSavedTrainKey, type StudioSavedTrainRuns, type StudioTrialReplayResponse } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, runName, runProjectFixture, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';
import { runPythonProcess } from './fixtures/python';
import { authorFirstTrain } from './fixtures/firstTrainAuthoring';
import { activate, activateMenuItem, chooseOutlineRow, chooseStepKind, number, palette, post, readSaved, tabTo } from './fixtures/studioKeyboard';
import { expectVisibleFocus, scanStudio } from './fixtures/studioAccessibility';
import { outlineRow } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const region = (page: Page, id: string) => page.locator(`[data-studio-focus-region="${id}"]:visible`);
const command = (page: Page, id: string) => page.locator(`button[data-command-id="${id}"]:visible`).first();

function alternateModelFiles(files: Awaited<ReturnType<typeof fixture>>, id: string) {
  const [model, task, objective, train] = structuredClone(files);
  const trial = train.step.base.inline.trial_documents;
  trial.model = { document: `${id}-model` };
  trial.task = { document: `${id}-task` };
  trial.objective = { document: `${id}-objective` };
  train.id = id;
  return [{ ...model, id: `${id}-model` },
    { ...task, id: `${id}-task` }, { ...objective, id: `${id}-objective` }, train];
}

test('an empty saved project starts its first Train by pointer', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  expect(snapshot.project.steps ?? {}).toEqual({});
  expect(snapshot.project.documents ?? {}).toEqual({});
  await page.goto(`/?project=${snapshot.project_id}`);

  const add = page.getByRole('button', { name: 'New Step', exact: true });
  await expect(add).toBeEnabled();
  await add.click();
  const chooser = page.getByRole('menu');
  const train = chooser.getByRole('menuitem', { name: 'Train', exact: true });
  await expect(chooser.getByRole('menuitem')).toHaveText(['Train', 'Evaluate', 'Analyze', 'Report']);
  await expect(chooser.getByRole('menuitem', { name: 'Evaluate', exact: true })).toHaveAttribute('aria-disabled', 'true');
  const replayPath = `${projectPath}/history/replay`;
  const creation = post(page, `${projectPath}/files`);
  await train.click();
  const created = await creation;
  expect(created.ok(), await created.text()).toBe(true);
  const { snapshot: saved } = await created.json() as StudioProjectDocument;
  expect(Object.keys(saved.project.steps ?? {})).toEqual(['train']);
  expect(saved.project.steps?.train).toMatchObject({
    kind: 'train', base: { inline: { trial_documents: {
      model: { document: 'model' }, task: { document: 'simple-reaches' },
      objective: { document: 'objective' },
    } } },
  });
  expect(saved.project.documents).toMatchObject({
    model: { schema: 'feedbax.spec.model', payload: { ports: {}, interior: {} } },
    'simple-reaches': { ...firstTrainTask, payload: { ...rootPayload(firstTrainTask) as object, links: [] } },
    objective: { schema: 'feedbax.spec.objective_document', payload: { terms: [] } },
  });

  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(chooser).toBeHidden();
  await expect(model).toBeFocused();
  await expect(model.locator('.react-flow')).toBeVisible();
  await expect(model.locator('[data-boundary="true"]')).toHaveCount(2);
  await expect(model.getByRole('button', { name: 'Model Document Model', exact: true })).toBeVisible();
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await expect(card.locator('[data-status-mark="cannot_run"]')).toBeVisible();

  const undo = post(page, replayPath);
  await page.keyboard.press('Meta+Z');
  const undone = await undo;
  expect(undone.ok(), await undone.text()).toBe(true);
  expect((await readSaved(request, projectPath)).file_revisions).toEqual(snapshot.file_revisions);
  await expect(mapStep(page, 'train')).toHaveCount(0);
  await expect(add).toBeEnabled();

  const redo = post(page, replayPath);
  await page.keyboard.press('Meta+Shift+Z');
  const redone = await redo;
  expect(redone.ok(), await redone.text()).toBe(true);
  expect((await readSaved(request, projectPath)).file_revisions).toEqual(saved.file_revisions);
  await mapStep(page, 'train').click();

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
  expect(plan.window_map?.problems).toMatchObject([
    { code: 'cannot_run', path: '/trial_documents/model/interior/components', place: { document: 'model' } },
  ]);
  const refused = await request.post(`${projectPath}/train-runs`, {
    data: { revision: saved.revision, run_keys: [run.key], token: crypto.randomUUID() },
  });
  expect(refused.status(), await refused.text()).toBe(202);
  expect((await refused.json() as StudioJobSubmission).items).toMatchObject([
    { disposition: 'rejected', reason: expect.stringMatching(/^cannot_run \/trial_documents\/model\/interior\/components: /) },
  ]);

  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  await tab('Runs').click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('textbox', { name: 'Batches', exact: true })).toBeEnabled();
  const current = await readSaved(request, projectPath);

  await page.reload();
  await expect(page).toHaveURL(new RegExp(`[?&]project=${snapshot.project_id}(?:&|$)`));
  await mapStep(page, 'train').click();
  await tab('Model').click();
  await expect(model.locator('.react-flow')).toBeVisible();
  await expect(model.getByRole('button', { name: 'Model Document Model', exact: true })).toBeVisible();
  await expect(card.locator('[data-status-mark="cannot_run"]')).toBeVisible();
  const persisted = await readSaved(request, projectPath);
  expect(persisted).toMatchObject({ project_id: snapshot.project_id, revision: current.revision, project: current.project });
  const reopened = await readPlan(persisted.revision);
  expect(reopened.rows).toMatchObject([{ key: run.key, status: 'cannot_run', attempts: [] }]);
  expect(reopened.window_map?.problems).toMatchObject([
    { code: 'cannot_run', path: '/trial_documents/model/interior/components', place: { document: 'model' } },
  ]);
});

test('authors, checkpoint-resumes and evaluates a subset from one fresh managed project', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(100_000);
  const totalUpdates = 2400;
  const home = studio.home;
  const { saved, projectPath, savedProject, plan } = await authorFirstTrain(page, request, totalUpdates);
  const runsPath = projectPath + '/train-runs';
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const train = page.getByRole('article', { name: 'Train train', exact: true });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activateMenuItem(page, page.getByRole('menuitemradio', { name: 'Sweep', exact: true }));
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true });
  await expect(rateText).toBeFocused();
  await number(page, rateText, '0.001, 0.002, 0.003');
  await expect(rate.getByText('0.003', { exact: true })).toBeVisible();
  const swept = await savedProject();
  const sourceRows = (await plan(swept.revision)).window_map!.rows.filter(({ key }) => key.step === 'train');
  expect(sourceRows.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.002, 0.003]);
  const [selected] = sourceRows;
  await page.keyboard.press('Control+Meta+2');
  await activate(page, sweepLabel(train, 'learning-rate'));
  const slow = page.getByRole('menuitem', { name: '0.001', exact: true });
  await activateMenuItem(page, slow);
  await expect(train.locator('.studio-map-run-mark[aria-pressed="true"]')).toHaveCount(1);

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
  const loss = results.getByRole('img', { name: 'Metric train_loss', exact: true });
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
  const witness = { module: 'web.e2e.studio.fixtures.managed_lifecycle' };
  const witnessArgs = (action: string, ...args: string[]) => [action, home, saved.project_id, initial.job_id!, ...args];
  const observe = async (action: string, ...args: string[]) =>
    JSON.parse(await runProjectFixture(witness.module, ...witnessArgs(action, ...args)));
  // Restart kills the Host and pings until a replacement answers, and that ping can start the
  // replacement, which must not become a child of the worker's shared fixture runner.
  const paused = JSON.parse(await runPythonProcess(witness, witnessArgs('restart')));
  expect(paused.checkpoint.completed_updates).toBeGreaterThan(0);
  expect(paused.checkpoint.completed_updates).toBeLessThan(totalUpdates);
  expect(paused.new_host).not.toBe(paused.old_host);

  await page.reload();
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.getByRole('button', { name: /^Activity( \d+)?$/ }).click();
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

  await page.getByRole('button', { name: /^Activity( \d+)?$/ }).click();
  await page.keyboard.press('Control+Meta+2');
  await activate(page, mapStep(page, 'train'));
  await activate(page, sweepLabel(train, 'learning-rate'));
  await activateMenuItem(page, slow);
  await expect(train.locator('.studio-map-run-mark[aria-pressed="true"]')).toHaveCount(1);
  const creating = post(page, projectPath + '/files');
  await activate(page, train.getByRole('button', { name: 'New Step train', exact: true }));
  await chooseStepKind(page, 'Evaluate', true);
  const createdEvaluate = await creating;
  expect(createdEvaluate.status(), await createdEvaluate.text()).toBe(201);
  const { snapshot: evaluated }: StudioProjectDocument = await createdEvaluate.json();
  expect(evaluated.project.steps!.evaluate.inputs?.subject).toEqual({
    step: 'train', checkpoint: 'last', where: { rule: `learning-rate=${selected.coordinates['learning-rate'].value}` },
  });
  const subject = page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('list', { name: 'Subject', exact: true });
  await expect(subject.getByRole('combobox', { name: 'Runs', exact: true })).toHaveValue('learning-rate=0.001');
  await expect(subject).toContainText('1/3');
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
  expect(result.trials).toHaveLength((rootPayload(firstTrainTask) as { validation_trials: unknown[] }).validation_trials.length);

  const replaying = post(page, projectPath + '/evaluate-replay');
  await page.reload();
  await activate(page, mapStep(page, 'evaluate'));
  await page.keyboard.press('Meta+1');
    await activate(page, runs.getByRole('table', { name: 'Runs', exact: true }).getByRole('button', { name: runName(pending.key.step) }));
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('button', { name: /^Select Made From / })).toContainText('1 of 3');
  expect(await evaluatedRow()).toEqual(result);
  await info.attach('managed-checkpoint-and-evaluate', {
    body: JSON.stringify({ project: saved.project_id, paused, continued, result }),
    contentType: 'application/json',
  });
  const sceneTab = page.getByRole('tab', { name: 'Scene', exact: true });
  await expect(sceneTab).toHaveAttribute('aria-keyshortcuts', 'Meta+6');
  await page.keyboard.press('Meta+6');
  const replayed = await replaying;
  await expect(sceneTab).toHaveAttribute('aria-selected', 'true');
  expect(replayed.ok(), await replayed.text()).toBe(true);
  expect(replayed.request().postDataJSON()).toMatchObject({ revision: evaluated.revision, key: result.key, job_id: evaluation.job_id, trial });
  const replay: StudioTrialReplayResponse = await replayed.json();
  const observed = await observe('result', continued.job_id, evaluated.revision, evaluation.job_id!, trial);
  expect(observed.receipt).toEqual(result.result!.receipt);
  expect(replay.receipt).toEqual(observed.receipt);
  expect(replay.drawn).toMatchObject({ kind: 'trials', trials: [{ id: trial }] });
  expect(replay.drawn).toMatchObject({ trials: [{ states: { [replay.position!]: { samples: observed.samples } } }] });
  expect(observed.samples.length).toBeGreaterThan(1);
  expect(observed.samples.every((sample: number[]) => sample.length === 2 && sample.every(Number.isFinite))).toBe(true);
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toBeVisible();
  await info.attach('authenticated-scene-replay', {
    body: JSON.stringify({ replay, observed }),
    contentType: 'application/json',
  });
});

test('Model edits follow the lit users of a saved shared document', async ({ page, request }) => {
  test.fixme(true, 'Owned by I1 (element selection and inspector): undo selects the edited field address, which the Inspector does not resolve to its owner.');
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
    const response = await request.post(`${projectPath}/model/fields`, { data: { revision, run, document: model.id,
      owner: { composites: [], element: { kind: 'component', id: 'plant' } },
    } });
    expect(response.ok(), await response.text()).toBe(true);
    const field = (await response.json() as ModelFieldResponse).fields?.find(({ label }) => label === 'Mass');
    return field?.effective ?? field?.text;
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'alpha')).toBeVisible();
  await page.keyboard.press('Meta+2');
  const modelView = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const count = modelView.locator('.studio-window-slot__header').getByRole('button', { name: /^Model Document / });
  const card = (step: string) => page.getByRole('article', { name: `Train ${step}`, exact: true });
  await expect(modelView).toBeFocused();
  await expect(count).not.toContainText(' of ');
  await expect(modelView.locator('header').first()).toContainText(model.id);
  for (const step of ['alpha', 'beta', 'gamma']) await expect(card(step)).toHaveAttribute('data-document-scope', 'lit');
  await expect(card('other')).toHaveAttribute('data-document-scope', 'faint');
  await expect(mapStep(page, 'alpha')).toHaveAttribute('aria-current', 'step');
  const initial = await readPlan(snapshot.revision);

  const component = modelView.getByRole('article', { name: 'Point mass', exact: true }).locator('header');
  await component.click();
  const inspector = region(page, 'inspector');
  const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
  const documentValue = (saved: StudioProjectSnapshot) => rootPayload(saved.project.documents?.[model.id]);
  const valueAt = (mass: number) => ({ interior: { components: { plant: { fields: { mass } } } } });

  async function savedModelWrite(writing: ReturnType<typeof post>) {
    const response = await writing;
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    expect(result.status).toBe('applied');
    if (result.status !== 'applied') throw new Error(JSON.stringify(result));
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

  await expect(count).not.toContainText(' of ');
  await activate(page, mapStep(page, 'gamma'), 'Shift+Enter');
  await activate(page, mapStep(page, 'alpha'), 'Shift+Enter');
  await expect(count).toContainText('2 of 3');
  await expect(card('gamma')).toHaveAttribute('data-document-scope', 'plain');
  await expect(card('alpha')).toHaveAttribute('data-document-scope', 'lit');
  await expect(mapStep(page, 'alpha')).toHaveAttribute('aria-current', 'step');

  await page.keyboard.press('Meta+1');
  await expect(page.getByRole('tabpanel', { name: 'Runs', exact: true })).toBeFocused();
  await expect(card('beta')).not.toHaveAttribute('data-document-scope');
  await page.keyboard.press('Meta+2');
  await expect(modelView).toBeFocused();
  await expect(count).toContainText('2 of 3');
  await expect(mass).toHaveValue('2');

  const rejected = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === `${projectPath}/model/fields` && !!response.request().postDataJSON()?.edit);
  await enter(mass, '3');
  const refusal = await (await rejected).json() as ModelFieldResponse;
  expect(refusal.problems?.[0].reason).toBe('Select all steps that use this Model to change its document value.');
  await expect(mass.locator('..').getByRole('img', { name: 'Problem', exact: true }))
    .toHaveAttribute('data-app-tip', refusal.problems![0].reason);
  expect((await readSaved(request, projectPath)).revision).toBe(shared.revision);
  for (const step of ['alpha', 'beta', 'gamma']) expect(await readMass(shared.revision, afterShared[step].key)).toBe('2');
  await mass.press('Escape');

  await activate(page, mapStep(page, 'gamma'), 'Shift+Enter');
  await expect(count).not.toContainText(' of ');
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
  await expect(mapStep(page, 'alpha')).toBeVisible();
  await page.keyboard.press('Meta+2');
  await expect(count).toHaveText('· 3 of 3');
  await component.click();
  await expect(mass).toHaveValue('2');
  const reopened = await readSaved(request, projectPath);
  expect(reopened.revision).toBe(beforeReload.revision);
  const plan = await readPlan(reopened.revision);
  for (const step of ['alpha', 'beta', 'gamma', 'other']) expect(plan[step].request_id).toEqual(afterShared[step].request_id);
});

test('the window commands, tab rail and selection history work without a pointer', async ({ page, request }, info) => {
  test.setTimeout(50_000);
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
  await expect(page.getByRole('table', { name: 'Runs', exact: true })).toBeVisible();
  await scanStudio(page, info, 'train-runs');
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
  await expectVisibleFocus(modelTab);
  await expect(modelTab).toHaveAttribute('tabindex', '0');
  await expect(tabs.getByRole('tab', { name: 'Runs', exact: true })).toHaveAttribute('tabindex', '-1');
  await expect(page.getByRole('tooltip')).toHaveText('Model · ⌘2');
  await expect(modelTab).not.toHaveAttribute('title', /.+/);
  await page.keyboard.press('Enter');
  await expect(modelTab).toHaveAttribute('aria-selected', 'true');
  const modelPanel = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(modelPanel).toBeFocused();
  await expect(modelTab).toHaveAttribute('aria-controls', (await modelPanel.getAttribute('id'))!);
  await expect(tabs.getByRole('tab', { name: 'Runs', exact: true })).not.toHaveAttribute('aria-controls');

  await page.keyboard.press('Meta+0');
  await expect(region(page, 'drawer')).toHaveCount(0);
  await palette(page, 'showDrawer', 'Show Drawer');
  await expect(region(page, 'drawer')).toBeFocused();
  await chooseOutlineRow(page, outlineRow(page, 'Point mass'), 'Point mass');
  await expect(region(page, 'inspector').getByRole('textbox', { name: 'Mass', exact: true })).toHaveValue('1');
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toHaveCount(0);
  await page.keyboard.press('Alt+Meta+0');
  await expect(region(page, 'inspector')).toBeFocused();
  await scanStudio(page, info, 'train-model-inspector');
  const trainTab = (name: string) => page.getByRole('tabpanel', { name, exact: true });
  await page.keyboard.press('Meta+3');
  await expect(trainTab('Task').getByRole('button', { name: /^Task Document / })).toBeVisible();
  await scanStudio(page, info, 'train-task');
  await page.keyboard.press('Meta+4');
  await expect(trainTab('Objective').getByRole('table')).toBeVisible();
  await scanStudio(page, info, 'train-objective');
  await page.keyboard.press('Meta+1');
  await expect(region(page, 'inspector').getByRole('textbox', { name: 'Batches', exact: true })).toBeVisible();
  await scanStudio(page, info, 'train-runs-inspector');
  await page.keyboard.press('Meta+2');
  await expect(trainTab('Model')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'inspector')).toBeFocused();
  await page.keyboard.press('F6');
  await expect(region(page, 'bottom')).toBeFocused();
  await page.keyboard.press('Shift+F6');
  await expect(region(page, 'inspector')).toBeFocused();

  await page.keyboard.press('Meta+Backslash');
  await expect(region(page, 'secondary')).toBeFocused();
  await expect(page.locator('[data-slot="secondary"]').getByRole('tabpanel', { name: 'Scene', exact: true })).toBeVisible();
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
  await expect(dialog).toHaveJSProperty('open', true);
  await command(page, 'activity').evaluate((button) => button.focus());
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await scanStudio(page, info, 'narrow-command-palette');
  await page.keyboard.press('Shift+Tab');
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(region(page, 'primary')).toBeFocused();

  await palette(page, 'activity', 'Activity');
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const closeActivity = activity.getByRole('button', { name: 'Close', exact: true });
  await expect(closeActivity).toBeFocused();
  await scanStudio(page, info, 'narrow-activity');
  await page.keyboard.press('Meta+K');
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await expect(dialog.locator('[data-command-id="run"], [data-command-id="stop"]')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(closeActivity).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(activity).toBeHidden();
});

test('the keyboard authors, runs and stops a strict checkpoint continuation and follows Results', { tag: '@round' }, async ({ page, request }) => {
  test.setTimeout(55_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const runsPath = `${projectPath}/train-runs`;
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('Meta+1');
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true });
  await activate(page, rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }));
  await activateMenuItem(page, page.getByRole('menuitemradio', { name: 'Sweep', exact: true }));
  await expect(rateText).toBeFocused();
  await number(page, rateText, '0.001, 0.002, 0.003');
  await expect(rate.getByText('0.003', { exact: true })).toBeVisible();
  const savedSweep = await request.get(projectPath);
  expect(savedSweep.ok()).toBe(true);
  const swept = await savedSweep.json() as StudioProjectSnapshot;
  expect(swept.revision).not.toBe(snapshot.revision);
  await page.reload();
  await expect(mapStep(page, 'train')).toBeVisible();
  await page.keyboard.press('Control+Meta+2');
  const value = (name: string) => page.getByRole('menuitem', { name, exact: true });
  await activate(page, sweepLabel(card, 'learning-rate'));
  await activateMenuItem(page, value('0.001'));
  await activate(page, sweepLabel(card, 'learning-rate'));
  await activateMenuItem(page, value('0.003'), 'Shift+Enter');
  await expect(card.locator('.studio-map-run-mark[aria-pressed="true"]')).toHaveCount(2);
  const marks = card.getByRole('grid', { name: 'Runs' });
  await tabTo(page, marks.getByRole('gridcell').first().getByRole('button'));
  await page.keyboard.press('ArrowRight');
  await expect(marks.getByRole('gridcell').nth(1).getByRole('button')).toBeFocused();

  const prefix = post(page, runsPath);
  await page.keyboard.press('Meta+R');
  const sourceRun = await prefix;
  expect(sourceRun.status(), await sourceRun.text()).toBe(202);
  const sourceKeys = sourceRun.request().postDataJSON().run_keys as StudioSavedTrainRuns['rows'][number]['key'][];
  expect(sourceKeys).toHaveLength(2);
  const producer = (await sourceRun.json() as StudioJobSubmission).items[0].job_id;
  await expect(card.locator('[data-status-mark="succeeded"]')).toHaveCount(2, { timeout: firstJobTimeout });
  await expect(card.locator('[data-status-mark="not_run"]')).toHaveCount(1);

  await expect(card.getByRole('button', { name: 'New Step train' })).toHaveAttribute('data-command-id', 'newStep');
  const create = post(page, `${projectPath}/files`);
  await palette(page, 'newStep', 'New Step');
  const train = page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true });
  await expect(train).toBeFocused();
  await page.keyboard.press('Enter');
  const created = await create;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: authored } = await created.json() as StudioProjectDocument;
  const child = 'train-2';
  const plan = await (await request.get(`${runsPath}?revision=${authored.revision}&window_map=true`)).json() as StudioSavedTrainRuns;
  const selected = plan.window_map!.rows.filter(({ key }) => key.step === 'train' && sourceKeys.some(({ row }) => row === key.row));
  expect(selected.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.003]);
  expect(authored.project.steps?.[child]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: { rule: 'learning-rate=0.001,0.003' } } },
  });
  expect(plan.rows.filter(({ key }) => key.step === child).map(({ key }) => key.row)).toEqual(sourceKeys.map(({ row }) => row));

  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('combobox', { name: 'Method', exact: true })).toBeFocused();
  const batches = inspector.getByRole('textbox', { name: 'Batches' });
  const childEdit = page.waitForResponse((response) => response.request().method() === 'PATCH' &&
    new URL(response.url()).pathname === projectPath);
  await number(page, batches, '1200');
  const changed = await childEdit;
  expect(changed.ok(), await changed.text()).toBe(true);
  const reviewed = await changed.json() as StudioProjectSnapshot;
  expect(reviewed.project.steps![child].deltas).toEqual(expect.arrayContaining([
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
  await expect(results.getByRole('img', { name: 'Metric train_loss', exact: true }).locator(`line[data-attempt="${job}"] title`).first()).toBeAttached({ timeout: 16_000 });
  const stopped = post(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stop = await stopped;
  expect(stop.status(), await stop.text()).toBe(202);
  expect(stop.request().postDataJSON()).toEqual({ job_ids: [job], mode: 'checkpoint' });
  await page.keyboard.press('Meta+1');
  await expect(table.getByText('Stopped', { exact: true })).toBeVisible({ timeout: 12_000 });
  await expect(table.getByRole('row').filter({ hasText: 'Stopped' }).getByText(/^[\d,]+ \/ [\d,]+$/)).toBeVisible();
  await expect(table.getByText('Not run', { exact: true })).toBeVisible();
  await page.keyboard.press('Meta+7');
  await expect(results).toBeFocused();
  await expect(results.getByRole('img', { name: 'Metric train_loss', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(results.getByRole('img', { name: 'Metric train_loss', exact: true })).toBeHidden();
  await page.keyboard.press('Meta+BracketLeft');
  await expect(results.getByRole('img', { name: 'Metric train_loss', exact: true })).toBeVisible();

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
