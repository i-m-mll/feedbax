import { expect, test } from './fixtures/studio';
import { type APIRequestContext, type Page, type TestInfo } from '@playwright/test';
import type {
  StudioEvaluationObjectiveSeries, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot, StudioSavedEvaluationRow,
  StudioSavedTrainRuns, StudioTrialReplayResponse, TrialRecords,
} from '../../src/generated/studioContracts';
import { savedRunId } from '../../src/features/projectFolder/projectWindowPlan';
import { studioBytes } from '../../src/components/activity/studioVocabulary';
import { authorEvaluateConditions } from './fixtures/authorEvaluateConditions';
import { captureStudioWindow, chooseSweepValue, drawnRunLabel, enter, mapStep, runName, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { expectStudioMapContrast, scanStudio } from './fixtures/studioAccessibility';
import { pick, pickRecordedTrial, reveal, trialPicker } from './fixtures/controls';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

const fixture = 'web.e2e.studio.fixtures.canonical_evaluate';
const trialIndex = 1;
const position = '/v1/model/state/plant/effector~1position';
const plans = (page: Page) => page.getByRole('table', { name: 'Runs', exact: true });
const footer = (page: Page) => page.locator('[data-studio-focus-region="bottom"]');

function tab(page: Page, name: string) {
  return page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });
}

function posted(page: Page, path: string, trial?: string) {
  return page.waitForResponse((response) => {
    const request = response.request();
    return request.method() === 'POST' && new URL(response.url()).pathname === path
      && (!trial || request.postDataJSON()?.trial === trial);
  });
}

async function submit(page: Page, path: string) {
  const response = posted(page, path);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await response;
  expect(accepted.status(), await accepted.text()).toBe(202);
  return (await accepted.json() as StudioJobSubmission).items;
}

async function chooseStep(page: Page, step: string) {
  await mapStep(page, step).click();
  await tab(page, 'Runs').click();
}

test('authors Evaluate conditions from one authenticated Train checkpoint', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(60_000);
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.evaluate_authoring', studio.home, 'authored-evaluate-conditions',
  ));
  const projectPath = '/api/projects/' + project;
  const runsPath = projectPath + '/train-runs';
  const snapshot = async () => {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  };
  const plan = async (revision: string) => {
    const response = await request.get(runsPath, { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const variants = page.getByRole('combobox', { name: 'Variant', exact: true });
  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded', timeout: 10_000 });
  await expect(variants).toHaveText('base');
  const initial = await snapshot();
  expect(Object.keys(initial.project.steps!)).toEqual(['train']);
  await chooseStep(page, 'train');
  const trainItems = await submit(page, runsPath);
  expect(trainItems).toHaveLength(1);
  const [admitted] = trainItems;
  expect(admitted.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 20_000 });
  const trained = (await plan(initial.revision)).rows.find(({ key }) => key.variant === 'base')!;
  expect(admitted.run_key).toBe(savedRunId(trained.key));
  expect(trained.result?.job_id).toBe(admitted.job_id);
  // A step's only run is named by its step.
  await plans(page).getByRole('button', { name: runName(trained.key.step) }).click();

  const conditions = [1.25, 2];
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const { stepId, step, revision, rows } = await authorEvaluateConditions({
    page, request, projectPath, train: trained.key, field: 'Mass', conditions,
    selectComponent: () => model.locator('[data-id="plant"] header').click(),
  });
  expect(step.kind).toBe('evaluate');
  expect(step.inputs?.subject).toMatchObject({ step: trained.key.step, checkpoint: 'last' });
  const axis = Object.values(step.sweep?.axes ?? {}).find((axis) => axis && 'values' in axis);
  expect(axis).toBeDefined();
  expect(rows.map(({ coordinates }) => Object.values(coordinates).find(({ value }) =>
    conditions.includes(value as number))?.value)).toEqual(conditions);
  const saved = await snapshot();
  expect(saved.project.documents).toEqual(initial.project.documents);
  expect(saved.project.steps![trained.key.step]).toEqual(initial.project.steps![trained.key.step]);
  expect(saved.project.sweep).toEqual(initial.project.sweep);

  await tab(page, 'Runs').click();
  await expect(page.getByRole('combobox', { name: 'Trials', exact: true })).toBeVisible();
  const items = await submit(page, runsPath);
  expect(items.map(({ disposition, run_key }) => ({ disposition, run_key }))).toEqual(
    rows.map(({ key }) => ({ disposition: 'admitted', run_key: savedRunId(key) })),
  );
  await expect(plans(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 24_000 });
  const complete = await plan(revision);
  const evaluations = complete.window_map!.evaluations!.filter(({ key }) => key.step === stepId);
  const completed = evaluations.filter(({ key }) => key.variant === 'base');
  expect(completed.map(({ key }) => key)).toEqual(rows.map(({ key }) => key));
  expect(completed.map(({ result }) => result?.job_id)).toEqual(items.map(({ job_id }) => job_id));
  expect(new Set(completed.map(({ request_id }) => request_id)).size).toBe(2);
  expect(new Set(completed.map(({ result }) => result?.receipt.manifest_sha256)).size).toBe(2);
  expect(completed[0].trials.length).toBeGreaterThan(0);
  expect(completed[0].trials).toEqual(completed[1].trials);
  for (const { result } of completed) {
    expect(result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result?.checkpoint_source).toMatchObject({
      job_id: trained.result!.job_id, request_id: trained.request_id,
      completed_updates: trained.result!.progress!.completed_updates,
    });
  }
  expect(completed[0].result?.checkpoint_source).toEqual(completed[1].result?.checkpoint_source);
  const comparison = evaluations.filter(({ key }) => key.variant === 'compare');
  expect(comparison.map(({ result }) => result?.receipt)).toEqual(completed.map(({ result }) => result?.receipt));
  expect(complete.rows.find(({ key }) => key.variant === 'base')).toEqual(trained);

  await page.reload({ waitUntil: 'domcontentloaded', timeout: 10_000 });
  await expect(variants).toHaveText('base');
  await chooseStep(page, stepId);
  await expect(plans(page).getByText('Done', { exact: true })).toHaveCount(2);
  expect((await snapshot()).project.steps![stepId]).toEqual(step);
  expect((await plan(revision)).window_map!.evaluations!.filter(({ key }) => key.step === stepId)).toEqual(evaluations);
  await info.attach('authored-evaluate-conditions', {
    contentType: 'application/json',
    body: JSON.stringify({ stepId, step, revision, rows, evaluations, train: trained.result }),
  });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  for (const [index, condition] of conditions.entries()) {
    await tab(page, 'Runs').click();
    await plans(page).getByRole('button', { name: runName(rows[index].label) }).click();
    await tab(page, 'Model').click();
    await model.locator('[data-id="plant"] header').click();
    const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
    await expect(mass).toHaveValue(String(condition));
    await expect(mass).toHaveAccessibleDescription('Changed by ' + stepId);
    await expect(mass).not.toBeEditable();
  }
});

function expectPositionSamples(samples: number[][]) {
  expect(samples).toHaveLength(81);
  expect(samples.every((sample) => sample.length === 2 && sample.every(Number.isFinite))).toBe(true);
}

function replayPosition(replay: StudioTrialReplayResponse) {
  if (replay.drawn.kind !== 'trials' || replay.position === null) throw new Error('Expected recorded position');
  const track = replay.drawn.trials[0]?.states?.[replay.position];
  if (!track) throw new Error('Expected recorded position samples');
  return track;
}

async function showTrial(page: Page, path: string, row: StudioSavedEvaluationRow, label: string, checkpoint?: number) {
  const trialId = row.trials[trialIndex];
  // The picked trial is the window's: once picked for one run, the Scene opens the next run on the same trial, so its
  // replay can be requested as soon as the Scene shows that run.
  const response = page.waitForResponse((response) => {
    const request = response.request();
    return request.method() === 'POST' && new URL(response.url()).pathname === path &&
      request.postDataJSON()?.trial === trialId && request.postDataJSON()?.job_id === row.result!.job_id;
  });
  await chooseStep(page, row.key.step);
  await plans(page).getByRole('button', { name: runName(label) }).click();
  await tab(page, 'Scene').click();
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  await pickRecordedTrial(page, trialIndex);
  const reply = await response;
  expect(reply.ok(), await reply.text()).toBe(true);
  expect(reply.request().postDataJSON()).toMatchObject({
    key: row.key, job_id: row.result!.job_id, trial: trialId, ...(checkpoint === undefined ? {} : { checkpoint }),
  });
  const replay = await reply.json() as StudioTrialReplayResponse;
  expect(replay.drawn).toMatchObject({ kind: 'trials', trials: [{ id: trialId }] });
  expect(replay.receipt).toEqual(row.result!.receipt);
  expect(replay.state_time.length).toBe(81);
  expect(replay.state_time[0]).toBe(0);
  const cursor = scene.locator('[data-scene-cursor]');
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  const scrubber = clock.getByRole('slider', { name: 'Playback Position' });
  await expect(scrubber).toHaveValue('0');
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toHaveCount(1);
  const coordinates = () => cursor.evaluate((point) => [point.getAttribute('cx'), point.getAttribute('cy')]);
  const reset = await coordinates();
  await scrubber.press('End');
  await expect.poll(coordinates).not.toEqual(reset);
  await expect(scrubber).toHaveValue(String(replay.state_time.at(-1)));
  // Catches a recorded trial whose shelf draws no epochs, or epochs other than those the run recorded.
  const [record] = (replay.drawn as TrialRecords).trials;
  const period = replay.state_time[1];
  const recorded = record.timeline.filter(({ start, end }) => end > start);
  expect(recorded.length).toBeGreaterThan(0);
  const shelfSwitch = clock.getByRole('button', { name: 'Shelf', exact: true });
  await shelfSwitch.click();
  const segments = page.getByRole('region', { name: 'Shelf', exact: true })
    .getByRole('group', { name: 'Epochs', exact: true }).locator('[data-shelf-mark]');
  await expect(segments).toHaveCount(recorded.length);
  for (const [index, { label, start, end }] of recorded.entries()) {
    const segment = segments.nth(index);
    await expect(segment).toHaveAccessibleName(label);
    expect(Number(await segment.getAttribute('data-start'))).toBeCloseTo(start * period, 9);
    expect(Number(await segment.getAttribute('data-end'))).toBeCloseTo(end * period, 9);
  }
  await shelfSwitch.click();
  return replay;
}

async function runSavedEvaluation(page: Page, request: APIRequestContext, info: TestInfo, home: string) {
  const projectId = JSON.parse(await runProjectFixture(fixture, 'seed', home, 'saved-evaluate'));
  const projectPath = `/api/projects/${projectId}`;
  const opened = await request.get(projectPath);
  expect(opened.ok(), await opened.text()).toBe(true);
  const snapshot = await opened.json() as StudioProjectSnapshot;
  const path = `${projectPath}/train-runs`;
  const replayPath = `${projectPath}/evaluate-replay`;
  const revision = snapshot.revision;
  async function plan() {
    const response = await request.get(`${path}?revision=${revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await chooseStep(page, 'baseline');
  await expect(plans(page).getByText('Waiting', { exact: true })).toBeVisible();
  const initialPlan = (await plan()).window_map!;
  const [initial] = initialPlan.evaluations!.filter(({ key }) => key.step === 'baseline');
  expect(initial).toMatchObject({ can_run: false, status: 'waiting', reason: 'Waiting for train.' });
  expect(initialPlan.problems.filter(({ step }) => step === 'baseline')).toEqual([]);
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeDisabled();
  // A card accepts a new step whatever its runs' status.
  await expect(page.getByRole('button', { name: 'New Step baseline', exact: true })).toBeEnabled();

  await chooseStep(page, 'unsupported');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  // The inspector gives a run's reasons once the run is chosen.
  await plans(page).getByRole('button', { name: runName('unsupported') }).click();
  await expect(inspector.getByText(/requires a Train checkpoint selected by last/)).toBeVisible();
  await chooseStep(page, 'train');
  const [train] = await submit(page, path);
  expect(train.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });

  await chooseStep(page, 'baseline');
  const baseline = await submit(page, path);
  await chooseStep(page, 'conditions');
  const conditions = await submit(page, path);
  await expect(plans(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 22_000 });
  const baselineCard = page.getByRole('article', { name: 'Evaluate baseline', exact: true });
  await expect(baselineCard.getByRole('img', { name: 'Done', exact: true })).toBeVisible({ timeout: 22_000 });
  await expect(baselineCard).toContainText('1');
  await expect(plans(page).getByRole('columnheader', { name: 'Size', exact: true })).toBeVisible();
  const settled = await plan();
  const conditionSizes = settled.window_map!.evaluations!
    .filter(({ key }) => key.step === 'conditions').map(({ result }) => studioBytes(result!.size_bytes));
  expect(conditionSizes).toHaveLength(2);
  for (const size of new Set(conditionSizes)) {
    await expect(plans(page).getByText(size, { exact: true })).toHaveCount(conditionSizes.filter((other) => other === size).length);
  }
  await chooseStep(page, 'baseline');
  const baselineSize = settled.window_map!.evaluations!.find(({ key }) => key.step === 'baseline')!.result!.size_bytes;
  await expect(plans(page).getByText(studioBytes(baselineSize), { exact: true })).toHaveCount(1);
  await chooseStep(page, 'finite');
  const [finite] = await submit(page, path);
  const submitted = [...baseline, ...conditions, finite];
  expect(submitted.map(({ disposition }) => disposition)).toEqual(['admitted', 'admitted', 'admitted', 'admitted']);
  const finiteStatus = plans(page).getByRole('cell').filter({ hasText: /^(Done|Blocked|Failed)/ });
  await expect(finiteStatus).toBeVisible({ timeout: 22_000 });
  expect(await finiteStatus.innerText()).toBe('Done');
  const completed = await plan();
  const rows = completed.window_map!.evaluations!.filter(({ result }) => result);
  expect(rows).toHaveLength(4);
  expect(new Set(rows.map(({ request_id }) => request_id)).size).toBe(4);
  expect(new Set(rows.map(({ result }) => result!.job_id))).toEqual(new Set(submitted.map(({ job_id }) => job_id)));
  expect(new Set(rows.map(({ result }) => result!.receipt.manifest_sha256)).size).toBe(4);
  const source = settled.rows.find(({ key }) => key.step === 'train')!;
  const finiteRow = rows.find(({ key }) => key.step === 'finite')!;
  expect(finiteRow.result).toMatchObject({
    checkpoint_source: null,
    checkpoint_sources: [2, 1].map((completed_updates) => ({
      job_id: train.job_id, request_id: source.request_id, completed_updates,
    })),
  });

  const single = rows.filter(({ key }) => key.step !== 'finite');
  for (const { result } of single) {
    expect(result!.checkpoint_source).toEqual({ job_id: train.job_id, request_id: source.request_id, completed_updates: 2 });
    expect(result!.size_bytes).toBeGreaterThan(0);
  }
  const base = rows.find(({ key }) => key.step === 'baseline')!;
  const invalidTrial = await request.post(replayPath, {
    data: { revision, key: base.key, job_id: base.result!.job_id, trial: 'absent' },
  });
  expect(invalidTrial.status()).toBe(422);
  const wrongJob = await request.post(replayPath, {
    data: { revision, key: base.key, job_id: train.job_id, trial: base.trials[trialIndex] },
  });
  expect(wrongJob.status()).toBe(404);

  await page.reload();
  const reopened = (await plan()).window_map!.evaluations!.filter(({ result }) => result);
  expect(reopened).toEqual(rows);

  const analysis = JSON.parse(await runProjectFixture(
    fixture, 'observe', home, snapshot.project_id, revision, String(trialIndex),
  ));
  expect(Object.keys(analysis)).toHaveLength(4);
  for (const { key, request_id, result } of single) {
    const observed = analysis[result!.job_id];
    expect(observed).toMatchObject({
      key: [key.variant, key.step, key.row], request_id, receipt: result!.receipt,
      checkpoint_source: result!.checkpoint_source, size_bytes: result!.size_bytes,
      trial_index: trialIndex, tensor_reads: [position],
    });
    expectPositionSamples(observed.samples);
  }
  const finiteResult = finiteRow.result!;
  const fromCheckpoint = analysis[finiteResult.job_id];
  expect(fromCheckpoint).toMatchObject({
    key: [finiteRow.key.variant, finiteRow.key.step, finiteRow.key.row],
    request_id: finiteRow.request_id, receipt: finiteResult.receipt,
    checkpoint_sources: finiteResult.checkpoint_sources, size_bytes: finiteResult.size_bytes, trial_index: trialIndex,
    tensor_reads: [position, position],
    checkpoint_coordinate_reads: 2,
  });
  expect(fromCheckpoint.checkpoints.map(({ coordinate }: { coordinate: number }) => coordinate)).toEqual([2, 1]);
  for (const { samples } of fromCheckpoint.checkpoints) {
    expectPositionSamples(samples);
  }

  await info.attach('saved-evaluation-lineage', {
    body: JSON.stringify({ project: snapshot.project_id, revision, train: source.result, rows: reopened, analysis }),
    contentType: 'application/json',
  });
  return { revision, replayPath, completed, single, finiteRow, fromCheckpoint, analysis };
}

test('authors selected saved Evaluate Model and Task inputs and repairs a refused edit', { tag: '@round' }, async ({ page, request }) => {
  test.setTimeout(59_000);
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const path = `${projectPath}/train-runs`;
  const saved = async () => {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  };
  const plan = async () => {
    let view: StudioSavedTrainRuns | undefined;
    await expect.poll(async () => {
      const { revision } = await saved();
      const response = await request.get(path, { params: { revision, window_map: true } });
      const body = await response.text();
      if (response.status() === 409) {
        const { detail } = JSON.parse(body);
        if (detail?.resource === 'project' && /^[a-f0-9]{64}$/.test(detail.current) && detail.current !== revision) {
          return false;
        }
      }
      expect(response.ok(), body).toBe(true);
      view = JSON.parse(body) as StudioSavedTrainRuns;
      expect(view.revision).toBe(revision);
      return true;
    }, { message: 'Read the Plan at the saved project revision' }).toBe(true);
    return view!;
  };
  const selectedKey = { variant: 'default', step: 'conditions', row: 'seed=control' };
  const siblingKey = { ...selectedKey, row: 'seed=comparison' };
  const unselectedKey = { ...selectedKey, row: 'seed=unselected' };
  const row = (view: StudioSavedTrainRuns, key: typeof selectedKey) => view.window_map?.evaluations?.find(
    (item) => item.key.variant === key.variant && item.key.step === key.step && item.key.row === key.row,
  );
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
  const stepField = inspector.getByRole('textbox', { name: 'Step', exact: true });
  const chooseRow = async (run = selectedKey, extend = false) => {
    const definition = (await plan()).window_map?.rows.find(({ key }) => key.step === run.step && key.row === run.row);
    expect(definition).toBeDefined();
    await plans(page).getByRole('button', { name: runName(definition!.label) })
      .click(extend ? { modifiers: ['Shift'] } : {});
  };
  await page.goto(`/?project=${snapshot.project_id}`);
  await chooseStep(page, 'train');
  const [train] = await submit(page, path);
  expect(train.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });
  const before = await plan();
  const [trained] = before.rows;
  const sibling = row(before, siblingKey);
  const unselected = row(before, unselectedKey);
  const graphPath = '/params/variation/raw-mass/path';
  const authoredPath = '/sweep/exceptions/raw-control/patches/params.variation/value/raw-mass/path';
  const invalidGraph = (view: StudioSavedTrainRuns) => view.window_map?.problems.find(({ code, step, row }) =>
    code === 'authored_graph_variation' && step === selectedKey.step && row === selectedKey.row);
  const storedControl = (state: StudioProjectSnapshot) =>
    state.project.steps?.conditions?.sweep?.exceptions?.['raw-control']?.patches?.['params.variation'];
  const problem = invalidGraph(before);
  expect(problem).toMatchObject({ path: graphPath, address: 'steps/conditions.json', authored_path: authoredPath });
  expect(row(before, selectedKey)).toBeUndefined();
  expect(sibling?.can_run).toBe(true);
  await chooseStep(page, 'conditions');
  await chooseRow();
  await expect(plans(page).getByText("Can't run", { exact: true })).toBeVisible();
  await expect(inspector).toContainText(`Model · ${problem!.detail}`);
  const revealVariation = async () => {
    await page.getByRole('button', { name: /^Problems\b/ }).click();
    await page.getByRole('dialog', { name: 'Problems' }).getByRole('button')
      .filter({ hasText: problem!.detail }).click();
  };
  await revealVariation();
  await expect(model).toBeVisible();
  const remove = model.getByRole('button', { name: 'Remove Variation', exact: true });
  await expect(remove).toBeEnabled();
  const saveEdit = async (gesture: () => Promise<void>) => {
    const response = page.waitForResponse((reply) =>
      reply.request().method() === 'PATCH' && new URL(reply.url()).pathname === projectPath);
    await gesture();
    const savedEdit = await response;
    expect(savedEdit.ok(), await savedEdit.text()).toBe(true);
  };
  await saveEdit(() => remove.click());
  await expect.poll(async () => row(await plan(), selectedKey)?.can_run).toBe(true);
  const repaired = await plan();
  const original = row(repaired, selectedKey);
  expect(invalidGraph(repaired)).toBeUndefined();
  expect(repaired.rows).toEqual(before.rows);
  expect(row(repaired, siblingKey)).toEqual(sibling);
  expect(row(repaired, unselectedKey)).toEqual(unselected);
  expect(storedControl(await saved())).toEqual({
    op: 'replace', value: { 'row-y': { op: 'replace', path: 'task.variables.target.value.args.1.value', value: 0.04 } },
  });
  await model.focus();
  await saveEdit(() => page.keyboard.press('Meta+Z'));
  await expect.poll(async () => invalidGraph(await plan())?.path).toBe(graphPath);
  expect(storedControl(await saved())).toEqual(storedControl(snapshot));
  await model.focus();
  await saveEdit(() => page.keyboard.press('Meta+Shift+Z'));
  await expect.poll(async () => row(await plan(), selectedKey)?.request_id).toBe(original?.request_id);
  await expect(remove).toHaveCount(0);

  await model.locator('[data-id="plant"] header').click();
  await expect(mass).toHaveAccessibleDescription('Trained by train');
  await expect(mass).toHaveValue('1.25');
  await enter(mass, '2');
  await expect(mass).toHaveAccessibleDescription('Changed by conditions');
  await expect(mass).toHaveValue('2');
  const changed = await plan();
  expect(changed.rows).toEqual(before.rows);
  expect(row(changed, siblingKey)).toEqual(sibling);
  expect(row(changed, unselectedKey)).toEqual(unselected);
  expect(row(changed, selectedKey)?.request_id).not.toBe(original?.request_id);
  const selectedProblem = async () => (await plan()).window_map?.problems.find(({ step, row }) =>
    step === selectedKey.step && row === selectedKey.row);
  await tab(page, 'Model').click();
  await model.locator('[data-id="plant"] header').click();
  await expect(mass).toHaveValue('2');
  await model.locator('.react-flow__pane').click({ position: { x: 10, y: 10 } });
  await expect(stepField).toBeEditable();
  const clock = await stepField.inputValue();
  await enter(stepField, '0.03 s');
  await expect(stepField).toBeEditable();
  await expect(stepField).toHaveValue('0.03 s');
  await expect.poll(selectedProblem).toMatchObject({
    path: '/step_seconds', detail: 'This Model field changes the trained graph interface or clock',
    place: { part: 'model', document: 'point-mass', member: { element: { kind: 'field', path: ['step_seconds'] } } },
  });
  await tab(page, 'Runs').click();
  await expect(plans(page).getByText("Can't run", { exact: true })).toBeVisible();
  await expect(inspector).toContainText('Model · This Model field changes the trained graph interface or clock');
  await tab(page, 'Model').click();
  await expect(stepField).toBeEditable();
  await enter(stepField, clock);
  await expect.poll(async () => row(await plan(), selectedKey)?.request_id).toBe(row(changed, selectedKey)?.request_id);
  await tab(page, 'Runs').click();
  await chooseRow();
  const [evaluation] = await submit(page, path);
  expect(evaluation.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  const complete = await plan();
  expect(complete.rows).toEqual(before.rows);
  expect(row(complete, siblingKey)).toEqual(row(changed, siblingKey));
  expect(row(complete, unselectedKey)).toEqual(unselected);
  expect(row(complete, selectedKey)?.result).toMatchObject({
    job_id: evaluation.job_id, checkpoint_source: { job_id: train.job_id, request_id: trained.request_id, completed_updates: 2 },
    receipt: { manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
  });
});

test('authors checkpoint rules and replays covered updates from Results', { tag: '@round' }, async ({ page, request }, info) => {
  test.setTimeout(60_000);
  const files = JSON.parse(await runProjectFixture(fixture, 'series-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const runsPath = projectPath + '/train-runs';
  async function plan(revision: string) {
    const response = await request.get(runsPath, { params: { revision, window_map: true } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await chooseStep(page, 'train');
  const [train] = await submit(page, runsPath);
  expect(train.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 19_000 });
  const trainRequest = (await plan(snapshot.revision)).rows.find(({ key }) => key.step === 'train')!.request_id;
  const creating = posted(page, projectPath + '/files');
  await page.getByRole('article', { name: 'Train train', exact: true })
    .getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Evaluate');
  const created = await creating;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: initial } = await created.json() as StudioProjectDocument;
  const checkpoint = page.getByRole('region', { name: 'Checkpoint', exact: true });
  const kind = checkpoint.getByRole('combobox', { name: 'Checkpoint', exact: true });
  const every = checkpoint.getByRole('textbox', { name: 'Every', exact: true });
  const updates = checkpoint.getByRole('textbox', { name: 'Updates', exact: true });
  await expect(kind).toHaveText('Last');
  async function saved(action: () => Promise<unknown>) {
    const saving = page.waitForResponse((response) =>
      response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  await pick(kind, 'Every');
  await expect(every).toBeFocused();
  const cadence = await saved(async () => {
    await every.fill('2');
    await every.press('Enter');
  });
  expect(cadence.project.steps!.evaluate.inputs?.subject).toMatchObject({ step: 'train', checkpoint: { every: 2 } });
  const cadencePlan = await plan(cadence.revision);
  const [planned] = cadencePlan.window_map!.evaluations!;
  expect(planned.checkpoint_series?.requested).toEqual([2, 4]);
  await page.reload();
  await chooseStep(page, 'evaluate');
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })).toBeVisible();
  await expect(kind).toHaveText('Every');
  await expect(every).toHaveValue('2');
  await pick(kind, 'List');
  const held = await saved(async () => {
    await updates.fill('2, 2');
    await updates.press('Enter');
  });
  expect(held.project.steps!.evaluate.inputs?.subject).toMatchObject({ checkpoint: { kind: 'unresolved', text: '2, 2' } });
  await expect(updates).toHaveAttribute('aria-invalid', 'true');
  await expect(updates).toHaveAccessibleDescription(/\S/);
  await expect(checkpoint.getByRole('img', { name: 'Problem', exact: true })).toBeVisible();
  const explicit = await saved(async () => {
    await updates.fill('4, 3, 2');
    await updates.press('Enter');
  });
  expect(explicit.project.steps!.evaluate.inputs?.subject).toMatchObject({
    step: 'train', checkpoint: [{ at: 4 }, { at: 3 }, { at: 2 }],
  });
  const revised = await plan(explicit.revision);
  const [series] = revised.window_map!.evaluations!;
  expect(series.request_id).not.toBe(planned.request_id);
  expect(series.checkpoint_series?.requested).toEqual([4, 3, 2]);
  expect(revised.rows.find(({ key }) => key.step === 'train')!.request_id).toBe(trainRequest);
  const [evaluate] = await submit(page, runsPath);
  expect(evaluate.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  const [completed] = (await plan(explicit.revision)).window_map!.evaluations!;
  expect(completed.result).toMatchObject({
    job_id: evaluate.job_id, placement: 'ordinary',
    checkpoint_sources: [4, 2].map((completed_updates) => ({ job_id: train.job_id, completed_updates })),
    checkpoint_gaps: [{ update: 3, reason: 'checkpoint_unavailable' }],
  });
  await tab(page, 'Results').click();
  const results = page.getByRole('table', { name: 'Checkpoint Results' });
  await expect(results.getByRole('rowheader')).toHaveText(['4', '3', '2']);
  await expect(results.getByRole('row').filter({ has: page.getByRole('rowheader', { name: '3', exact: true }) }))
    .toContainText('Checkpoint unavailable');
  await expect(page.getByText('From saved checkpoints', { exact: true })).toBeVisible();
  await scanStudio(page, info, 'evaluate-checkpoint-results');
  await captureStudioWindow(page, info, 'checkpoint-results');
  const replay = posted(page, projectPath + '/evaluate-replay');
  await results.getByRole('button', { name: 'Replay update 2' }).click();
  const replayed = await replay;
  expect(replayed.ok(), await replayed.text()).toBe(true);
  expect(replayed.request().postDataJSON()).toMatchObject({ checkpoint: 2, job_id: evaluate.job_id });
  await expect(page.getByRole('tabpanel', { name: 'Scene', exact: true })).toBeVisible();
  await expect(footer(page).getByRole('slider', { name: 'Playback Position' })).toBeVisible();

  async function observeObjective() {
    const reading = posted(page, projectPath + '/evaluate-objective');
    await chooseStep(page, 'train');
    await plans(page).getByRole('button').first().click();
    await page.getByRole('button', { name: 'Open Run', exact: true }).click();
    const read = await reading;
    expect(read.ok(), await read.text()).toBe(true);
    expect(read.request().postDataJSON()).toEqual({
      revision: explicit.revision, key: completed.key, job_id: evaluate.job_id,
    });
    const curve = await read.json() as StudioEvaluationObjectiveSeries;
    expect(curve.lineage).toEqual([train.job_id]);
    expect(curve.gaps.map(({ update }) => update)).toEqual([3]);
    const group = page.getByRole('group', { name: 'Objective evaluate · effort', exact: true });
    const marks = group.locator('circle');
    await expect(marks).toHaveCount(2);
    await expect(group.locator('polyline')).toHaveCount(0);
    const sorted = [...curve.points].sort((left, right) =>
      left.checkpoint.completed_updates - right.checkpoint.completed_updates);
    await expect(marks.locator('title')).toHaveText(sorted.map(({ checkpoint, terms: [term] }) =>
      `evaluate · effort · Updates ${checkpoint.completed_updates} · ${term.value} · Coverage ${term.coverage}`));
    return curve;
  }
  const curve = await observeObjective();
  await page.reload();
  expect(await observeObjective()).toEqual(curve);
  await info.attach('checkpoint-series-lineage', {
    body: JSON.stringify({ train, planned, series, completed, curve }),
    contentType: 'application/json',
  });

  await chooseStep(page, 'evaluate');
  await reveal(page, 'Inspector');
  await tab(page, 'Runs').click();
  async function edit(action: () => Promise<unknown>) {
    const saving = page.waitForResponse((response) =>
      response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  const swept = await edit(() => page.getByRole('button', { name: 'To Sweep', exact: true }).click());
  const definition = swept.project.steps!.evaluate;
  expect(definition.inputs?.subject).not.toHaveProperty('checkpoint');
  expect(definition.sweep?.axes?.update).toMatchObject({ checkpoint_rule: [{ at: 4 }, { at: 3 }, { at: 2 }] });
  const split = await plan(swept.revision);
  const points = split.window_map!.evaluations!;
  expect(points.map((point) => point.checkpoint_series!.requested)).toEqual([[4], [3], [2]]);
  expect(split.window_map!.rows.filter(({ key }) => key.step === 'evaluate')
    .map(({ coordinates }) => coordinates.update.value_id)).toEqual(['4', '3', '2']);
  const gap = points[1];
  const gapRow = split.window_map!.rows.find(({ key }) => key.row === gap.key.row && key.step === 'evaluate')!;
  await plans(page).getByRole('button', { name: runName(gapRow.label) }).click();
  const [gapJob] = await submit(page, runsPath);
  expect(gapJob.disposition).toBe('admitted');
  await expect.poll(async () => (await plan(swept.revision)).window_map!.evaluations!
    .find(({ key }) => key.row === gap.key.row)?.result?.checkpoint_gaps, { timeout: 14_000 })
    .toEqual([{ update: 3, reason: 'checkpoint_unavailable' }]);
  const undone = await edit(() => page.keyboard.press('Meta+Z'));
  expect(undone.project.steps!.evaluate.inputs?.subject).toMatchObject({ checkpoint: [{ at: 4 }, { at: 3 }, { at: 2 }] });
  expect(undone.project.steps!.evaluate.sweep?.axes?.update).toBeUndefined();
  await page.reload();
  await chooseStep(page, 'evaluate');
  await reveal(page, 'Inspector');
  await expect(updates).toHaveValue('4, 3, 2');

  // An update the finished Train never saved is the Checkpoint field's problem before anything runs.
  await pick(kind, 'At');
  const update = checkpoint.getByRole('textbox', { name: 'Update', exact: true });
  const missing = await saved(async () => {
    await update.fill('3');
    await update.press('Enter');
  });
  expect(missing.project.steps!.evaluate.inputs?.subject).toMatchObject({ checkpoint: { at: 3 } });
  await expect(checkpoint.getByRole('img', { name: 'Problem', exact: true }))
    .toHaveAttribute('data-app-tip', /Update 3 .*2, 4/);
  await expect(update).toHaveAttribute('aria-invalid', 'true');
});

test('edits the checkpoint rule in the selected authored variant', async ({ page, request, studio }, info) => {
  const project = JSON.parse(await runProjectFixture(fixture, 'seed-series-variants', studio.home));
  const projectPath = `/api/projects/${project}`;
  const host = await request.get('/api/jobs');
  expect(host.ok(), await host.text()).toBe(true);
  const initial = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  async function plan(revision: string) {
    const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: true } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  const before = await plan(initial.revision);
  await page.goto(`/?project=${project}`);
  await chooseStep(page, 'evaluate');
  await pick(page.getByRole('combobox', { name: 'Variant', exact: true }), 'comparison');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await reveal(page, 'Inspector');
  const kind = inspector.getByRole('combobox', { name: 'Checkpoint', exact: true });
  await expect(kind).toHaveText('Last');
  await pick(kind, 'Every');
  await inspector.getByRole('textbox', { name: 'From', exact: true }).fill('3');
  const saving = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  const every = inspector.getByRole('textbox', { name: 'Every', exact: true });
  await every.fill('2');
  await every.press('Enter');
  const response = await saving;
  expect(response.ok(), await response.text()).toBe(true);
  const after = await response.json() as StudioProjectSnapshot;
  expect(after.project.steps!.evaluate.inputs?.subject).toMatchObject({ checkpoint: 'last' });
  expect(after.project.sweep?.rows?.comparison?.patches?.['steps.evaluate.inputs.subject.checkpoint'])
    .toMatchObject({ value: { every: 2, from: 3 } });
  const revised = await plan(after.revision);
  const [base, comparison] = ['base', 'comparison'].map((variant) =>
    revised.window_map!.evaluations!.find(({ key }) => key.variant === variant)!);
  expect(base.request_id).toBe(before.window_map!.evaluations!.find(({ key }) => key.variant === 'base')!.request_id);
  expect(comparison.checkpoint_series?.requested).toEqual([4]);
  expect(revised.rows.map(({ request_id }) => request_id)).toEqual(before.rows.map(({ request_id }) => request_id));
  await expect(kind).toHaveText('Every');
  await expect(inspector.getByRole('textbox', { name: 'From', exact: true })).toHaveValue('3');
  await pick(page.getByRole('combobox', { name: 'Variant', exact: true }), 'base');
  await expect(kind).toHaveText('Last');
});

test('edits a checkpoint axis value and one sibling row through saved Plan', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture(fixture, 'scoped-series-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const host = await request.get('/api/jobs');
  expect(host.ok(), await host.text()).toBe(true);
  async function plan(revision: string) {
    const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: true } });
    expect(response.ok(), await response.text()).toBe(true);
    const result = await response.json() as StudioSavedTrainRuns;
    const evaluations = result.window_map!.evaluations!;
    const [left, right] = ['left', 'right'].map((value) =>
      evaluations.find(({ key }) => key.row.includes(`replicate=${value}`))!);
    return { left, right, train: result.rows[0].request_id };
  }
  const before = await plan(snapshot.revision);
  await page.goto(`/?project=${snapshot.project_id}`);
  await chooseStep(page, 'evaluate');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await reveal(page, 'Inspector');
  const kind = inspector.getByRole('combobox', { name: 'Checkpoint', exact: true });
  const every = inspector.getByRole('textbox', { name: 'Every', exact: true });
  const updates = inspector.getByRole('textbox', { name: 'Updates', exact: true });
  async function save(action: () => Promise<unknown>) {
    const saving = page.waitForResponse((response) =>
      response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  await chooseSweepValue(page.getByRole('article', { name: 'Evaluate evaluate', exact: true }), 'replicate', 'left');
  await expect(every).toHaveValue('2');
  await pick(kind, 'List');
  const axis = await save(async () => {
    await updates.fill('3');
    await updates.press('Enter');
  });
  const axisPlan = await plan(axis.revision);
  expect(axisPlan.left.checkpoint_series?.requested).toEqual([3]);
  expect(axisPlan.right.request_id).toBe(before.right.request_id);
  expect(axisPlan.train).toBe(before.train);
  await expect(updates).toHaveValue('3');
  await plans(page).getByRole('button', { name: /replicate=right/ }).click();
  await expect(every).toHaveValue('2');
  await pick(kind, 'List');
  const row = await save(async () => {
    await updates.fill('1, 4');
    await updates.press('Enter');
  });
  const rowPlan = await plan(row.revision);
  expect(rowPlan.right.checkpoint_series?.requested).toEqual([1, 4]);
  expect(rowPlan.left.request_id).toBe(axisPlan.left.request_id);
  expect(rowPlan.train).toBe(before.train);
  // The field keeps the keyboard after its save, and Meta+Z there undoes its text; the project's Undo is the window's.
  await updates.blur();
  const undone = await save(() => page.keyboard.press('Meta+Z'));
  expect((await plan(undone.revision)).right.request_id).toBe(before.right.request_id);
  await expect(every).toHaveValue('2');
});

/** An edit left while the plan of the one before it is read saves on the run it was made on, not on the run chosen next. */
test('a checkpoint edit left for another run while the plan is read saves on its own run', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture(fixture, 'scoped-series-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  async function evaluations(revision: string) {
    const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: true } });
    expect(response.ok(), await response.text()).toBe(true);
    const { window_map } = await response.json() as StudioSavedTrainRuns;
    const [left, right] = ['left', 'right'].map((value) =>
      window_map!.evaluations!.find(({ key }) => key.row.includes(`replicate=${value}`))!);
    return { left, right };
  }
  const before = await evaluations(snapshot.revision);
  await page.goto(`/?project=${snapshot.project_id}`);
  await chooseStep(page, 'evaluate');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await reveal(page, 'Inspector');
  const kind = inspector.getByRole('combobox', { name: 'Checkpoint', exact: true });
  const updates = inspector.getByRole('textbox', { name: 'Updates', exact: true });
  await chooseSweepValue(page.getByRole('article', { name: 'Evaluate evaluate', exact: true }), 'replicate', 'left');
  await expect(inspector.getByRole('textbox', { name: 'Every', exact: true })).toHaveValue('2');
  await pick(kind, 'List');
  const patches: StudioProjectSnapshot[] = [];
  page.on('response', async (response) => {
    if (response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath) {
      patches.push(await response.json() as StudioProjectSnapshot);
    }
  });
  const plan = await hold(page, projectPath + '/train-runs', (_, sent) => sent.method() === 'GET');
  await updates.fill('3');
  await updates.press('Enter');
  await plan.sent;
  await updates.fill('4');
  await plans(page).getByRole('button', { name: /replicate=right/ }).click();
  plan.release();
  await expect(inspector.getByRole('heading', { name: /replicate right/ })).toBeVisible();
  await expect(inspector.getByRole('textbox', { name: 'Every', exact: true })).toHaveValue('2');
  await expect.poll(() => patches.length).toBe(2);
  const after = await evaluations(patches[1].revision);
  expect(after.left.checkpoint_series?.requested).toEqual([4]);
  expect(after.right.request_id).toBe(before.right.request_id);
});

test('authors and replays Evaluate from the selected stopped Train checkpoint', { tag: '@round' }, async ({ page, request }, info) => {
  test.setTimeout(58_000);
  const files = JSON.parse(await runProjectFixture(fixture, 'walkthrough-files', '1200'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const runsPath = projectPath + '/train-runs';
  const projection = async (revision: string, stoppedJob?: string) => {
    const response = await request.get(runsPath, { params: {
      revision, window_map: 'true', ...(stoppedJob && { stopped_checkpoint_job: stoppedJob }),
    } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  await page.goto('/?project=' + snapshot.project_id);
  await chooseStep(page, 'train');
  const [producer] = await submit(page, runsPath);
  expect(producer.disposition).toBe('admitted');
  await plans(page).getByRole('button').first().click();
  await page.getByRole('button', { name: 'Open Run', exact: true }).click();
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  await expect(results.getByRole('img', { name: 'Metric train_loss', exact: true }).locator('line title').first())
    .toBeAttached({ timeout: 18_000 });
  const stop = posted(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stopped = await stop;
  expect(stopped.status(), await stopped.text()).toBe(202);
  expect(stopped.request().postDataJSON()).toEqual({ job_ids: [producer.job_id], mode: 'checkpoint' });
  const checkpoint = results.getByRole('region', { name: /Checkpoint at update [0-9]+/ });
  await expect(checkpoint).toBeVisible({ timeout: 16_000 });
  const [train] = (await projection(snapshot.revision, producer.job_id!)).rows;
  const [attempt] = train.attempts;
  expect(attempt).toMatchObject({ job_id: producer.job_id, status: 'stopped', successful_receipt: null });
  const at = attempt.stopped_checkpoint!;
  expect(at).toBeGreaterThan(0);
  expect(at).toBeLessThan(1200);
  const label = `Checkpoint at update ${at}`;
  await expect(checkpoint.getByRole('heading')).toHaveText(label);

  const creating = posted(page, projectPath + '/files');
  await checkpoint.getByRole('button', { name: 'New Step', exact: true }).click();
  await expect(page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true })).toHaveAttribute('aria-disabled', 'true');
  await chooseStepKind(page, 'Evaluate');
  const created = await creating;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: saved } = await created.json() as StudioProjectDocument;
  expect(saved.project.steps!.evaluate.inputs?.subject).toMatchObject({
    step: 'train',
    checkpoint: { at },
  });
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })).toContainText(label);
  await expect(page.getByRole('combobox', { name: 'Subject', exact: true })).toBeFocused();
  const [consumer] = await submit(page, runsPath);
  expect(consumer.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  const [evaluation] = (await projection(saved.revision)).window_map!.evaluations!;
  expect(evaluation.result).toMatchObject({
    job_id: consumer.job_id, checkpoint_source: null,
    checkpoint_sources: [{ job_id: producer.job_id, request_id: train.request_id, completed_updates: at }],
    receipt: { manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
  });
  await page.reload();
  await chooseStep(page, 'evaluate');
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })).toContainText(label);
  const row = plans(page).getByRole('button').first();
  await row.click();
  const lineage = page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('listitem')
    .filter({ has: page.getByRole('button', { name: /^Select Made From / }) });
  await lineage.getByRole('button', { name: label, exact: true }).click();
  await expect(checkpoint.getByRole('heading')).toHaveText(label);

  const plannedLabel = drawnRunLabel((await projection(saved.revision)).window_map!.rows, evaluation.key);
  const replay = await showTrial(page, projectPath + '/evaluate-replay', evaluation, plannedLabel, at);
  expectPositionSamples(replayPosition(replay).samples);
  await info.attach('stopped-checkpoint-evaluation', {
    body: JSON.stringify({ revision: saved.revision, source: train, evaluation, receipt: replay.receipt }),
    contentType: 'application/json',
  });
});

test('saved document Evaluate replays authenticated trials and checkpoints in the Scene', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(85_000);
  const { revision, replayPath, completed, single, finiteRow, fromCheckpoint, analysis } =
    (await runSavedEvaluation(page, request, info, studio.home))!;
  const plannedRows = completed.window_map!.rows;
  const labels = new Map(plannedRows.map(({ key, label }) => [JSON.stringify(key), label]));
  const drawn = (key: StudioSavedEvaluationRow['key']) => drawnRunLabel(plannedRows, key);
  const conditions = single.filter(({ key }) => key.step === 'conditions');
  expect(conditions).toHaveLength(2);
  const activityRun = conditions[1];
  const activityLabel = `${activityRun.key.step} · ${labels.get(JSON.stringify(activityRun.key))!}`;
  await chooseStep(page, 'conditions');
  await plans(page).getByRole('button', { name: runName(drawn(activityRun.key)) }).click();
  await expect(plans(page).getByText('Done', { exact: true }).first()).toBeVisible();
  await expectStudioMapContrast(page);
  const mapAndSteps = page.locator('[data-map-canvas], [data-studio-focus-region="steps"]');
  const vision = await page.context().newCDPSession(page);
  try {
    await vision.send('Emulation.setEmulatedVisionDeficiency', { type: 'achromatopsia' });
    await expect(mapAndSteps).toHaveCount(2);
    // Without colour, a step's kind is carried by its glyph's shape on the card and in the drawer row.
    for (const kind of ['Train', 'Evaluate']) {
      await expect(page.locator('[data-map-canvas]').getByRole('article', { name: new RegExp(`^${kind} `) })
        .first().locator('.studio-card__glyph svg')).toBeVisible();
    }
    const rows = page.getByRole('tree', { name: 'Steps', exact: true }).getByRole('treeitem');
    for (const row of await rows.all()) await expect(row.locator('svg').first()).toBeVisible();
    await captureStudioWindow(page, info, 'train-evaluate-map-greyscale');
  } finally {
    await vision.send('Emulation.setEmulatedVisionDeficiency', { type: 'none' });
    await vision.detach();
  }
  await scanStudio(page, info, 'evaluate-runs');
  await tab(page, 'Model').click();
  await expect(page.getByRole('tabpanel', { name: 'Model', exact: true }).locator('[data-id="plant"]')).toBeVisible();
  const descriptions = await page.locator('[id^="react-flow__"][id*="-desc-"]').evaluateAll((nodes) => nodes.map(({ id }) => id));
  expect(descriptions.length).toBeGreaterThanOrEqual(4);
  expect(new Set(descriptions).size).toBe(descriptions.length);
  await scanStudio(page, info, 'evaluate-model');
  await tab(page, 'Task').click();
  // The Evaluate reads its Task from the Train it evaluates, so a variable's value names that source.
  await page.getByRole('tabpanel', { name: 'Task', exact: true }).getByRole('table', { name: 'Variables', exact: true })
    .getByRole('rowheader').first().click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true }).getByText('Trained by train', { exact: true })).toBeVisible();
  await scanStudio(page, info, 'evaluate-task');
  await chooseStep(page, 'finite');
  await footer(page).getByRole('button', { name: /^Activity( \d+)?$/ }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  // Activity lists every project's runs, grouped by project, with this window's project first.
  await expect(activity.getByRole('region').first()).toHaveAccessibleName('saved-evaluate');
  const ownRuns = activity.getByRole('region', { name: 'saved-evaluate', exact: true });
  const openActivityRun = ownRuns.getByRole('button').filter({ has: page.getByText(activityLabel, { exact: true }) });
  await expect(openActivityRun).toHaveAccessibleName(/^Evaluate/);
  await scanStudio(page, info, 'evaluate-activity');
  const activityReplay = page.waitForResponse((response) => {
    const submitted = response.request();
    if (submitted.method() !== 'POST' || new URL(response.url()).pathname !== replayPath) return false;
    const body = submitted.postDataJSON();
    return body.revision === revision && body.job_id === activityRun.result!.job_id
      && body.key?.step === activityRun.key.step
      && body.key?.row === activityRun.key.row && body.key?.variant === activityRun.key.variant;
  });
  await openActivityRun.click();
  const opened = await activityReplay;
  expect(opened.ok(), await opened.text()).toBe(true);
  expect(opened.request().postDataJSON()).toMatchObject({
    key: activityRun.key, job_id: activityRun.result!.job_id, revision,
  });
  expect((await opened.json() as StudioTrialReplayResponse).receipt).toEqual(activityRun.result!.receipt);
  await expect(activity).toBeHidden();
  await expect(page.getByRole('tabpanel', { name: 'Scene', exact: true })).toBeFocused();
  await expect(footer(page).getByRole('slider', { name: 'Playback Position' })).toBeVisible();
  await scanStudio(page, info, 'evaluate-scene');
  const replayRow = (row: StudioSavedEvaluationRow) => showTrial(page, replayPath, row, drawn(row.key));
  const replays = new Map<string, StudioTrialReplayResponse>();
  for (const row of single) {
    const replay = await replayRow(row);
    const track = replayPosition(replay);
    const observed = analysis[row.result!.job_id];
    expect(track.samples).toEqual(observed.samples);
    expect(replay.position).toBe(observed.position_address);
    replays.set(row.result!.job_id, replay);
  }
  await page.reload();
  for (const row of single) {
    expect(await replayRow(row)).toEqual(replays.get(row.result!.job_id));
  }
  const finiteResult = finiteRow.result!;
  const trialId = finiteRow.trials[trialIndex];
  const finiteReplay = { revision, key: finiteRow.key, job_id: finiteResult.job_id, trial: trialId };
  expect((await request.post(replayPath, { data: finiteReplay })).status()).toBe(422);
  for (const { coordinate, samples } of fromCheckpoint.checkpoints) {
    const response = await request.post(replayPath, { data: { ...finiteReplay, checkpoint: coordinate } });
    expect(response.ok(), await response.text()).toBe(true);
    const replay = await response.json() as StudioTrialReplayResponse;
    expect(replayPosition(replay).samples).toEqual(samples);
    expect(replay.position).toBe(fromCheckpoint.position_address);
    expect(replay.state_time.length).toBe(81);
  }
  await chooseStep(page, 'finite');
  await plans(page).getByRole('button', { name: runName(drawn(finiteRow.key)) }).click();
  await tab(page, 'Scene').click();
  const checkpoints = page.getByRole('combobox', { name: 'Checkpoint', exact: true });
  await expect(checkpoints).toHaveText('2');
  let selectedReplay = posted(page, replayPath, trialId);
  await pickRecordedTrial(page, trialIndex);
  let selected = await selectedReplay;
  expect(selected.ok(), await selected.text()).toBe(true);
  expect(selected.request().postDataJSON()).toMatchObject({ checkpoint: 2 });
  expect(replayPosition(await selected.json() as StudioTrialReplayResponse).samples).toEqual(fromCheckpoint.checkpoints[0].samples);
  selectedReplay = posted(page, replayPath, trialId);
  await pick(checkpoints, '1');
  selected = await selectedReplay;
  expect(selected.ok(), await selected.text()).toBe(true);
  expect(selected.request().postDataJSON()).toMatchObject({ checkpoint: 1 });
  expect(replayPosition(await selected.json() as StudioTrialReplayResponse).samples).toEqual(fromCheckpoint.checkpoints[1].samples);
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toHaveValue('0');
  await page.getByRole('slider', { name: 'Playback Position' }).press('End');

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const cursors = page.getByRole('tabpanel', { name: 'Scene', exact: true }).locator('[data-scene-cursor]');
  await expect(cursors).toHaveCount(2);
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  await expect(clock).toHaveCount(1);
  await expect(trialPicker(page)).toHaveText(`trial ${trialIndex + 1} of ${finiteRow.trials.length}`);
  // Each slot projects the same trial into its own fitted viewport.
  const frame = (end: boolean) => cursors.evaluateAll((points, end) => points.map((point) => {
    const cursor = point as SVGCircleElement;
    const svg = cursor.ownerSVGElement!;
    const trail = cursor.parentElement!.querySelector('polyline')!.points;
    const sample = trail.getItem(end ? trail.numberOfItems - 1 : 0);
    return {
      fitsSlot: Math.abs(svg.viewBox.baseVal.width - svg.clientWidth) <= 1 &&
        Math.abs(svg.viewBox.baseVal.height - svg.clientHeight) <= 1,
      atSample: Math.abs(cursor.cx.baseVal.value - sample.x) < 0.01 &&
        Math.abs(cursor.cy.baseVal.value - sample.y) < 0.01,
    };
  }), end);
  const fitted = Array.from({ length: 2 }, () => ({ fitsSlot: true, atSample: true }));
  await expect.poll(() => frame(true)).toEqual(fitted);
  await captureStudioWindow(page, info, 'saved-evaluate-replay');
  await clock.getByRole('slider').press('Home');
  await expect(clock.getByRole('slider')).toHaveValue('0');
  await expect.poll(() => frame(false)).toEqual(fitted);
  await tab(page, 'Runs').click();
  await expect(clock).toBeVisible();
});
