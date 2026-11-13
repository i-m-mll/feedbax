import { expect, test, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectSnapshot, StudioSavedEvaluationRow, StudioSavedTrainRuns, WorkspaceReplayTrial,
} from '../../src/generated/studioContracts';
import { captureStudioWindow, mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const enabled = process.env.FEEDBAX_STUDIO_EVALUATION_CANDIDATE === '1';
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

function expectPositionSamples(samples: number[][]) {
  expect(samples).toHaveLength(81);
  expect(samples.every((sample) => sample.length === 2 && sample.every(Number.isFinite))).toBe(true);
}

async function showTrial(page: Page, path: string, row: StudioSavedEvaluationRow, label: string) {
  const trialId = row.trials[trialIndex];
  await chooseStep(page, row.key.step);
  await plans(page).getByRole('button', { name: label, exact: true }).click();
  await tab(page, 'Scene').click();
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  if (!await drawer.isVisible()) await footer(page).getByRole('button', { name: 'Show Drawer' }).click();
  const trials = drawer.getByRole('combobox', { name: 'Trials' });
  const response = posted(page, path, trialId);
  await trials.selectOption(trialId);
  const reply = await response;
  expect(reply.ok(), await reply.text()).toBe(true);
  expect(reply.request().postDataJSON()).toMatchObject({ key: row.key, job_id: row.result!.job_id, trial: trialId });
  const replay = await reply.json() as WorkspaceReplayTrial;
  expect(replay.identity).toMatchObject({ index: 1, stable_id: trialId, source: 'stable_id' });
  expect(replay.manifest_refs.producer_manifest).toMatchObject({
    kind: 'EvaluationRunManifest', id: row.result!.receipt.manifest_id,
    metadata: { manifest_sha256: row.result!.receipt.manifest_sha256 },
  });
  expect(replay.time.length).toBe(81);
  expect(replay.time.values?.[0]).toBe(0);
  const cursor = scene.locator('[data-scene-cursor]');
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  const scrubber = clock.getByRole('slider', { name: 'Playback Position' });
  await expect(scrubber).toHaveValue('0');
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toHaveCount(1);
  const coordinates = () => cursor.evaluate((point) => [point.getAttribute('cx'), point.getAttribute('cy')]);
  const reset = await coordinates();
  await scrubber.press('End');
  await expect.poll(coordinates).not.toEqual(reset);
  await expect(scrubber).toHaveValue(String(replay.time.values!.at(-1)));
  await drawer.getByRole('button', { name: 'Effector', exact: true }).click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  if (!await inspector.isVisible()) await footer(page).getByRole('button', { name: 'Show Inspector' }).click();
  await expect(inspector.getByRole('heading', { name: 'Effector' })).toBeVisible();
  return replay;
}

async function runSavedEvaluation(page: Page, request: APIRequestContext, info: TestInfo) {
  const home = String(info.config.metadata.home);
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
  await expect(plans(page).getByText("Can't run", { exact: true })).toBeVisible();
  const initialPlan = (await plan()).window_map!;
  const [initial] = initialPlan.evaluations!.filter(({ key }) => key.step === 'baseline');
  expect(initial.can_run).toBe(false);
  expect(initialPlan.problems).toContainEqual(expect.objectContaining({
    step: 'baseline', code: 'cannot_run', path: enabled ? '/inputs/subject' : '/evaluation_type',
  }));
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'New Step baseline', exact: true })).toHaveCount(0);
  if (!enabled) {
    const refused = await request.post(path, { data: { revision, run_keys: [initial.key], token: 'held' } });
    expect(refused.status()).toBe(202);
    expect((await refused.json() as StudioJobSubmission).items).toMatchObject([
      { disposition: 'rejected', reason: expect.stringContaining('/evaluation_type') },
    ]);
    await tab(page, 'Scene').click();
    await expect(footer(page).getByRole('group', { name: 'Clock' })).toHaveCount(0);
    return;
  }

  await chooseStep(page, 'unsupported');
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
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
  await expect(page.getByRole('article', { name: 'Evaluate baseline', exact: true })
    .getByText('1 Done', { exact: true })).toBeVisible({ timeout: 22_000 });
  await expect(plans(page).getByRole('columnheader', { name: 'Size', exact: true })).toBeVisible();
  const settled = await plan();
  const formatBytes = new Intl.NumberFormat('en-US', { style: 'unit', unit: 'byte', unitDisplay: 'narrow' });
  const conditionSizes = settled.window_map!.evaluations!
    .filter(({ key }) => key.step === 'conditions').map(({ result }) => result!.size_bytes);
  expect(conditionSizes).toHaveLength(2);
  for (const size of new Set(conditionSizes)) {
    await expect(plans(page).getByText(formatBytes.format(size), { exact: true }))
      .toHaveCount(conditionSizes.filter((other) => other === size).length);
  }
  await chooseStep(page, 'baseline');
  const baselineSize = settled.window_map!.evaluations!.find(({ key }) => key.step === 'baseline')!.result!.size_bytes;
  await expect(plans(page).getByText(formatBytes.format(baselineSize), { exact: true })).toHaveCount(1);
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

test(`saved Evaluate ${enabled ? 'runs in the isolated candidate' : 'is held by default'}`, async ({ page, request }, info) => {
  test.setTimeout(60_000);
  await runSavedEvaluation(page, request, info);
});

test('saved document Evaluate replays authenticated trials and checkpoints in the Scene', async ({ page, request }, info) => {
  test.skip(!enabled, 'Requires the Evaluate candidate flag');
  test.setTimeout(65_000);
  const { revision, replayPath, completed, single, finiteRow, fromCheckpoint, analysis } =
    (await runSavedEvaluation(page, request, info))!;
  const labels = new Map(completed.window_map!.rows.map(({ key, label }) => [JSON.stringify(key), label]));
  const replayRow = (row: StudioSavedEvaluationRow) => showTrial(page, replayPath, row, labels.get(JSON.stringify(row.key))!);
  const replays = new Map<string, WorkspaceReplayTrial>();
  for (const row of single) {
    const replay = await replayRow(row);
    const [track] = replay.tracks!;
    const observed = analysis[row.result!.job_id];
    expect(track.samples).toEqual(observed.samples);
    expect(observed.tensor_reads).toContain(track.selector.compact);
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
    const replay = await response.json() as WorkspaceReplayTrial;
    expect(replay.tracks![0].samples).toEqual(samples);
    expect(fromCheckpoint.tensor_reads).toContain(replay.tracks![0].selector.compact);
    expect(replay.time.length).toBe(81);
  }
  await chooseStep(page, 'finite');
  await plans(page).getByRole('button', { name: labels.get(JSON.stringify(finiteRow.key))!, exact: true }).click();
  await tab(page, 'Scene').click();
  const checkpoints = page.getByRole('combobox', { name: 'Checkpoint', exact: true });
  await expect(checkpoints).toHaveValue('2');
  const finiteTrials = page.getByRole('combobox', { name: 'Trials', exact: true });
  let selectedReplay = posted(page, replayPath, trialId);
  await finiteTrials.selectOption(trialId);
  let selected = await selectedReplay;
  expect(selected.ok(), await selected.text()).toBe(true);
  expect(selected.request().postDataJSON()).toMatchObject({ checkpoint: 2 });
  expect((await selected.json() as WorkspaceReplayTrial).tracks![0].samples).toEqual(fromCheckpoint.checkpoints[0].samples);
  selectedReplay = posted(page, replayPath, trialId);
  await checkpoints.selectOption('1');
  selected = await selectedReplay;
  expect(selected.ok(), await selected.text()).toBe(true);
  expect(selected.request().postDataJSON()).toMatchObject({ checkpoint: 1 });
  expect((await selected.json() as WorkspaceReplayTrial).tracks![0].samples).toEqual(fromCheckpoint.checkpoints[1].samples);
  await expect(page.getByRole('slider', { name: 'Playback Position' })).toHaveValue('0');
  await page.getByRole('slider', { name: 'Playback Position' }).press('End');

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const cursors = page.getByRole('tabpanel', { name: 'Scene', exact: true }).locator('[data-scene-cursor]');
  await expect(cursors).toHaveCount(2);
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  await expect(clock).toHaveCount(1);
  await expect(page.getByRole('combobox', { name: 'Trials', exact: true })).toHaveValue(trialId);
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
