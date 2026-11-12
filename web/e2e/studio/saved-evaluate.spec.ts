import { expect, test, type Page } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectSnapshot, StudioSavedEvaluationRow, StudioSavedTrainRuns, WorkspaceReplayTrial,
} from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const enabled = process.env.FEEDBAX_STUDIO_EVALUATION_CANDIDATE === '1';
const fixture = 'web.e2e.studio.fixtures.canonical_evaluate';
const trialId = 'beta';
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

async function showTrial(page: Page, path: string, row: StudioSavedEvaluationRow, label: string) {
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
  expect(replay.time.length).toBe(101);
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

test(`saved Evaluate is ${enabled ? 'playable in the isolated candidate' : 'held by default'}`, async ({ page, request }, info) => {
  test.setTimeout(50_000);
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
  const initialPlan = (await plan()).window_map!;
  const [initial] = initialPlan.evaluations!.filter(({ key }) => key.step === 'baseline');
  expect(initial.can_run).toBe(false);
  expect(initialPlan.problems).toContainEqual(expect.objectContaining({
    step: 'baseline', code: 'cannot_run', path: enabled ? '/inputs/subject' : '/evaluation_type',
  }));
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  if (!enabled) {
    await expect(plans(page).getByText('Can\'t run', { exact: true })).toBeVisible();
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
  await expect(plans(page).getByText(/requires a Train checkpoint selected by last/)).toBeVisible();
  await chooseStep(page, 'train');
  const [train] = await submit(page, path);
  expect(train.disposition).toBe('admitted');
  await expect(plans(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });

  await chooseStep(page, 'baseline');
  const baseline = await submit(page, path);
  await chooseStep(page, 'conditions');
  const conditions = await submit(page, path);
  const submitted = [...baseline, ...conditions];
  expect(submitted.map(({ disposition }) => disposition)).toEqual(['admitted', 'admitted', 'admitted']);
  await expect(plans(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 22_000 });
  await expect(plans(page).getByRole('columnheader', { name: 'Size', exact: true })).toBeVisible();
  const settled = await plan();
  const formatBytes = new Intl.NumberFormat('en-US', { style: 'unit', unit: 'byte', unitDisplay: 'narrow' });
  const conditionSizes = settled.window_map!.evaluations!
    .filter(({ key }) => key.step === 'conditions').map(({ result }) => result!.size_bytes);
  expect(new Set(conditionSizes).size).toBe(1);
  await expect(plans(page).getByText(formatBytes.format(conditionSizes[0]), { exact: true })).toHaveCount(2);
  await chooseStep(page, 'baseline');
  const baselineSize = settled.window_map!.evaluations!.find(({ key }) => key.step === 'baseline')!.result!.size_bytes;
  await expect(plans(page).getByText(formatBytes.format(baselineSize), { exact: true })).toHaveCount(1);
  const labels = new Map(settled.window_map!.rows.map(({ key, label }) => [JSON.stringify(key), label]));
  const replayRow = (row: StudioSavedEvaluationRow) => showTrial(page, replayPath, row, labels.get(JSON.stringify(row.key))!);
  const rows = settled.window_map!.evaluations!.filter(({ result }) => result);
  expect(rows).toHaveLength(3);
  expect(new Set(rows.map(({ result }) => result!.job_id))).toEqual(new Set(submitted.map(({ job_id }) => job_id)));
  expect(new Set(rows.map(({ result }) => result!.receipt.manifest_sha256)).size).toBe(3);
  const source = settled.rows.find(({ key }) => key.step === 'train')!;
  for (const { result } of rows) {
    expect(result!.checkpoint_source).toEqual({ job_id: train.job_id, request_id: source.request_id, completed_updates: 2 });
    expect(result!.size_bytes).toBeGreaterThan(13_328);
  }
  const replays = new Map<string, WorkspaceReplayTrial>();
  for (const row of rows) replays.set(row.result!.job_id, await replayRow(row));
  const base = rows.find(({ key }) => key.step === 'baseline')!;
  const invalidTrial = await request.post(replayPath, {
    data: { revision, key: base.key, job_id: base.result!.job_id, trial: 'absent' },
  });
  expect(invalidTrial.status()).toBe(422);
  const wrongJob = await request.post(replayPath, {
    data: { revision, key: base.key, job_id: train.job_id, trial: trialId },
  });
  expect(wrongJob.status()).toBe(404);

  await page.reload();
  const reopened = (await plan()).window_map!.evaluations!.filter(({ result }) => result);
  expect(reopened).toEqual(rows);
  for (const row of reopened) {
    expect(await replayRow(row)).toEqual(replays.get(row.result!.job_id));
  }

  const analysis = JSON.parse(await runProjectFixture(
    fixture, 'observe', home, snapshot.project_id, revision, trialId,
  ));
  expect(Object.keys(analysis)).toHaveLength(3);
  for (const { key, request_id, result } of reopened) {
    const [track] = replays.get(result!.job_id)!.tracks!;
    expect(analysis[result!.job_id]).toEqual({
      key: [key.variant, key.step, key.row], request_id, receipt: result!.receipt,
      checkpoint_source: result!.checkpoint_source, size_bytes: result!.size_bytes,
      trial: trialId, samples: track.samples, tensor_reads: [track.selector.compact],
    });
  }

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const cursors = page.getByRole('tabpanel', { name: 'Scene', exact: true }).locator('[data-scene-cursor]');
  await expect(cursors).toHaveCount(2);
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  await expect(clock).toHaveCount(1);
  await expect(page.getByRole('combobox', { name: 'Trials', exact: true })).toHaveValue(trialId);
  const coordinates = () => cursors.evaluateAll((points) =>
    points.map((point) => [point.getAttribute('cx'), point.getAttribute('cy')]));
  const atEnd = await coordinates();
  expect(atEnd[0]).toEqual(atEnd[1]);
  await clock.getByRole('slider').press('Home');
  await expect(clock.getByRole('slider')).toHaveValue('0');
  const atReset = await coordinates();
  expect(atReset[0]).toEqual(atReset[1]);
  expect(atReset[0]).not.toEqual(atEnd[0]);
  await tab(page, 'Runs').click();
  await expect(clock).toBeVisible();
  await info.attach('saved-evaluation-lineage', {
    body: JSON.stringify({ project: snapshot.project_id, revision, train: source.result, rows: reopened, analysis }),
    contentType: 'application/json',
  });
});
