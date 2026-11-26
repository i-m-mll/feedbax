import { expect, test } from './fixtures/studio';
import { type Page } from '@playwright/test';
import { authorLearningRateSweep, captureStudioWindow, chooseSweepValue, chooseTrainSuccessor, fixture, mapStep, openResultsSlot, runName, shownRunLabel, seedManagedProject, mapCard } from './fixtures/canonicalTrain';
import { mapName } from './fixtures/members';
import { scanStudio } from './fixtures/studioAccessibility';
import { formatMetric } from '../../src/components/activity/studioPlot';
import type {
  StudioActivityJob,
  StudioJobEventHistory,
  StudioJobSnapshot,
  StudioJobSubmission,
  StudioProjectDocument,
  StudioProjectSnapshot,
  StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { pick } from './fixtures/controls';
import { palette } from './fixtures/studioKeyboard';

function submitted(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('standalone Activity opens an existing run and hydrates its retained history', { tag: '@round' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const admission = await request.post(`${projectPath}/train-runs`, { data: {
    token: 'background-activity', revision: snapshot.revision,
    run_keys: [{ variant: 'default', step: 'train', row: 'default' }],
  } });
  expect(admission.status(), await admission.text()).toBe(202);
  const { items: [attempt] } = await admission.json() as StudioJobSubmission;
  await page.goto('/?activity=1');
  await expect(page.getByRole('heading', { name: 'Activity', exact: true })).toBeVisible();
  const announcements = page.getByRole('log', { name: 'Status', exact: true });
  await expect(announcements).toHaveCount(1);
  const row = page.getByRole('list', { name: 'Runs', exact: true })
    .getByRole('region', { name: snapshot.project_id, exact: true }).getByRole('listitem');
  await expect(row.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await row.getByRole('button', { name: /^Train / }).click();
  const inspector = page.getByRole('region', { name: 'Run Inspector' });
  await expect(inspector.getByRole('region', { name: 'Log', exact: true })).toBeVisible();
  await expect(row.getByRole('button', { name: 'Dismiss', exact: true })).toBeHidden();
  await expect(announcements).toHaveCount(1);
  expect((await request.get('/api/jobs').then((response) => response.json()) as StudioJobSnapshot)
    .jobs.some(({ job_id, seen }) => job_id === attempt.job_id && seen)).toBe(true);
});

test('Mounted saved Train runs a canonical successor through Jobs', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(48_000);
  const created = await request.post('/api/projects', { data: { name: 'canonical-successor' } });
  expect(created.status()).toBe(201);
  let { snapshot } = await created.json() as StudioProjectDocument;
  const id = snapshot.project_id;
  const projectPath = `/api/projects/${id}`;
  for (const file of await fixture('managed-files')) {
    const saved = await request.post(`${projectPath}/files`, { data: { revision: snapshot.revision, file } });
    expect(saved.status(), await saved.text()).toBe(201);
    ({ snapshot } = await saved.json() as StudioProjectDocument);
  }
  const path = `${projectPath}/train-runs`;
  const table = page.getByRole('table', { name: 'Runs' });
  const runButton = page.getByRole('button', { name: 'Run', exact: true });
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  await table.getByRole('button').first().click();
  const initial = await request.get(`${path}?revision=${snapshot.revision}`);
  expect(initial.ok(), await initial.text()).toBe(true);
  const reviewed = await initial.json() as StudioSavedTrainRuns;
  const original = reviewed.rows.find(({ key }) => key.step === 'train')!;
  expect(original.attempts).toEqual([]);
  const admission = submitted(page, path);
  await runButton.click();
  const accepted = await admission;
  expect(accepted.status(), await accepted.text()).toBe(202);
  const { items } = await accepted.json() as StudioJobSubmission;
  expect(items).toEqual([expect.objectContaining({
    run_key: '["default","train","default"]', disposition: 'admitted', job_id: expect.any(String),
  })]);
  const job = items[0].job_id!;
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  await page.reload();
  await mapStep(page, 'train').click();
  await expect(table.getByText('Done', { exact: true })).toBeVisible();
  const reopened = await request.get(`${path}?revision=${snapshot.revision}`);
  const current = await reopened.json() as StudioSavedTrainRuns;
  const finished = current.rows.find(({ key }) => key.step === 'train')!;
  expect(finished.request_id).toBe(original.request_id);
  expect(finished.attempts).toEqual([expect.objectContaining({
    job_id: job, status: 'succeeded', progress: { completed_updates: 2, total_updates: 2 },
    successful_receipt: expect.objectContaining({ manifest_sha256: expect.any(String) }),
  })]);

  await mapStep(page, 'successor').click();
  await table.getByRole('button').first().click();
  const chained = submitted(page, path);
  await runButton.click();
  const successorAdmission = await chained;
  expect(successorAdmission.status(), await successorAdmission.text()).toBe(202);
  const successorSubmission = await successorAdmission.json() as StudioJobSubmission;
  expect(successorSubmission.items).toEqual([expect.objectContaining({
    run_key: '["default","successor","default"]', disposition: 'admitted', job_id: expect.any(String),
  })]);
  const stoppedJob = successorSubmission.items[0].job_id!;
  await openResultsSlot(page);
  await expect(runButton).toBeVisible();
  const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
  const loss = results.getByRole('img', { name: 'Metric train_loss', exact: true });
  await expect(loss.locator(`line[data-attempt="${stoppedJob}"] title`).first()).toBeAttached({ timeout: 14_000 });
  const stop = await request.post('/api/jobs/stop', { data: { job_ids: [stoppedJob], mode: 'checkpoint' } });
  expect(stop.status(), await stop.text()).toBe(202);
  await expect(table.getByText('Stopped', { exact: true })).toBeVisible({ timeout: 10_000 });
  const resume = await request.post(`/api/jobs/${stoppedJob}/resume`, { data: { token: 'resume-successor' } });
  expect(resume.ok(), await resume.text()).toBe(true);
  const resumed = await resume.json() as StudioActivityJob;
  expect(resumed.continues).toBe(stoppedJob);
  const successorJob = resumed.job_id;
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });
  const replay = await request.post(path, { data: successorAdmission.request().postDataJSON() });
  expect(replay.status(), await replay.text()).toBe(202);
  expect(await replay.json()).toEqual(successorSubmission);

  await page.reload();
  await mapStep(page, 'successor').click();
  await expect(table.getByText('Done', { exact: true })).toBeVisible();
  await table.getByRole('button').first().click();
  const evidence = await fixture(
    'observe', studio.home, id, snapshot.revision, job, stoppedJob, successorJob,
  );
  expect(evidence).toMatchObject({
    request: original.request_id, attempt: job, delivery: 1,
    completion_updates: 2, joined_status: 'succeeded',
    successor: { attempt: successorJob, delivery: 1, completion_updates: 1200, joined_status: 'succeeded' },
  });
  expect(evidence.successor.resumed_from).toBeGreaterThan(0);
  expect(evidence.successor.resumed_from).toBeLessThan(1200);
  await openResultsSlot(page);
  await expect(loss.locator('polyline')).toHaveCount(1);
  await expect(results.getByText('Cumulative updates')).toBeVisible();
  const [trainName, successorName] = [await mapName(page, 'train'), await mapName(page, 'successor')];
  await expect(results.getByRole('list', { name: 'Train steps' }).getByRole('listitem')).toHaveText([
    `${trainName} 1–2`, `${successorName} 3–1202`,
  ]);
  await expect(loss.locator(`line[data-attempt="${job}"] title`).last()).toContainText(`${trainName} · Local updates `);
  await expect(loss.locator(`line[data-attempt="${stoppedJob}"] title`).last()).toContainText(
    `–${evidence.successor.resumed_from}`,
  );
  await expect(loss.locator(`line[data-attempt="${successorJob}"] title`).first()).toContainText(
    `Local updates ${evidence.successor.resumed_from + 1}–`,
  );
  await expect(loss.locator(`line[data-attempt="${successorJob}"] title`).last()).toContainText(`${successorName} · Local updates `);
  const { last } = evidence.successor.terminal_loss.metrics.train_loss;
  await expect(results.getByRole('region', { name: 'Metric train_loss', exact: true }).locator('[aria-description="Last value at update 1202"]')).toHaveText(
    formatMetric(last),
  );
  await page.getByRole('button', { name: 'Current Result', exact: true }).click();
  const successorResult = page.getByRole('region', { name: 'Current Result' }).first();
  await expect(successorResult).not.toContainText(evidence.successor.request);
  await expect(successorResult).not.toContainText(evidence.successor.receipt.manifest_sha256);
  await expect(successorResult.getByRole('button', { name: 'Copy Details', exact: true })).toBeEnabled();

  await mapStep(page, 'train').click();
  await table.getByRole('button').first().click();
  await expect(results.getByText('Cumulative updates')).toHaveCount(0);
  await expect(results.getByText('Updates', { exact: true })).toBeVisible();
  let edited: StudioProjectSnapshot | undefined;
  await page.route(`**${path}`, async (route) => {
    const change = await request.patch(projectPath, { data: {
      revision: snapshot.revision, address: { kind: 'step', id: 'train' },
      patches: [{ op: 'add', path: '/base/inline/seed', value: 7 }],
    } });
    expect(change.ok(), await change.text()).toBe(true);
    edited = await change.json() as StudioProjectSnapshot;
    await route.continue();
  }, { times: 1 });
  const stale = submitted(page, path);
  await runButton.click();
  const conflict = await stale;
  expect(conflict.status()).toBe(409);
  expect(await conflict.json()).toMatchObject({ detail: { resource: 'project', current: edited!.revision } });
  await expect(page.getByText('Project changed before this edit.', { exact: true }).first()).toBeVisible();
  const jobs = await (await request.get(`/api/jobs?project=${id}`)).json() as StudioJobSnapshot;
  expect(jobs.jobs.map(({ job_id }) => job_id).sort()).toEqual([job, stoppedJob, successorJob].sort());
  console.info('Mounted canonical Train', JSON.stringify({ ...evidence, refused_revision: edited!.revision }));
});

test('Saved Train compares authenticated receipts after a partial run and stopped edit', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(65_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const created = await request.post('/api/projects', { data: { name: 'canonical-resume' } });
  expect(created.status()).toBe(201);
  let { snapshot } = await created.json() as StudioProjectDocument;
  const id = snapshot.project_id;
  const projectPath = `/api/projects/${id}`;
  const path = `${projectPath}/train-runs`;
  for (const file of await fixture('history-files')) {
    const response = await request.post(`${projectPath}/files`, { data: { revision: snapshot.revision, file } });
    expect(response.status(), await response.text()).toBe(201);
    ({ snapshot } = await response.json() as StudioProjectDocument);
  }
  const table = page.getByRole('table', { name: 'Runs' });
  const runButton = page.getByRole('button', { name: 'Run', exact: true });
  async function select(name: string) {
    await mapStep(page, name).click();
    await table.getByRole('button').first().click();
  }
  async function runSelected(name: string) {
    await select(name);
    const admission = submitted(page, path);
    await runButton.click();
    const response = await admission;
    expect(response.status(), await response.text()).toBe(202);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items).toEqual([expect.objectContaining({
      run_key: JSON.stringify(['default', name, 'default']),
      disposition: 'admitted',
      job_id: expect.any(String),
    })]);
    return items[0].job_id!;
  }
  async function rows() {
    const saved = await request.get(projectPath);
    expect(saved.ok(), await saved.text()).toBe(true);
    snapshot = await saved.json() as StudioProjectSnapshot;
    const response = await request.get(`${path}?revision=${snapshot.revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).rows;
  }
  async function trainRow() {
    return (await rows()).find(({ key }) => key.step === 'train')!;
  }
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  const loss = results.getByRole('img', { name: 'Metric train_loss', exact: true });
  const observed = () => fixture('observe-history', studio.home, id, snapshot.revision);
  type LossSample = { update: number; value: number };
  async function expectLoss(jobId: string, expected: LossSample[]) {
    const series = loss.locator(`[data-result-attempt="${jobId}"]`);
    await expect(series.locator('polyline')).toBeAttached();
    await expect.poll(() => series.locator('line[data-update][data-value]').evaluateAll((points) =>
      points.map((point) => ({ update: Number(point.getAttribute('data-update')), value: Number(point.getAttribute('data-value')) })),
    )).toEqual(expected);
    await expect.poll(() => series.locator('polyline').evaluate((line: SVGPolylineElement) => line.points.numberOfItems))
      .toBe(expected.length);
    const locations = await series.evaluate((group) => {
      const curve = group.querySelector('polyline')!;
      return Array.from(group.querySelectorAll<SVGLineElement>('line[data-update]'), (line, index) => {
        const point = curve.points.getItem(index);
        const first = Number(line.getAttribute('y1'));
        const last = Number(line.getAttribute('y2'));
        return {
          update: Number(line.getAttribute('data-update')),
          aligned: Math.abs(point.x - Number(line.getAttribute('x1'))) < 0.001
            && point.y >= Math.min(first, last) - 0.001 && point.y <= Math.max(first, last) + 0.001,
        };
      });
    });
    expect(locations).toEqual(expected.map(({ update }) => ({ update, aligned: true })));
  }
  async function editNumber(label: string, value: number, before: string) {
    await mapStep(page, 'train').click();
    await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
    const field = page.getByRole('textbox', { name: label, exact: true });
    await expect(field).toHaveValue(before);
    const saved = page.waitForResponse((response) => {
      const request = response.request();
      return request.method() === 'PATCH' && new URL(response.url()).pathname === projectPath
        && request.postDataJSON().address?.id === 'train';
    });
    await field.fill(String(value));
    await field.press('Enter');
    const acknowledged = await saved;
    expect(acknowledged.ok(), await acknowledged.text()).toBe(true);
    const edited = await acknowledged.json() as StudioProjectSnapshot;
    expect(edited.revision).not.toBe(snapshot.revision);
    snapshot = edited;
    await expect(field).toHaveValue(String(value));
    await expect(field).toBeEnabled();
  }

  await page.goto(`/?project=${encodeURIComponent(id)}`);
  const firstJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const first = (await trainRow()).result!;
  expect(first.job_id).toBe(firstJob);
  await editNumber('Seed', 7, '0');
  await expect(table.getByText('Out of date', { exact: true })).toBeVisible();
  await table.getByRole('row').filter({ hasText: 'Out of date' }).getByRole('button').click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('button', { name: 'Seed · 0 → 7', exact: true })).toBeVisible();
  const stale = await trainRow();
  expect(stale).toMatchObject({ status: 'stale', result: null, previous: first });
  expect(stale.request_id).not.toBe(first.request_id);

  await runSelected('sibling');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  expect(await trainRow()).toMatchObject({ status: 'stale', result: null, previous: first, attempts: [] });

  await select('train');
  await editNumber('Batches', 10000, '2');
  const stoppedJob = await runSelected('train');
  const stop = await request.post('/api/jobs/stop', { data: { job_ids: [stoppedJob], mode: 'force' } });
  expect(stop.status(), await stop.text()).toBe(202);
  await expect.poll(async () => (await trainRow()).status).toBe('stopped');
  expect(await trainRow()).toMatchObject({ result: null, previous: first });

  await openResultsSlot(page);
  await results.getByRole('button', { name: 'Previous Result', exact: true }).click();
  const stopped = await observed();
  expect(stopped).toMatchObject({ current: null, previous: first });
  await expectLoss(firstJob, stopped.loss.previous);
  await expect(loss.locator(`[data-result-attempt="${stoppedJob}"]`)).toHaveCount(0);
  await expect(loss.locator('polyline')).toHaveCount(1);
  await results.getByRole('button', { name: 'Close Slot', exact: true }).click();

  await editNumber('Batches', 2, '10000');
  const changedJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const changed = await trainRow();
  expect(changed.result!.job_id).toBe(changedJob);
  expect(changed.previous).toEqual(first);
  expect(changed.change).toMatchObject({ path: expect.stringMatching(/seed$/), before: 0, after: 7 });
  const evidence = await observed();
  expect(evidence).toMatchObject({ current: changed.result, previous: first });
  async function compare() {
    await openResultsSlot(page);
    await results.getByRole('button', { name: 'Compare Results', exact: true }).click();
    await expect(results.getByRole('region', { name: 'Compare Results', exact: true })).toContainText(changed.change!.path);
    await expect(results.getByRole('region', { name: 'Compare Results', exact: true })).toContainText(/0\s*→\s*7/);
    await expect(loss.locator('polyline')).toHaveCount(2);
    await expectLoss(changedJob, evidence.loss.current);
    await expectLoss(firstJob, evidence.loss.previous);
    await expect(loss.locator(`[data-result-attempt="${changedJob}"] polyline`)).not.toHaveAttribute('stroke-dasharray');
    await expect(loss.locator(`[data-result-attempt="${firstJob}"] polyline`)).toHaveAttribute('stroke-dasharray', '4 3');
  }
  await compare();
  await page.reload();
  await select('train');
  await compare();
  await scanStudio(page, info, 'train-results-comparison');
  const current = page.getByRole('region', { name: 'Current Result' });
  const previous = page.getByRole('region', { name: 'Previous Result' });
  await expect(current).not.toContainText(changed.result!.receipt.manifest_sha256);
  await expect(previous).not.toContainText(first.receipt.manifest_sha256);
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  for (const [section, result] of [[current, changed.result!], [previous, first]] as const) {
    await section.getByRole('button', { name: 'Copy Details', exact: true }).click();
    await expect.poll(() => page.evaluate(async () => JSON.parse(await navigator.clipboard.readText()))).toEqual(result);
  }
  await captureStudioWindow(page, info, 'saved-train-loss-comparison');

  let withheld = 0;
  await page.route('**/api/events/history?**', async (route) => {
    const response = await route.fetch();
    const history = await response.json() as StudioJobEventHistory;
    const events = history.events.map((event) => {
      if (event.job_id !== firstJob || event.kind !== 'progress' || !event.data.metric_interval) return event;
      withheld += 1;
      const { metrics: _metrics, metric_interval: _interval, ...data } = event.data;
      return { ...event, data };
    });
    await route.fulfill({ response, json: { ...history, events } });
  });
  await page.reload();
  await select('train');
  await openResultsSlot(page);
  await results.getByRole('button', { name: 'Compare Results', exact: true }).click();
  await expectLoss(changedJob, evidence.loss.current);
  expect(withheld).toBe(evidence.loss.previous.length);
  await expect(loss.locator(`[data-result-attempt="${firstJob}"] polyline`)).toHaveCount(0);
  await expect(results.getByRole('list', { name: 'Metric train_loss', exact: true })
    .getByRole('listitem').filter({ hasText: 'Previous Result' }).getByRole('status')).toHaveText('train_loss unavailable');
  console.info('Mounted saved Train history', JSON.stringify(evidence));
});

test('Mounted field sweep continues exactly two Train rows with state', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(47_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  let { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const id = snapshot.project_id;
  const filesPath = `${projectPath}/files`;
  const runsPath = `${projectPath}/train-runs`;
  const plan = async () => {
    const response = await request.get(`${runsPath}?revision=${snapshot.revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const map = page.getByRole('region', { name: 'Pipeline Map' });
  const source = mapCard(map, 'train');
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const stepInspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const table = runs.getByRole('table', { name: 'Runs' });
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  await expect(table).toBeInViewport();
  await expect(runs.getByRole('region', { name: 'Definition' })).toHaveCount(0);
  await expect(drawer).toBeHidden();
  await expect(stepInspector.getByRole('region', { name: 'Definition' })).toBeVisible();
  await expect(source).toHaveAttribute('data-selected');
  await authorLearningRateSweep(stepInspector.getByRole('region', { name: 'Learning Rate field', exact: true }));

  await expect(source.getByRole('button', { name: 'Sweep Learning rate 3', exact: true })).toBeVisible();
  const saved = await request.get(projectPath);
  expect(saved.ok(), await saved.text()).toBe(true);
  snapshot = await saved.json() as StudioProjectSnapshot;
  const authored = (await plan()).window_map!;
  expect(authored.problems).toEqual([]);
  expect(authored.rows).toHaveLength(3);
  const selected = authored.rows.filter(({ coordinates }) => coordinates['learning-rate'].value !== 0.002);
  const rowKeys = selected.map(({ key }) => key.row);
  expect(selected.map(({ coordinates }) => coordinates['learning-rate'].value)).toEqual([0.001, 0.003]);
  const run = async (step: string) => {
    const response = submitted(page, runsPath);
    await runs.getByRole('button', { name: 'Run', exact: true }).click();
    const admitted = await response;
    expect(admitted.status(), await admitted.text()).toBe(202);
    expect(admitted.request().postDataJSON()).toMatchObject({
      revision: snapshot.revision,
      run_keys: rowKeys.map((row) => ({ variant: 'default', step, row })),
    });
    const result = await admitted.json() as StudioJobSubmission;
    expect(result.items.map(({ disposition }) => disposition)).toEqual(['admitted', 'admitted']);
    await expect(table.getByText('Done', { exact: true })).toHaveCount(2, { timeout: 16_000 });
    return result.items;
  };

  await chooseSweepValue(source, 'Learning rate', '0.001');
  await chooseSweepValue(source, 'Learning rate', '0.003', true);
  await expect(source.locator('.studio-map-group-box')).toHaveCount(2);
  const mark = source.locator('.studio-map-run-mark').first();
  await mark.focus();
  await expect(mark).not.toHaveCSS('box-shadow', 'none');
  const firstRow = table.locator('tbody tr[data-index]').first();
  await firstRow.getByRole('button').focus();
  await expect(firstRow).toHaveCSS('outline-style', 'solid');
  const prefix = await run('train');

  const create = submitted(page, filesPath);
  await source.getByRole('button', { name: 'New Step Train', exact: true }).click();
  await chooseTrainSuccessor(page);
  const accepted = await create;
  expect(accepted.status(), await accepted.text()).toBe(201);
  ({ snapshot } = await accepted.json() as StudioProjectDocument);
  const childId = 'train-2';
  expect(snapshot.project.steps?.[childId]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: { clauses: [{ 'learning-rate': selected.map(({ coordinates }) => ({
      id: coordinates['learning-rate'].value_id, label: String(coordinates['learning-rate'].value),
    })).sort((left, right) => left.id.localeCompare(right.id)) }] } } },
    deltas: [{ patches: [{ value: 'initialize_from' }] }],
  });
  const child = mapCard(map, childId);
  await expect(mapStep(page, childId)).toHaveAttribute('aria-pressed', 'true');
  await expect(child).toContainText('2 of 3');
  await expect(stepInspector.getByRole('combobox', { name: 'Method', exact: true })).toBeFocused();

  const carryover = stepInspector.getByRole('combobox', { name: 'Checkpoint carryover', exact: true });
  const carryoverPath = 'checkpoint_initialization';
  const checkpointInput = snapshot.project.steps?.[childId]?.inputs?.checkpoint;
  const beforeCarry = await plan();
  await expect(carryover).toHaveText('Initialize model only');
  const carryEdit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  await pick(carryover, 'Carry training state');
  const carried = await carryEdit;
  expect(carried.ok(), await carried.text()).toBe(true);
  expect(carried.request().postDataJSON()).toMatchObject({ address: { kind: 'step', id: childId } });
  snapshot = await carried.json() as StudioProjectSnapshot;
  expect(snapshot.project.steps?.[childId]?.inputs?.checkpoint).toEqual(checkpointInput);
  expect(snapshot.project.steps?.[childId]?.deltas?.[1]?.patches).toEqual([
    { op: 'replace', path: carryoverPath, value: {
      mode: 'continue_from', source_completed_updates: 2, state: 'preserve', rng: 'carry', schedule: 'continue',
    } },
  ]);
  await expect(carryover).toHaveText('Carry training state');
  const afterCarry = await plan();
  expect(afterCarry.rows.filter(({ key }) => key.step === 'train'))
    .toEqual(beforeCarry.rows.filter(({ key }) => key.step === 'train'));
  expect(afterCarry.window_map?.edges).toEqual(beforeCarry.window_map?.edges);
  expect(afterCarry.window_map?.problems).toEqual([]);

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await runs.focus();
  const edit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  const batches = stepInspector.getByRole('textbox', { name: 'Batches', exact: true });
  await batches.fill('4');
  await batches.press('Enter');
  await page.locator('[data-slot="secondary"]').getByRole('tabpanel').focus();
  const acknowledged = await edit;
  expect(acknowledged.ok(), await acknowledged.text()).toBe(true);
  snapshot = await acknowledged.json() as StudioProjectSnapshot;
  expect(snapshot.project.steps?.[childId]?.deltas).toHaveLength(2);
  expect(snapshot.project.steps?.[childId]?.deltas?.[1]?.patches).toEqual([
    expect.objectContaining({ path: carryoverPath }),
    expect.objectContaining({ path: 'training_config.n_batches', value: 4 }),
  ]);
  await page.locator('[data-slot="secondary"]').getByRole('button', { name: 'Close Slot', exact: true }).click();
  const successors = await run(childId);
  const compiled = await plan();
  expect(compiled.window_map?.problems).toEqual([]);
  const successorRows = compiled.rows.filter(({ key }) => key.step === childId);
  expect(successorRows.map(({ key }) => key.row)).toEqual(rowKeys);
  const jobFor = (items: StudioJobSubmission['items'], { variant, step, row }: StudioSavedTrainRuns['rows'][number]['key']) =>
    items.find((item) => item.run_key === JSON.stringify([variant, step, row]))?.job_id;
  for (const row of successorRows) {
    const producer = compiled.rows.find(({ key }) => key.step === 'train' && key.row === row.key.row)!;
    expect(compiled.window_map?.edges).toContainEqual({ from_key: producer.key, to_key: row.key });
    expect(row.attempts).toEqual([expect.objectContaining({
      job_id: jobFor(successors, row.key),
      progress: { completed_updates: 6, total_updates: 6 },
      checkpoint_source: {
        job_id: jobFor(prefix, producer.key), request_id: producer.request_id, completed_updates: 2,
      },
    })]);
  }
  const unselected = compiled.rows.find(({ key }) => key.step === 'train' && !rowKeys.includes(key.row))!;
  expect(unselected.attempts).toEqual([]);

  await mapStep(page, 'train').click();
  await chooseSweepValue(source, 'Learning rate', '0.001');
  await chooseSweepValue(source, 'Learning rate', '0.003', true);
  await expect(source.locator('.studio-map-group-box')).toHaveCount(2);
  await expect(source.locator('.studio-map-run-mark[data-boxed]')).toHaveCount(0);
  await captureStudioWindow(page, info, 'visual-map-group');
  await page.keyboard.press('Tab');
  await mark.focus();
  await expect(mark).not.toHaveCSS('box-shadow', 'none');
  await mark.press('Enter');
  await expect(mark).toHaveAttribute('data-boxed', '');
  await expect(source.locator('.studio-map-group-box')).toHaveCount(0);
  await page.keyboard.press('Tab');
  await firstRow.getByRole('button').focus();
  await expect(firstRow).toHaveCSS('outline-style', 'solid');
  await captureStudioWindow(page, info, '01-map-runs');

  await mapStep(page, 'train').click();
  await table.getByRole('button', { name: runName(shownRunLabel(selected[0])) }).click();
  const explicit = submitted(page, filesPath);
  await palette(page, 'newStep', 'New Step');
  await chooseTrainSuccessor(page);
  const rowSaved = await explicit;
  expect(rowSaved.status(), await rowSaved.text()).toBe(201);
  ({ snapshot } = await rowSaved.json() as StudioProjectDocument);
  const rowChild = 'train-3';
  expect(snapshot.project.steps?.[rowChild]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: {
      step: 'train', checkpoint: 'last',
      where: { clauses: [{ 'learning-rate': [{
        id: selected[0].coordinates['learning-rate'].value_id, label: String(selected[0].coordinates['learning-rate'].value),
      }] }] },
    } },
  });
  const reopened = await plan();
  expect(reopened.rows.filter(({ key }) => key.step === childId)).toEqual(successorRows);
  expect(reopened.rows.filter(({ key }) => key.step === rowChild).map(({ key }) => key.row)).toEqual(rowKeys.slice(0, 1));

  await page.reload();
  await mapStep(page, childId).click();
  await expect(table.getByText('Done', { exact: true })).toHaveCount(2);
  const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
  const loss = results.getByRole('img', { name: 'Metric train_loss', exact: true });
  const evidence = await fixture(
    'observe-swept', studio.home, id, ...successorRows.map(({ attempts }) => attempts[0].job_id),
  );
  for (const [index, row] of successorRows.entries()) {
    const attempt = row.attempts[0];
    const producer = attempt.checkpoint_source!;
    expect(evidence[attempt.job_id]).toMatchObject({
      producer: producer.job_id, receipt: attempt.successful_receipt, completed_updates: 6,
      source_checkpoint: {
        completed_updates: 2, transaction_id: expect.any(String), manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      },
      schedule_origin: 0,
      initialization: {
        mode: 'continue_from', source_completed_updates: 2, state: 'preserve', rng: 'carry', schedule: 'continue',
      },
    });
    const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
    const selectedRow = table.getByRole('button', { name: runName(shownRunLabel(selected[index])) });
    await selectedRow.click();
    await expect(selectedRow).toHaveAttribute('aria-pressed', 'true');
    await expect(inspector.getByRole('button', { name: /^Select Made From Train / })).toBeVisible();
    await expect(table.getByRole('progressbar')).toHaveCount(0);
    if (index === 0) {
      await openResultsSlot(page);
      await expect(drawer).toBeHidden();
      await expect(inspector).toBeHidden();
      await expect(page.getByRole('button', { name: 'Show Drawer', exact: true })).toBeDisabled();
      await runs.focus();
      await expect(drawer).toBeHidden();
      await expect(inspector).toBeVisible();
      await page.getByRole('tabpanel', { name: 'Results', exact: true }).focus();
      await expect(drawer).toBeHidden();
      await expect(inspector).toBeHidden();
    }
    await expect(loss.locator('polyline')).toHaveCount(1);
    await expect(loss.locator('polyline')).toHaveCSS('stroke', await loss.evaluate((node) => getComputedStyle(node).color));
    await expect(results.getByRole('list', { name: 'Train steps' }).getByRole('listitem')).toHaveText([
      `${await mapName(page, 'train')} 1–2`, `${await mapName(page, childId)} 3–6`,
    ]);
    await expect(loss.locator(`line[data-attempt="${producer.job_id}"] title`).last()).toHaveText(/Updates \d+–2 .*Local updates \d+–2/);
    await expect(loss.locator(`line[data-attempt="${attempt.job_id}"] title`).last()).toHaveText(/Updates \d+–6 .*Local updates \d+–4/);
    await expect(results.getByRole('region', { name: 'Metric train_loss', exact: true }).locator('[aria-description="Last value at update 6"]')).toHaveText(
      formatMetric(evidence[attempt.job_id].last_loss),
    );
  }
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await page.locator('[data-slot="secondary"]').getByRole('button', { name: 'Close Slot', exact: true }).click();
  await rail.getByRole('tab', { name: 'Results', exact: true }).click();
  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  const desktopLoss = page.locator('[data-slot="primary"]').getByRole('img', { name: 'Metric train_loss', exact: true });
  await expect(desktopLoss.locator('text').filter({ hasText: /^6$/ })).toBeVisible();
  const desktopWidth = await desktopLoss.evaluate((svg: SVGSVGElement) => svg.viewBox.baseVal.width);
  expect(desktopWidth).toBeGreaterThan(700);
  await captureStudioWindow(page, info, '03-results');
  await rail.getByRole('tab', { name: 'Runs', exact: true }).click();
  await openResultsSlot(page);
  const chartWidth = () => loss.evaluate((svg: SVGSVGElement) => svg.viewBox.baseVal.width);
  const bottom = page.locator('.studio-window-bottom');
  const row = await bottom.evaluate(({ clientHeight }) => clientHeight);
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.getByRole('complementary', { name: 'Details Drawer' })).toBeHidden();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toBeHidden();
  await expect(page.locator('.studio-window-slot__strip')).toHaveCount(0);
  await expect.poll(chartWidth).toBeLessThan(desktopWidth);
  await expect(loss.locator('polyline')).toHaveCount(1);
  // A narrow bottom bar wraps its transport onto a second row, and no further.
  await expect.poll(() => bottom.evaluate(({ clientHeight }) => clientHeight)).toBeLessThanOrEqual(2 * row);
  await captureStudioWindow(page, info, '04-narrow-results');
  const splitWidth = await chartWidth();
  await page.setViewportSize({ width: 520, height: 860 });
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  await expect.poll(chartWidth).toBeGreaterThan(splitWidth);
  await captureStudioWindow(page, info, '05-folded-results');
  console.info('Mounted stateful Train pairing', JSON.stringify({
    project: id, revision: snapshot.revision, selection: selected, rows: successorRows, evidence,
    excluded: unselected.key, explicit: snapshot.project.steps?.[rowChild]?.inputs,
  }));
});
