import { expect, test, type Page } from '@playwright/test';
import { authorLearningRateSweep, captureStudioWindow, chooseTrainSuccessor, fixture, mapStep, openResultsSlot, seedManagedProject } from './fixtures/canonicalTrain';
import { formatTrainingLoss } from '../../src/components/activity/studioPlot';
import type {
  StudioActivityJob,
  StudioJobSnapshot,
  StudioJobSubmission,
  StudioProjectDocument,
  StudioProjectSnapshot,
  StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';

test.use({ trace: 'off', launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] } });

function submitted(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('standalone Activity opens an existing run and hydrates its retained history', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const admission = await request.post(`${projectPath}/train-runs`, { data: {
    token: 'background-activity', revision: snapshot.revision,
    run_keys: [{ variant: 'default', step: 'train', row: 'default' }],
  } });
  expect(admission.status(), await admission.text()).toBe(202);
  const { items: [attempt] } = await admission.json() as StudioJobSubmission;
  await page.goto('/?activity=1');
  await expect(page.getByRole('heading', { name: 'Activity', exact: true })).toBeVisible();
  const row = page.getByRole('list', { name: 'Runs', exact: true }).getByRole('listitem')
    .filter({ hasText: snapshot.project_id });
  await expect(row.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await row.getByRole('button', { name: new RegExp(snapshot.project_id) }).click();
  const inspector = page.getByRole('region', { name: 'Run Inspector' });
  await expect(inspector.getByRole('region', { name: 'Log', exact: true })).toBeVisible();
  expect((await request.get('/api/jobs').then((response) => response.json()) as StudioJobSnapshot)
    .jobs.some(({ job_id, seen }) => job_id === attempt.job_id && seen)).toBe(true);
});

test('Mounted saved Train runs a canonical successor through Jobs', async ({ page, request }, info) => {
  test.setTimeout(48_000);
  const created = await request.post('/api/projects');
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
  const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
  const loss = results.getByRole('img', { name: 'Loss', exact: true });
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
    'observe', String(info.config.metadata.home), id, snapshot.revision, job, stoppedJob, successorJob,
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
  await expect(results.getByRole('list', { name: 'Train steps' }).getByRole('listitem')).toHaveText([
    'train 1–2', 'successor 3–1202',
  ]);
  await expect(loss.locator(`line[data-attempt="${job}"] title`).last()).toHaveText(
    /Updates [12]–2 .*train · Local updates [12]–2/,
  );
  await expect(loss.locator(`line[data-attempt="${stoppedJob}"] title`).last()).toContainText(
    `–${evidence.successor.resumed_from}`,
  );
  await expect(loss.locator(`line[data-attempt="${successorJob}"] title`).first()).toContainText(
    `Local updates ${evidence.successor.resumed_from + 1}–`,
  );
  await expect(loss.locator(`line[data-attempt="${successorJob}"] title`).last()).toHaveText(
    /Updates \d+–1202 .*successor · Local updates \d+–1200/,
  );
  const { last } = evidence.successor.terminal_loss.metrics.train_loss;
  await expect(results.locator('[aria-description="Last loss at update 1202"]')).toHaveText(
    formatTrainingLoss(last),
  );
  await page.locator('summary').filter({ hasText: /^Current Result$/ }).click();
  const successorResult = page.getByRole('region', { name: 'Current Result' });
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
      patches: [{ op: 'add', path: '/base/inline/method_payload/payload/seed', value: 7 }],
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

test('Saved Train compares authenticated receipts after a partial run and stopped edit', async ({ page, request }, info) => {
  test.setTimeout(65_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const created = await request.post('/api/projects');
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
  const loss = results.getByRole('img', { name: 'Loss', exact: true });
  const observed = () => fixture('observe-history', String(info.config.metadata.home), id, snapshot.revision);
  type LossSample = { update: number; value: number };
  async function expectLoss(jobId: string, expected: LossSample[]) {
    const series = loss.locator(`[data-result-attempt="${jobId}"]`);
    await expect(series.locator('polyline')).toBeAttached();
    await expect.poll(() => series.locator('line[data-update][data-loss]').evaluateAll((points) =>
      points.map((point) => ({ update: Number(point.getAttribute('data-update')), value: Number(point.getAttribute('data-loss')) })),
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
    .getByText(/Definition changed at .*seed/)).toBeVisible();
  const stale = await trainRow();
  expect(stale).toMatchObject({ status: 'stale', result: null, previous: first });
  expect(stale.request_id).not.toBe(first.request_id);

  await runSelected('sibling');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  expect(await trainRow()).toMatchObject({ status: 'stale', result: null, previous: first, attempts: [] });

  await select('train');
  await editNumber('N Batches', 10000, '2');
  const stoppedJob = await runSelected('train');
  const stop = await request.post('/api/jobs/stop', { data: { job_ids: [stoppedJob], mode: 'force' } });
  expect(stop.status(), await stop.text()).toBe(202);
  await expect.poll(async () => (await trainRow()).status).toBe('stopped');
  expect(await trainRow()).toMatchObject({ result: null, previous: first });

  await openResultsSlot(page);
  await results.locator('summary').getByText('Previous Result', { exact: true }).click();
  const stopped = await observed();
  expect(stopped).toMatchObject({ current: null, previous: first });
  await expectLoss(firstJob, stopped.loss.previous);
  await expect(loss.locator(`[data-result-attempt="${stoppedJob}"]`)).toHaveCount(0);
  await expect(loss.locator('polyline')).toHaveCount(1);
  await results.getByRole('button', { name: 'Close Slot', exact: true }).click();

  await editNumber('N Batches', 2, '10000');
  const changedJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const changed = await trainRow();
  expect(changed.result!.job_id).toBe(changedJob);
  expect(changed.previous).toEqual(first);
  expect(changed.change).toMatchObject({ path: expect.stringMatching(/seed$/), before: null, after: 7 });
  const evidence = await observed();
  expect(evidence).toMatchObject({ current: changed.result, previous: first });
  async function compare() {
    await openResultsSlot(page);
    await results.locator('summary').getByText('Compare Results', { exact: true }).click();
    await expect(results.locator('details[open]')).toContainText(changed.change!.path);
    await expect(results.locator('details[open]')).toContainText(/null\s*→\s*7/);
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
  const current = page.getByRole('region', { name: 'Current Result' });
  const previous = page.getByRole('region', { name: 'Previous Result' });
  await expect(current).not.toContainText(changed.result!.receipt.manifest_sha256);
  await expect(previous).not.toContainText(first.receipt.manifest_sha256);
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  for (const [section, result] of [[current, changed.result!], [previous, first]] as const) {
    await section.getByRole('button', { name: 'Copy Details', exact: true }).click();
    await expect.poll(() => page.evaluate(async () => JSON.parse(await navigator.clipboard.readText()))).toEqual(result);
  }
  console.info('Mounted saved Train history', JSON.stringify(evidence));
});

test('Mounted field sweep continues exactly two Train rows with state', async ({ page, request }, info) => {
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
  const source = map.getByRole('article', { name: 'Train train', exact: true });
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const table = runs.getByRole('table', { name: 'Runs' });
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  await expect(table).toBeInViewport();
  await expect(runs.getByRole('region', { name: 'Definition' })).toHaveCount(0);
  await expect(drawer.getByRole('region', { name: 'Definition' })).toBeVisible();
  await expect(source).toHaveCSS('outline-width', '2px');
  await expect(mapStep(page, 'train')).toHaveCSS('background-color', await source.evaluate((node) => getComputedStyle(node).color));
  await authorLearningRateSweep(drawer.getByRole('region', { name: 'Learning Rate field', exact: true }));

  const sweep = source.getByRole('button', { name: 'Sweep learning-rate', exact: true });
  await expect(sweep).toHaveText('learning-rate 3');
  const saved = await request.get(projectPath);
  expect(saved.ok(), await saved.text()).toBe(true);
  snapshot = await saved.json() as StudioProjectSnapshot;
  const authored = (await plan()).window_map!;
  expect(authored.problems).toEqual([]);
  expect(authored.rows).toHaveLength(3);
  const selected = authored.rows.filter(({ coordinates }) => coordinates['learning-rate'].label !== 'Medium');
  const valueIds = selected.map(({ coordinates }) => coordinates['learning-rate'].value_id);
  const rowKeys = selected.map(({ key }) => key.row);
  expect(selected.map(({ coordinates }) => coordinates['learning-rate'].label)).toEqual(['Slow', 'Fast']);
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

  await sweep.click();
  const valueButton = (name: string) => source.getByRole('listitem').getByRole('button', { name: `Select learning-rate ${name}`, exact: true });
  await valueButton('Slow · 0.001').click();
  await valueButton('Fast · 0.003').click({ modifiers: ['Shift'] });
  await expect(valueButton('Slow · 0.001')).toHaveAttribute('aria-pressed', 'true');
  await expect(valueButton('Fast · 0.003')).toHaveAttribute('aria-pressed', 'true');
  await expect(valueButton('Medium · 0.002')).toHaveAttribute('aria-pressed', 'false');
  await expect(source).toContainText('2 of 3');
  await sweep.click();
  await page.keyboard.press('Tab');
  const mark = source.locator('.studio-map-run-mark').first();
  await mark.focus();
  await expect(mark).not.toHaveCSS('box-shadow', 'none');
  const firstRow = table.locator('tbody tr[data-index]').first();
  await firstRow.getByRole('button').focus();
  await expect(firstRow).toHaveCSS('outline-style', 'solid');
  const prefix = await run('train');

  const create = submitted(page, filesPath);
  await source.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseTrainSuccessor(page);
  const accepted = await create;
  expect(accepted.status(), await accepted.text()).toBe(201);
  ({ snapshot } = await accepted.json() as StudioProjectDocument);
  const childId = 'train-continuation';
  expect(snapshot.project.steps?.[childId]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: { axes: { 'learning-rate': valueIds } } } },
    deltas: [{ patches: [{ value: 'initialize_from' }] }],
  });
  const child = map.getByRole('article', { name: `Train ${childId}`, exact: true });
  await expect(child.getByRole('button', { name: `Select ${childId}`, exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(child).toContainText('2 of 3');
  await expect(drawer.getByRole('combobox', { name: 'Edit Training method' })).toBeFocused();

  const carryover = drawer.getByRole('combobox', { name: 'Edit Checkpoint carryover', exact: true });
  const carryoverPath = 'method_payload.payload.checkpoint_initialization';
  const methodTarget = 'method_payload.payload.target_total';
  const checkpointInput = snapshot.project.steps?.[childId]?.inputs?.checkpoint;
  const beforeCarry = await plan();
  await expect(carryover.locator('option:checked')).toHaveText('Initialize model only');
  const carryEdit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  await carryover.selectOption({ label: 'Carry training state' });
  const carried = await carryEdit;
  expect(carried.ok(), await carried.text()).toBe(true);
  expect(carried.request().postDataJSON()).toMatchObject({ address: { kind: 'step', id: childId } });
  snapshot = await carried.json() as StudioProjectSnapshot;
  expect(snapshot.project.steps?.[childId]?.inputs?.checkpoint).toEqual(checkpointInput);
  expect(snapshot.project.steps?.[childId]?.deltas?.[1]?.patches).toEqual([
    { op: 'replace', path: carryoverPath, value: {
      mode: 'continue_from', progress: { source_completed_batches: 2, additional_batches: 2 },
      optimizer: 'preserve', schedule: 'continue', rng: 'carry', cursor: 'carry',
    } },
    { op: 'replace', path: methodTarget, value: 4 },
  ]);
  await expect(carryover.locator('option:checked')).toHaveText('Carry training state');
  const afterCarry = await plan();
  expect(afterCarry.rows.filter(({ key }) => key.step === 'train'))
    .toEqual(beforeCarry.rows.filter(({ key }) => key.step === 'train'));
  expect(afterCarry.window_map?.edges).toEqual(beforeCarry.window_map?.edges);
  expect(afterCarry.window_map?.problems).toEqual([]);

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await runs.focus();
  const edit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  const batches = drawer.getByRole('textbox', { name: 'N Batches', exact: true });
  await batches.fill('4');
  await batches.press('Enter');
  await page.getByRole('tabpanel', { name: 'Scene', exact: true }).focus();
  const acknowledged = await edit;
  expect(acknowledged.ok(), await acknowledged.text()).toBe(true);
  snapshot = await acknowledged.json() as StudioProjectSnapshot;
  expect(snapshot.project.steps?.[childId]?.deltas?.[2]?.patches).toEqual([
    expect.objectContaining({ path: 'training_config.n_batches', value: 4 }),
    expect.objectContaining({ path: methodTarget, value: 6 }),
    expect.objectContaining({ path: `${carryoverPath}.progress.additional_batches`, value: 4 }),
  ]);
  await page.getByRole('button', { name: 'Close Slot', exact: true }).click();
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

  await source.getByRole('button', { name: 'Select train', exact: true }).click();
  await sweep.click();
  await valueButton('Slow · 0.001').click();
  await valueButton('Fast · 0.003').click({ modifiers: ['Shift'] });
  await sweep.click();
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

  await source.getByRole('button', { name: 'Select train', exact: true }).click();
  await table.getByRole('row').filter({ hasText: selected[0].label }).getByRole('button').click();
  const explicit = submitted(page, filesPath);
  await runs.getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseTrainSuccessor(page);
  const rowSaved = await explicit;
  expect(rowSaved.status(), await rowSaved.text()).toBe(201);
  ({ snapshot } = await rowSaved.json() as StudioProjectDocument);
  const rowChild = 'train-continuation-2';
  expect(snapshot.project.steps?.[rowChild]).toMatchObject({
    base: { step: 'train' },
    inputs: { checkpoint: { step: 'train', checkpoint: 'last', where: { rows: [rowKeys[0]] } } },
  });
  const reopened = await plan();
  expect(reopened.rows.filter(({ key }) => key.step === childId)).toEqual(successorRows);
  expect(reopened.rows.filter(({ key }) => key.step === rowChild).map(({ key }) => key.row)).toEqual(rowKeys.slice(0, 1));

  await page.reload();
  await mapStep(page, childId).click();
  await expect(table.getByText('Done', { exact: true })).toHaveCount(2);
  const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
  const loss = results.getByRole('img', { name: 'Loss', exact: true });
  const evidence = await fixture(
    'observe-swept', String(info.config.metadata.home), id, ...successorRows.map(({ attempts }) => attempts[0].job_id),
  );
  for (const [index, row] of successorRows.entries()) {
    const attempt = row.attempts[0];
    const producer = attempt.checkpoint_source!;
    expect(evidence[attempt.job_id]).toMatchObject({
      producer: producer.job_id, receipt: attempt.successful_receipt, completed_updates: 6,
      source_checkpoint: {
        completed_updates: 2, transaction_id: expect.any(String), manifest_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      },
      schedule_origin_step: 0,
      initialization: {
        mode: 'continue_from', optimizer_policy: 'preserve', rng_policy: 'carry', cursor_policy: 'carry',
        learning_rate_schedule_policy: 'continue',
        producer_content_binding: { root_kind: 'checkpoint-custody:batch_counter,model,optimizer,prng' },
      },
    });
    const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
    const selectedRow = table.getByRole('row').filter({ hasText: selected[index].label }).getByRole('button');
    await selectedRow.click();
    await expect(selectedRow).toHaveAttribute('aria-pressed', 'true');
    await expect(runs.getByRole('region', { name: 'Made From' })).toContainText('train');
    await expect(table.getByRole('progressbar')).toHaveCount(0);
    if (index === 0) {
      await openResultsSlot(page);
      await expect(drawer).toBeHidden();
      await expect(inspector).toBeHidden();
      await expect(page.getByRole('button', { name: 'Show Drawer', exact: true })).toBeDisabled();
      await runs.focus();
      await expect(drawer).toBeVisible();
      await expect(inspector).toBeVisible();
      await page.getByRole('tabpanel', { name: 'Results', exact: true }).focus();
      await expect(drawer).toBeHidden();
      await expect(inspector).toBeHidden();
    }
    await expect(loss.locator('polyline')).toHaveCount(1);
    await expect(loss.locator('polyline')).toHaveCSS('stroke', await loss.evaluate((node) => getComputedStyle(node).color));
    await expect(results.getByRole('list', { name: 'Train steps' }).getByRole('listitem')).toHaveText([
      'train 1–2', `${childId} 3–6`,
    ]);
    await expect(loss.locator(`line[data-attempt="${producer.job_id}"] title`).last()).toHaveText(/Updates \d+–2 .*Local updates \d+–2/);
    await expect(loss.locator(`line[data-attempt="${attempt.job_id}"] title`).last()).toHaveText(/Updates \d+–6 .*Local updates \d+–4/);
    await expect(results.locator('[aria-description="Last loss at update 6"]')).toHaveText(
      formatTrainingLoss(evidence[attempt.job_id].last_loss),
    );
  }
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await page.getByRole('button', { name: 'Close Slot', exact: true }).click();
  await rail.getByRole('tab', { name: 'Results', exact: true }).click();
  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  const desktopLoss = page.locator('[data-slot="primary"]').getByRole('img', { name: 'Loss', exact: true });
  await expect(desktopLoss.locator('text').filter({ hasText: /^6$/ })).toBeVisible();
  const desktopWidth = await desktopLoss.evaluate((svg: SVGSVGElement) => svg.viewBox.baseVal.width);
  expect(desktopWidth).toBeGreaterThan(700);
  await captureStudioWindow(page, info, '03-results');
  await rail.getByRole('tab', { name: 'Runs', exact: true }).click();
  await openResultsSlot(page);
  const chartWidth = () => loss.evaluate((svg: SVGSVGElement) => svg.viewBox.baseVal.width);
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.getByRole('complementary', { name: 'Details Drawer' })).toBeHidden();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toBeHidden();
  await expect(page.locator('.studio-window-slot__strip')).toHaveCount(0);
  await expect.poll(chartWidth).toBeLessThan(desktopWidth);
  await expect(loss.locator('polyline')).toHaveCount(1);
  const bottom = page.locator('.studio-window-bottom');
  await expect.poll(() => bottom.evaluate(({ clientHeight }) => clientHeight)).toBeLessThanOrEqual(40);
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
