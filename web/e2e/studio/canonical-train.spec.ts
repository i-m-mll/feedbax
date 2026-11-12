import { expect, test, type Page } from '@playwright/test';
import { authorLearningRateSweep, enter, fixture, mapStep, openResultsSlot, seedManagedProject } from './fixtures/canonicalTrain';
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

test('Mounted saved Train runs a canonical successor through Jobs', async ({ page, request }, info) => {
  test.setTimeout(48_000);
  const created = await request.post('/api/projects');
  expect(created.status()).toBe(201);
  let { snapshot } = await created.json() as StudioProjectDocument;
  const id = snapshot.project_id;
  const projectPath = `/api/projects/${id}`;
  for (const file of await fixture('files')) {
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
  await expect(loss.locator('line title').filter({ hasText: stoppedJob }).first()).toBeAttached({ timeout: 14_000 });
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
  await expect(loss.locator('polyline')).toHaveCount(1);
  await expect(results.getByText('Cumulative updates')).toBeVisible();
  await expect(results.getByRole('list', { name: 'Train stages' }).getByRole('listitem')).toHaveText([
    'train 1–2', 'successor 3–1202',
  ]);
  await expect(loss.locator('line title').filter({ hasText: job }).last()).toHaveText(
    /Updates [12]–2 .*train · Local updates [12]–2/,
  );
  await expect(loss.locator('line title').filter({ hasText: stoppedJob }).last()).toContainText(
    `–${evidence.successor.resumed_from} · Attempt ${stoppedJob}`,
  );
  await expect(loss.locator('line title').filter({ hasText: successorJob }).first()).toContainText(
    `Local updates ${evidence.successor.resumed_from + 1}–`,
  );
  await expect(loss.locator('line title').filter({ hasText: successorJob }).last()).toHaveText(
    /Updates \d+–1202 .*successor · Local updates \d+–1200/,
  );
  const { last } = evidence.successor.terminal_loss.metrics.train_loss;
  await expect(results.getByTitle('Last loss at update 1202')).toHaveText(
    new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(last),
  );
  await page.locator('summary').filter({ hasText: /^Current result$/ }).click();
  const successorResult = page.getByRole('region', { name: 'Current result' });
  await expect(successorResult).toContainText(evidence.successor.request);
  await expect(successorResult).toContainText(evidence.successor.receipt.manifest_sha256);

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
  test.setTimeout(50_000);
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
  async function editNumber(label: string, value: number, before: string) {
    await mapStep(page, 'train').click();
    await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
    const field = page.getByRole('spinbutton', { name: `Edit ${label}` });
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
  await expect(table.getByText(/Definition changed at .*seed/)).toBeVisible();
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

  await editNumber('N Batches', 2, '10000');
  const changedJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const changed = await trainRow();
  expect(changed.result!.job_id).toBe(changedJob);
  expect(changed.previous).toEqual(first);
  await page.reload();
  await select('train');
  await openResultsSlot(page);
  await page.getByText('Compare receipts', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Current result' })).toContainText(changed.result!.receipt.manifest_sha256);
  await expect(page.getByRole('region', { name: 'Previous result' })).toContainText(first.receipt.manifest_sha256);
  const evidence = await fixture('observe-history', String(info.config.metadata.home), id, snapshot.revision);
  expect(evidence).toMatchObject({ current: changed.result, previous: first });
  console.info('Mounted saved Train history', JSON.stringify(evidence));
});

test('Mounted field sweep continues exactly two Train rows with state', async ({ page, request }, info) => {
  test.setTimeout(47_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  let { snapshot, projectPath } = await seedManagedProject(request, (await fixture('files')).slice(0, 3));
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
  const runs = page.getByRole('region', { name: 'Runs', exact: true });
  const table = runs.getByRole('table', { name: 'Runs' });
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  await authorLearningRateSweep(runs.getByRole('group', { name: 'Learning Rate', exact: true }));

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
  const prefix = await run('train');

  const create = submitted(page, filesPath);
  await source.getByRole('button', { name: 'New Step train', exact: true }).click();
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
  await expect(runs.getByRole('combobox', { name: 'Edit Model' })).toBeFocused();

  // Exercise the authored typed policy through the saved project API. New Step defaults to model initialization.
  const methodTarget = 'method_payload.payload.target_total';
  const changed = await request.patch(projectPath, { data: {
    revision: snapshot.revision, address: { kind: 'step', id: childId },
    patches: [
      { op: 'replace', path: '/deltas/0/patches/0/value', value: {
        mode: 'continue_from', progress: { source_completed_batches: 2, additional_batches: 2 },
        optimizer: 'preserve', schedule: 'continue', rng: 'carry', cursor: 'carry',
      } },
      { op: 'add', path: '/deltas/0/patches/-', value: { op: 'replace', path: methodTarget, value: 4 } },
      { op: 'add', path: '/deltas/0/acknowledges_ancestor_paths', value: [methodTarget] },
    ],
  } });
  expect(changed.ok(), await changed.text()).toBe(true);
  snapshot = await changed.json() as StudioProjectSnapshot;
  await page.reload();
  await mapStep(page, childId).click();

  const edit = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  await enter(runs.getByRole('spinbutton', { name: 'Edit N Batches' }), '4');
  const acknowledged = await edit;
  expect(acknowledged.ok(), await acknowledged.text()).toBe(true);
  snapshot = await acknowledged.json() as StudioProjectSnapshot;
  expect(snapshot.project.steps?.[childId]?.deltas?.[1]?.patches).toEqual([
    expect.objectContaining({ path: 'training_config.n_batches', value: 4 }),
    expect.objectContaining({ path: methodTarget, value: 6 }),
    expect.objectContaining({ path: 'method_payload.payload.checkpoint_initialization.progress.additional_batches', value: 4 }),
  ]);
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
  await table.getByRole('row').filter({ hasText: selected[0].label }).getByRole('button').click();
  const explicit = submitted(page, filesPath);
  await runs.getByRole('button', { name: 'New Step', exact: true }).click();
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
      initialization: { mode: 'continue_from', optimizer_policy: 'preserve', rng_policy: 'carry', cursor_policy: 'carry' },
    });
    await table.getByRole('row').filter({ hasText: selected[index].label }).getByRole('button').click();
    if (index === 0) await openResultsSlot(page);
    await expect(loss.locator('polyline')).toHaveCount(1);
    await expect(results.getByRole('list', { name: 'Train stages' }).getByRole('listitem')).toHaveText([
      'train 1–2', `${childId} 3–6`,
    ]);
    await expect(loss.locator('line title').filter({ hasText: producer.job_id }).last()).toHaveText(/Updates \d+–2 .*Local updates \d+–2/);
    await expect(loss.locator('line title').filter({ hasText: attempt.job_id }).last()).toHaveText(/Updates \d+–6 .*Local updates \d+–4/);
    await expect(results.getByTitle('Last loss at update 6')).toHaveText(
      new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(evidence[attempt.job_id].last_loss),
    );
  }
  console.info('Mounted stateful Train pairing', JSON.stringify({
    project: id, revision: snapshot.revision, selection: selected, rows: successorRows, evidence,
    excluded: unselected.key, explicit: snapshot.project.steps?.[rowChild]?.inputs,
  }));
});
