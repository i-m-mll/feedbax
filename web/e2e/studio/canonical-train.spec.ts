import { expect, test, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type {
  StudioJobSnapshot,
  StudioJobSubmission,
  StudioProjectDocument,
  StudioProjectSnapshot,
  StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
async function fixture(...arguments_: string[]) {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', '-m', 'web.e2e.studio.fixtures.canonical_train', ...arguments_,
  ], { cwd: repo, timeout: 12_000 });
  return JSON.parse(stdout);
}

function submitted(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('Mounted saved Train runs the selected canonical row through Jobs', async ({ page, request }, info) => {
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
  const step = page.getByRole('combobox', { name: 'Train step' });
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await step.selectOption('train');
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
  await step.selectOption('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible();
  const reopened = await request.get(`${path}?revision=${snapshot.revision}`);
  const current = await reopened.json() as StudioSavedTrainRuns;
  const finished = current.rows.find(({ key }) => key.step === 'train')!;
  expect(finished.request_id).toBe(original.request_id);
  expect(finished.attempts).toEqual([expect.objectContaining({
    job_id: job, status: 'succeeded', progress: { completed_updates: 2, total_updates: 2 },
    successful_receipt: expect.objectContaining({ manifest_sha256: expect.any(String) }),
  })]);

  await step.selectOption('successor');
  await table.getByRole('button').first().click();
  const chained = submitted(page, path);
  await runButton.click();
  const refusal = await chained;
  expect(refusal.status(), await refusal.text()).toBe(202);
  const rejected = await refusal.json() as StudioJobSubmission;
  expect(rejected.items).toEqual([expect.objectContaining({
    run_key: '["default","successor","default"]', disposition: 'rejected', job_id: null,
    reason: '/inputs: canonical Studio Train admission requires independent requests',
  })]);
  await expect(page.getByText(rejected.items[0].reason!, { exact: true })).toBeVisible();
  const evidence = await fixture('observe', String(info.config.metadata.home), id, snapshot.revision, job);
  expect(evidence).toMatchObject({
    request: original.request_id, attempt: job, delivery: 1,
    completion_updates: 2, joined_status: 'succeeded',
  });

  await step.selectOption('train');
  await table.getByRole('button').first().click();
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
  expect(jobs.jobs.map(({ job_id }) => job_id)).toEqual([job]);
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
  const step = page.getByRole('combobox', { name: 'Train step' });
  const runButton = page.getByRole('button', { name: 'Run', exact: true });
  async function select(name: string) {
    await step.selectOption(name);
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
  async function editNumber(label: string, value: number) {
    const field = page.getByRole('spinbutton', { name: `Edit ${label}` });
    const previous = snapshot.revision;
    const refreshed = page.waitForResponse((response) => {
      const url = new URL(response.url());
      return response.request().method() === 'GET' && url.pathname === path &&
        url.searchParams.get('revision') !== previous && response.ok();
    });
    await field.fill(String(value));
    await field.press('Enter');
    const plan = await (await refreshed).json() as StudioSavedTrainRuns;
    await rows();
    expect(snapshot.revision).toBe(plan.revision);
    await expect(field).toHaveValue(String(value));
    await expect(field).toBeEnabled();
  }

  await page.goto(`/?project=${encodeURIComponent(id)}`);
  const firstJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const first = (await trainRow()).result!;
  expect(first.job_id).toBe(firstJob);
  await editNumber('Seed', 7);
  await expect(table.getByText('Out of date', { exact: true })).toBeVisible();
  await expect(table.getByText(/Definition changed at .*seed/)).toBeVisible();
  const stale = await trainRow();
  expect(stale).toMatchObject({ status: 'stale', result: null, previous: first });
  expect(stale.request_id).not.toBe(first.request_id);

  await runSelected('sibling');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  expect(await trainRow()).toMatchObject({ status: 'stale', result: null, previous: first, attempts: [] });

  await select('train');
  await editNumber('N Batches', 10000);
  const stoppedJob = await runSelected('train');
  const stop = await request.post('/api/jobs/stop', { data: { job_ids: [stoppedJob], mode: 'force' } });
  expect(stop.status(), await stop.text()).toBe(202);
  await expect.poll(async () => (await trainRow()).status).toBe('stopped');
  expect(await trainRow()).toMatchObject({ result: null, previous: first });

  await editNumber('N Batches', 2);
  const changedJob = await runSelected('train');
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 14_000 });
  const changed = await trainRow();
  expect(changed.result!.job_id).toBe(changedJob);
  expect(changed.previous).toEqual(first);
  await page.reload();
  await select('train');
  await page.getByText('Compare receipts', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Current result' })).toContainText(changed.result!.receipt.manifest_sha256);
  await expect(page.getByRole('region', { name: 'Previous result' })).toContainText(first.receipt.manifest_sha256);
  const evidence = await fixture('observe-history', String(info.config.metadata.home), id, snapshot.revision);
  expect(evidence).toMatchObject({ current: changed.result, previous: first });
  console.info('Mounted saved Train history', JSON.stringify(evidence));
});
