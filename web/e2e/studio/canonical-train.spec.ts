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
