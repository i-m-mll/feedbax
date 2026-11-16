import { expect, test } from '@playwright/test';
import { readdir, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { StudioActivityJob, StudioJobSnapshot, StudioJobSubmission, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.skip(process.env.FEEDBAX_STUDIO_E2E_REMOTE_FIXTURE !== '1', 'Requires the disposable machine provider');

test('Machine approval reviews the saved selection and collects the remote result', async ({ page, request }, info) => {
  test.setTimeout(50_000);
  const fixtureRoot = join(String(info.config.metadata.home), 'state', 'studio-e2e-remote');
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const path = `${projectPath}/train-runs`;
  const key = { variant: 'default', step: 'train', row: 'default' };
  const attempts = async () => {
    const response = await request.get(`${path}?revision=${snapshot.revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const plan = await response.json() as StudioSavedTrainRuns;
    return plan.rows.find(({ key: row }) => row.step === key.step)!.attempts;
  };

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  const machine = page.getByRole('combobox', { name: 'Machine' });
  await expect(machine).toHaveValue('local');
  await machine.selectOption('rehearsal');
  const run = page.getByRole('button', { name: 'Run', exact: true });
  await run.click();
  const approval = page.getByRole('dialog', { name: 'Needs approval' });
  await expect(approval.getByText('Cost ceiling', { exact: true })).toBeVisible();
  await expect(approval.getByText('$0.12', { exact: true })).toBeVisible();
  await expect(approval.getByText('Expiry', { exact: true })).toBeVisible();
  await expect(approval.getByText('Results not downloaded are destroyed', { exact: true })).toBeVisible();
  expect(await attempts()).toEqual([]);
  const activity = await request.get('/api/jobs');
  expect((await activity.json() as StudioJobSnapshot).jobs).toEqual([]);
  expect(await readdir(join(fixtureRoot, 'venues'))).toEqual([]);

  const direct = await request.post(path, { data: {
    revision: snapshot.revision, run_keys: [key], target: 'rehearsal', token: 'unapproved',
  } });
  expect(direct.status()).toBe(409);
  await approval.getByRole('button', { name: 'Cancel' }).click();
  await expect(approval).toBeHidden();
  expect(await attempts()).toEqual([]);
  expect(await readdir(join(fixtureRoot, 'venues'))).toEqual([]);

  const downloadGate = join(fixtureRoot, 'download-gate');
  await run.click();
  await writeFile(downloadGate, '');
  const accepted = page.waitForResponse((response) => new URL(response.url()).pathname === `${path}/approve`);
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  let submission: StudioJobSubmission;
  try {
    await approval.getByRole('button', { name: 'Approve' }).click();
    const response = await accepted;
    expect(response.status(), await response.text()).toBe(202);
    submission = await response.json() as StudioJobSubmission;
    expect(submission.target).toMatchObject({ kind: 'session', profile: 'rehearsal', driver: 'local-rehearsal' });
    expect(submission.items).toEqual([expect.objectContaining({ disposition: 'admitted', job_id: expect.any(String) })]);
    await expect(approval).toBeHidden();
    await expect(runs.getByText('Downloading', { exact: true })).toBeVisible({ timeout: 18_000 });
    await expect(runs.getByRole('progressbar', { name: 'Downloading' })).toBeVisible();
  } finally {
    await unlink(downloadGate);
  }
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 20_000 });
  const [attempt] = await attempts();
  expect(attempt).toMatchObject({ status: 'succeeded', job_id: submission.items[0].job_id });
  await page.reload();
  await mapStep(page, 'train').click();
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const card = page.getByRole('region', { name: 'Rehearsal Machine', exact: true });
  await expect(card.getByText('Closes at', { exact: true })).toBeVisible();
  await expect(card.getByText('Estimated cost', { exact: true })).toBeVisible();
  const close = card.getByRole('button', { name: 'Close', exact: true });
  await expect(close).toBeEnabled();
  const closed = page.waitForResponse((response) => new URL(response.url()).pathname ===
    `/api/sessions/${submission.target?.kind === 'session' ? submission.target.session_id : ''}/close`);
  await close.click();
  const confirmation = await closed;
  expect(confirmation.status(), await confirmation.text()).toBe(204);
  await expect(card).toBeHidden({ timeout: 10_000 });
  expect(await attempts()).toEqual([attempt]);
});

test('A set of six remote rows keeps its Jobs states and commands across reloads', async ({ page, request }, info) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture(
    'ordinary-files', '20000', '0.001', '0.0011', '0.0012', '0.0013', '0.0014', '0.0015',
  ));
  const path = `${projectPath}/train-runs`;
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await expect(runs.getByText('Not run', { exact: true })).toHaveCount(6);
  const initial = await request.get(`${path}?revision=${snapshot.revision}`);
  expect(initial.ok(), await initial.text()).toBe(true);
  const rows = (await initial.json() as StudioSavedTrainRuns).rows;
  expect(rows).toHaveLength(6);
  const jobs = async () => {
    const response = await request.get('/api/jobs', { params: { project: snapshot.project_id } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioJobSnapshot).jobs;
  };
  const job = async (id: string) => (await jobs()).find(({ job_id }) => job_id === id);
  const rowName = (activity: StudioActivityJob) => rows.find(({ key }) =>
    JSON.stringify([key.variant, key.step, key.row]) === activity.run_key)!.label;
  const row = (activity: StudioActivityJob) => runs.getByRole('row').filter({
    has: page.getByRole('rowheader', { name: rowName(activity), exact: true }),
  });
  const reopen = async () => {
    await page.reload();
    await mapStep(page, 'train').click();
  };
  const command = (name: string) => page.getByRole('button', { name, exact: true });

  await page.getByRole('combobox', { name: 'Machine' }).selectOption('rehearsal');
  await command('Run').click();
  const approval = page.getByRole('dialog', { name: 'Needs approval' });
  await expect(approval.getByRole('listitem')).toHaveCount(6);
  await expect(approval.getByText('Cost ceiling', { exact: true })).toBeVisible();
  expect(await jobs()).toEqual([]);
  const accepted = page.waitForResponse((response) => new URL(response.url()).pathname === `${path}/approve`);
  await approval.getByRole('button', { name: 'Approve' }).click();
  const response = await accepted;
  expect(response.status(), await response.text()).toBe(202);
  const admitted = await response.json() as StudioJobSubmission;
  expect(admitted.items).toHaveLength(6);
  expect(admitted.items.every(({ disposition }) => disposition === 'admitted')).toBe(true);
  if (admitted.target?.kind !== 'session') throw new Error('Expected the rehearsal machine placement');
  const session = admitted.target.session_id;
  await expect(runs.getByText('Queued', { exact: true })).toHaveCount(2, { timeout: 12_000 });
  await expect(runs.getByText('Waiting for a training slot.', { exact: true })).toHaveCount(2);
  await expect.poll(async () => (await jobs()).filter(({ status }) => status === 'running').length,
    { timeout: 15_000 }).toBe(4);
  await reopen();
  await expect(runs.getByText('Queued', { exact: true })).toHaveCount(2);
  await expect(runs.getByText('Running', { exact: true })).toHaveCount(4);
  let active = (await jobs()).filter(({ status }) => status === 'running');
  const checkpoint = active[0];
  const force = active[1];
  const interrupt = active[2];

  await row(checkpoint).getByRole('button').click();
  const stop = page.waitForResponse((result) => new URL(result.url()).pathname === '/api/jobs/stop');
  await command('Stop').click();
  expect((await stop).status()).toBe(202);
  await expect(row(checkpoint).getByText('Stopped', { exact: true })).toBeVisible({ timeout: 10_000 });
  await reopen();
  await row(checkpoint).getByRole('button').click();
  await expect(row(checkpoint).getByText('Stopped', { exact: true })).toBeVisible();
  expect(await job(checkpoint.job_id)).toMatchObject({ status: 'stopped', can_resume: true });
  const continuation = page.waitForResponse((result) =>
    new URL(result.url()).pathname === `/api/jobs/${checkpoint.job_id}/resume`);
  await command('Resume').click();
  const continued = await continuation;
  expect(continued.status(), await continued.text()).toBe(202);
  expect(await continued.json()).toMatchObject({ continues: checkpoint.job_id, target: { kind: 'session', session_id: session } });

  await row(force).getByRole('button').click();
  const forced = page.waitForResponse((result) => new URL(result.url()).pathname === '/api/jobs/stop');
  await command('Force Stop').click();
  expect((await forced).status()).toBe(202);
  await expect.poll(async () => (await job(force.job_id))?.status).not.toMatch(/^(queued|running|stopping|starting)$/);
  await reopen();
  expect(await job(force.job_id)).toMatchObject({ status: expect.stringMatching(/^(stopped|interrupted)$/) });

  await runProjectFixture({ module: 'web.e2e.studio.fixtures.remote_run_process', home: String(info.config.metadata.home) },
    session, interrupt.job_id);
  await expect(row(interrupt).getByText('Interrupted', { exact: true })).toBeVisible({ timeout: 9_000 });
  await reopen();
  await row(interrupt).getByRole('button').click();
  await expect(row(interrupt).getByText('Interrupted', { exact: true })).toBeVisible();
  await expect(command('Resume')).toBeEnabled();
  expect(await job(interrupt.job_id)).toMatchObject({ status: 'interrupted', can_resume: true });

  active = (await jobs()).filter(({ status }) => /^(queued|running|starting|stopping)$/.test(status));
  if (active.length) {
    const stopped = await request.post('/api/jobs/stop', { data: { job_ids: active.map(({ job_id }) => job_id), mode: 'force' } });
    expect(stopped.status(), await stopped.text()).toBe(202);
    expect(await stopped.json()).toEqual(active.map(({ job_id }) => expect.objectContaining({ job_id, error: null })));
  }
  await expect.poll(async () => {
    const response = await request.get('/api/sessions');
    const machines = await response.json() as StudioJobSnapshot['sessions'];
    return machines.find((machine) => machine.id === session && 'can_close' in machine);
  }, { timeout: 12_000 }).toMatchObject({ can_close: true });
  const closed = await request.post(`/api/sessions/${session}/close`);
  expect(closed.status(), await closed.text()).toBe(204);
});
