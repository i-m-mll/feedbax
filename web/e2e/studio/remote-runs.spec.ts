import { expect, test } from './fixtures/studio';
import { readdir, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type {
  StudioActivityJob, StudioJobEventHistory, StudioJobSnapshot, StudioMachineApprovalReceipt, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { fixture, mapStep, runName, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ studioBackend: 'remote' });

test('Machine approval reviews the saved selection and collects the remote result', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(50_000);
  const fixtureRoot = join(studio.home, 'state', 'studio-e2e-remote');
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
  await expect(machine).toHaveText(/^This Mac/);
  await pick(machine, /^Rehearsal Machine/);
  const run = page.getByRole('button', { name: 'Run', exact: true });
  await run.click();
  const approval = page.getByRole('dialog', { name: 'Needs approval' });
  await expect(approval.getByText('Cost ceiling', { exact: true })).toBeVisible();
  await expect(approval.getByText('$0.12', { exact: true })).toBeVisible();
  await expect(approval.getByText('Expiry', { exact: true })).toBeVisible();
  await expect(approval.getByText('Results lost', { exact: true })).toBeVisible();
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
  const queueGate = join(fixtureRoot, 'queue-gate');
  const openingGate = join(fixtureRoot, 'opening-gate');
  const meter = join(fixtureRoot, 'meter');
  await run.click();
  await Promise.all([writeFile(downloadGate, ''), writeFile(queueGate, ''), writeFile(openingGate, ''), writeFile(meter, '12345')]);
  const accepted = page.waitForResponse((response) => new URL(response.url()).pathname === `${path}/approve`);
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  let approvalReceipt: StudioMachineApprovalReceipt;
  try {
    try {
      try {
        await approval.getByRole('button', { name: 'Approve' }).click();
        const response = await accepted;
        expect(response.status(), await response.text()).toBe(202);
        approvalReceipt = await response.json() as StudioMachineApprovalReceipt;
        expect(approvalReceipt).toEqual({ session_id: expect.any(String) });
        await expect(approval).toBeHidden();
        await expect(runs.getByText('Queued', { exact: true })).toBeVisible();
        expect(await readdir(join(fixtureRoot, 'venues'))).toEqual([]);
      } finally {
        await unlink(queueGate);
      }
      await expect(runs.getByText('Starting', { exact: true })).toBeVisible();
      await expect(runs.getByText('Opening machine: starting the worker service', { exact: true })).toBeVisible();
      expect(await attempts()).toEqual([]);
    } finally {
      await unlink(openingGate);
    }
    await expect(runs.getByText('Downloading', { exact: true })).toBeVisible({ timeout: 18_000 });
    const meter = runs.getByRole('progressbar', { name: 'Downloading' });
    await expect.poll(async () => Number(await meter.getAttribute('max'))).toBeGreaterThan(0);
    expect(Number(await meter.getAttribute('value'))).toBeLessThan(Number(await meter.getAttribute('max')));
    const live = await request.get('/api/jobs', { params: { project: snapshot.project_id } });
    expect((await live.json() as StudioJobSnapshot).jobs).toEqual([
      expect.objectContaining({ target: expect.objectContaining({ kind: 'session', session_id: approvalReceipt.session_id }) }),
    ]);
  } finally {
    await unlink(downloadGate);
  }
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 20_000 });
  const [attempt] = await attempts();
  expect(attempt).toMatchObject({ status: 'succeeded', successful_receipt: expect.any(Object) });
  await page.reload();
  await mapStep(page, 'train').click();
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const card = page.getByRole('region', { name: 'Rehearsal Machine', exact: true });
  await expect(card.getByText('Closes at', { exact: true })).toBeVisible();
  await expect(card.getByText('Billed cost', { exact: true })).toBeVisible();
  await expect(card.getByText('$0.012345', { exact: true })).toBeVisible();
  await expect(card.getByText('Idle since', { exact: true })).toBeVisible();
  await card.getByText('Opening history', { exact: true }).click();
  await expect(card.getByRole('listitem')).toHaveText([
    /Waiting for a machine/, /Connecting to the machine/, /Starting the worker service/,
  ]);
  const close = card.getByRole('button', { name: 'Close', exact: true });
  await expect(close).toBeEnabled();
  const closed = page.waitForResponse((response) => new URL(response.url()).pathname ===
    `/api/sessions/${approvalReceipt.session_id}/close`);
  await close.click();
  const confirmation = await closed;
  expect(confirmation.status(), confirmation.statusText()).toBe(204);
  await expect(card).toBeHidden({ timeout: 10_000 });
  await unlink(meter);
  expect(await attempts()).toEqual([attempt]);
  const activityAfter = await request.get('/api/jobs');
  const recorded = await activityAfter.json() as StudioJobSnapshot;
  const session = approvalReceipt.session_id;
  const source = recorded.venue_cursors?.find(({ session_id }) => session_id === session);
  expect(source).toBeDefined();
  const retained = await request.get('/api/events/history', { params: {
    project: snapshot.project_id, venue: source!.session_id, stream: source!.stream_id, through: source!.cursor,
  } });
  expect(retained.ok(), await retained.text()).toBe(true);
  const events = (await retained.json() as StudioJobEventHistory).events;
  expect(events.map(({ cursor }) => cursor)).toEqual([...new Set(events.map(({ cursor }) => cursor))].sort((a, b) => a - b));
  expect(events.some(({ job_id, kind, data }) =>
    job_id === attempt.job_id && kind === 'progress' && data.completed_updates === 2)).toBe(true);
  expect(events.every(({ venue }) => venue?.stream_id === source!.stream_id)).toBe(true);
  await page.reload();
  await mapStep(page, 'train').click();
  await runs.getByRole('rowheader', { name: 'default' }).getByRole('button').click();
  await page.getByRole('button', { name: 'Open Run', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Loss', exact: true }).locator(`line[data-attempt="${attempt.job_id}"]`).first())
    .toBeAttached();
});

test('A set of six remote rows keeps its Jobs states and commands across reloads', { tag: '@round' }, async ({ page, request, studio }, info) => {
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
    has: page.getByRole('rowheader', { name: runName(rowName(activity)) }),
  });
  const reopen = async () => {
    await page.reload();
    await mapStep(page, 'train').click();
  };
  const command = (name: string) => page.getByRole('button', { name, exact: true });

  await pick(page.getByRole('combobox', { name: 'Machine' }), /^Rehearsal Machine/);
  await command('Run').click();
  const approval = page.getByRole('dialog', { name: 'Needs approval' });
  await expect(approval.getByRole('listitem')).toHaveCount(6);
  await expect(approval.getByText('Cost ceiling', { exact: true })).toBeVisible();
  expect(await jobs()).toEqual([]);
  const accepted = page.waitForResponse((response) => new URL(response.url()).pathname === `${path}/approve`);
  await approval.getByRole('button', { name: 'Approve' }).click();
  const response = await accepted;
  expect(response.status(), await response.text()).toBe(202);
  const { session_id: session } = await response.json() as StudioMachineApprovalReceipt;
  await expect(runs.getByText('Queued', { exact: true })).toHaveCount(2, { timeout: 12_000 });
  await expect(runs.getByText('Waiting for a training slot.', { exact: true })).toHaveCount(2);
  await expect.poll(async () => (await jobs()).filter(({ status }) => status === 'running').length,
    { timeout: 15_000 }).toBe(4);
  expect((await jobs()).map(({ target }) => target)).toEqual(Array.from({ length: 6 }, () =>
    expect.objectContaining({ kind: 'session', session_id: session })));
  await reopen();
  await expect(runs.getByText('Queued', { exact: true })).toHaveCount(2);
  await expect(runs.getByText('Running', { exact: true })).toHaveCount(4);
  let active = (await jobs()).filter(({ status }) => status === 'running');
  expect(active.every(({ target }) => target?.kind === 'session' && target.session_id === session)).toBe(true);
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

  await runProjectFixture('web.e2e.studio.fixtures.remote_run_process', studio.home, session, interrupt.job_id);
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
  expect(closed.status(), closed.statusText()).toBe(204);
});

test('Opening refusal survives reload on the exact approved row', async ({ page, request, studio }, info) => {
  const refusal = join(studio.home, 'state', 'studio-e2e-remote', 'opening-refusal');
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const path = `${projectPath}/train-runs`;
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await pick(page.getByRole('combobox', { name: 'Machine' }), /^Rehearsal Machine/);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const approval = page.getByRole('dialog', { name: 'Needs approval' });
  await writeFile(refusal, '');
  try {
    const ack = page.waitForResponse((response) => new URL(response.url()).pathname === `${path}/approve`);
    await approval.getByRole('button', { name: 'Approve' }).click();
    const accepted = await ack;
    expect(accepted.status(), await accepted.text()).toBe(202);
    await expect(approval).toBeHidden();
    await expect(runs.getByText('Failed', { exact: true })).toBeVisible({ timeout: 10_000 });
    await page.reload();
    await mapStep(page, 'train').click();
    await expect(runs.getByText('Failed', { exact: true })).toBeVisible();
    await runs.getByRole('cell', { name: 'Failed', exact: true }).hover();
    await expect(page.getByRole('tooltip')).toContainText('Starting the worker service · The worker service could not be started.');
    const persisted = await request.get(`${path}?revision=${snapshot.revision}`);
    const projected = await persisted.json() as StudioSavedTrainRuns;
    expect(projected.rows).toEqual([expect.objectContaining({ status: 'failed', attempts: [] })]);
    const replay = await request.post(`${path}/approve`, { data: accepted.request().postDataJSON() });
    expect(replay.status(), await replay.text()).toBe(202);
    expect(await replay.json()).toEqual(await accepted.json());
  } finally {
    await unlink(refusal);
  }
});

test('The Machine picker follows provider failure and recovery without reload', async ({ page, request, studio }, info) => {
  const outage = join(studio.home, 'state', 'studio-e2e-remote', 'provider-unreachable');
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  const machine = page.getByRole('combobox', { name: 'Machine' });
  const run = page.getByRole('button', { name: 'Run', exact: true });
  await pick(machine, 'Rehearsal Machine · Ready');
  await writeFile(outage, '');
  try {
    await run.click();
    await expect(machine).toHaveText('Rehearsal Machine · Unreachable');
    await expect(machine).toHaveAccessibleDescription('The provider did not answer.');
    await expect(run).toBeDisabled();
  } finally {
    await unlink(outage);
  }
  await expect(machine).toHaveText('Rehearsal Machine · Ready', { timeout: 12_000 });
  await expect(run).toBeEnabled();
});
