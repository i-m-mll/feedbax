import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type {
  StudioActivityJob,
  StudioJobSnapshot,
  StudioJobSubmission,
} from '../../src/generated/studioContracts';
import { observeJobStreams } from './observeJobStreams';
import { fixture, mapStep, openResultsSlot, seedManagedProject } from './fixtures/canonicalTrain';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
const rates = ['0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008'];

async function hostCommand(home: string, command: string, ...args: string[]): Promise<string> {
  const { stdout } = await run('uv', ['run', '--no-sync', 'python', '-B', '-c', `
from pathlib import Path
import sys
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient

paths = AppPaths.resolve(instance_root=Path(sys.argv[1]))
host = HostClient(paths.host)
${command}
`, home, ...args], { cwd: repo, timeout: 8_000 });
  return stdout.trim();
}

function row(page: Page, label: string) {
  return page.locator('[data-slot="primary"]').getByRole('table', { name: 'Runs', exact: true }).getByRole('row').filter({
    has: page.getByRole('button', { name: label, exact: true }),
  });
}

function post(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path,
  );
}

async function projectJobs(request: APIRequestContext, project: string) {
  const response = await request.get('/api/jobs', { params: { project } });
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json() as StudioJobSnapshot).jobs;
}

async function drainProjects(request: APIRequestContext, projects: string[]) {
  const pending = new Set(['queued', 'awaiting_authorization', 'blocked', 'starting', 'running', 'stopping', 'collecting']);
  const activeIds = async () => (await Promise.all(projects.map((project) => projectJobs(request, project))))
    .flat().filter((job) => pending.has(job.status)).map((job) => job.job_id);
  const jobIds = await activeIds();
  if (!jobIds.length) return;
  const stopped = await request.post('/api/jobs/stop', { data: { job_ids: jobIds, mode: 'force' } });
  expect(stopped.status(), await stopped.text()).toBe(202);
  await expect.poll(activeIds, { timeout: 10_000 }).toEqual([]);
}

test('shares live Jobs across windows and observes the four-slot queue', async ({
  page, context, request, baseURL,
}, testInfo) => {
  test.skip(process.env.FEEDBAX_STUDIO_E2E_PREVIEW !== '1', 'Requires production preview');
  test.setTimeout(50_000);
  if (!baseURL) throw new Error('The real Studio origin is required');
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '6000', ...rates));
  const projectId = snapshot.project_id;
  const observer = await observeJobStreams(baseURL);
  const jobs = () => projectJobs(request, projectId);
  const second = await context.newPage();
  try {
    const open = async (window: Page) => {
      await window.goto(observer.origin + '/?project=' + projectId);
      await mapStep(window, 'train').click();
      await window.getByRole('tab', { name: 'Runs', exact: true }).click();
    };
    await open(page);
    const home = testInfo.config.metadata.home as string;
    const slots = Number(await hostCommand(home, 'print(host.policy()["compute"])'));
    expect(slots).toBe(4);

    const submitted = post(page, projectPath + '/train-runs');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const admission = await submitted;
    expect(admission.status(), await admission.text()).toBe(202);
    const { items } = await admission.json() as StudioJobSubmission;
    expect(items.map((item) => item.disposition)).toEqual(rates.map(() => 'admitted'));

    await expect.poll(async () => {
      const current = await jobs();
      return [
        current.filter((job) => job.status === 'starting' || job.status === 'running').length,
        current.filter((job) => job.status === 'queued').length,
      ];
    }, { timeout: 12_000 }).toEqual([slots, rates.length - slots]);
    const initial = await jobs();
    const queued = initial.filter((job) => job.status === 'queued');
    const first = initial.find((job) => job.status === 'running' || job.status === 'starting')!;
    await expect(row(page, first.label).getByRole('progressbar')).toHaveAttribute('value', /[1-9]/, {
      timeout: 15_000,
    });

    await expect.poll(() => observer.subscriptions().map((stream) => stream.projects)).toEqual([[projectId]]);
    const established = observer.subscriptions();
    await open(second);
    await expect.poll(observer.subscriptions).toEqual(established);
    for (const job of queued) {
      expect(job.reason).toBeTruthy();
      for (const window of [page, second]) {
        await expect(row(window, job.label).getByText('Queued', { exact: true })).toBeVisible();
      }
    }
    await row(second, first.label).getByRole('button', { name: first.label, exact: true }).click();
    await openResultsSlot(second);
    await expect(second.getByRole('img', { name: 'Loss', exact: true }).locator('line title').first()).toBeAttached();

    await page.reload();
    const activity = page.getByRole('region', { name: 'Activity', exact: true });
    await page.getByRole('button', { name: 'Activity', exact: true }).click();
    const selected = activity.getByRole('listitem').filter({ hasText: first.label });
    const stopping = post(page, '/api/jobs/stop');
    await selected.getByRole('button', { name: 'Stop', exact: true }).click();
    expect((await stopping).ok()).toBe(true);

    const secondActivity = second.getByRole('region', { name: 'Activity', exact: true });
    await second.getByRole('button', { name: 'Activity', exact: true }).click();
    const secondAttempt = (label: string) => secondActivity.getByRole('listitem').filter({ hasText: label }).first();
    const resume = secondAttempt(first.label).getByRole('button', { name: 'Resume', exact: true });
    await expect(resume).toBeEnabled({ timeout: 10_000 });
    const stopped = (await jobs()).find((job) => job.job_id === first.job_id);
    expect(stopped).toMatchObject({ status: 'stopped', can_resume: true });
    expect(stopped?.progress?.completed_updates).toBeGreaterThan(0);
    await page.close();
    await expect.poll(observer.subscriptions).toEqual(established);

    const resumed = post(second, `/api/jobs/${first.job_id}/resume`);
    await resume.click();
    const continuation = await resumed;
    expect(continuation.status(), await continuation.text()).toBe(202);
    const attempt = await continuation.json() as StudioActivityJob;
    expect(attempt).toMatchObject({
      continues: first.job_id,
      run_key: first.run_key,
      request_id: first.request_id,
    });
    expect(attempt.job_id).not.toBe(first.job_id);

    const sibling = (await jobs()).find((job) => job.status === 'running' && job.run_key !== first.run_key);
    if (!sibling) throw new Error('Expected another running row to force-stop');
    const forced = post(second, '/api/jobs/stop');
    await secondAttempt(sibling.label).getByRole('button', { name: 'Force Stop', exact: true }).click();
    expect((await forced).ok()).toBe(true);
    await expect.poll(async () =>
      (await jobs()).find((job) => job.job_id === sibling.job_id)?.outcome?.outcome,
    ).toBe('stopped');

    const interruptedId = await hostCommand(home, `
import os
import signal
import time

project, *exclude = sys.argv[2:]
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    for row in host.jobs(project=project):
        if row["id"] in exclude or row["state"] != "running" or not row["checkpoint"] or row["stop_mode"]:
            continue
        pid = row["worker_pid"]
        if pid <= 1 or os.getpgid(pid) != pid:
            raise RuntimeError("The isolated worker group is not live")
        os.killpg(pid, signal.SIGKILL)
        print(row["id"])
        sys.exit(0)
    time.sleep(0.05)
raise TimeoutError("No isolated sibling published a checkpoint")
`, projectId, first.job_id, sibling.job_id, attempt.job_id);
    const interrupted = (await jobs()).find((job) => job.job_id === interruptedId);
    if (!interrupted) throw new Error('The interrupted sibling disappeared from Jobs');
    await expect(secondAttempt(interrupted.label).getByRole('button', { name: 'Resume', exact: true })).toBeEnabled();
    await second.reload();
    await mapStep(second, 'train').click();
    await second.getByRole('tab', { name: 'Runs', exact: true }).click();
    await expect(row(second, sibling.label).getByText('Stopped', { exact: true })).toBeVisible();
    await expect(row(second, interrupted.label).getByText('Interrupted', { exact: true })).toBeVisible();
    await second.getByRole('button', { name: 'Activity', exact: true }).click();
    await expect(secondAttempt(interrupted.label).getByRole('button', { name: 'Resume', exact: true })).toBeEnabled();
    await expect.poll(async () => {
      const current = await jobs();
      return current.find((job) => job.job_id === attempt.job_id)?.progress?.completed_updates ?? 0;
    }, { timeout: 15_000 }).toBeGreaterThan(stopped!.progress!.completed_updates);
    await expect(row(second, first.label).getByRole('progressbar')).toHaveAttribute('value', /[1-9]/);
    const after = await jobs();
    expect(after.find((job) => job.job_id === first.job_id)?.outcome).toEqual(stopped?.outcome);
    expect(after).toHaveLength(rates.length + 1);
  } finally {
    try {
      await drainProjects(request, [projectId]);
    } finally {
      await second.close();
      await observer.close();
    }
  }
});

test('opens a ninth project window while one stream serves active projects', async ({
  context, page, request, baseURL,
}) => {
  test.skip(process.env.FEEDBAX_STUDIO_E2E_PREVIEW !== '1', 'Requires production preview');
  test.setTimeout(45_000);
  if (!baseURL) throw new Error('The real Studio origin is required');
  const files = await fixture('ordinary-files', '12000');
  const projects = await Promise.all(Array.from({ length: 4 }, async () =>
    (await seedManagedProject(request, files)).snapshot.project_id,
  ));
  const runningProjects = projects.slice(0, 3);
  const observer = await observeJobStreams(baseURL);
  const windows: Page[] = [page];
  const open = async (window: Page, projectId: string) => {
    await window.goto(observer.origin + '/?project=' + projectId);
    await mapStep(window, 'train').click();
    await window.getByRole('tab', { name: 'Runs', exact: true }).click();
  };
  const openWindow = async (name: string) => {
    const window = await context.newPage();
    windows.push(window);
    await open(window, name);
    return window;
  };
  const subscriptions = () => observer.subscriptions().map((stream) => [...stream.projects].sort());
  try {
    const owners = await Promise.all(runningProjects.map((project, index) =>
      index === 0 ? open(page, project).then(() => page) : openWindow(project),
    ));
    await Promise.all(owners.map(async (window, index) => {
      const submitted = post(window, '/api/projects/' + runningProjects[index] + '/train-runs');
      await window.getByRole('button', { name: 'Run', exact: true }).click();
      const response = await submitted;
      expect(response.status(), await response.text()).toBe(202);
      const { items } = await response.json() as StudioJobSubmission;
      expect(items.map((item) => item.disposition)).toEqual(['admitted']);
    }));
    await Promise.all(Array.from({ length: 5 }, (_, index) =>
      openWindow(runningProjects[index % runningProjects.length]),
    ));
    await expect.poll(subscriptions).toEqual([[...runningProjects].sort()]);
    await Promise.all(windows.map((window) =>
      expect(window.getByRole('table', { name: 'Runs', exact: true }).getByRole('progressbar')).toHaveAttribute('value', /[1-9]/),
    ));
    expect(windows).toHaveLength(8);

    const opening = performance.now();
    const ninth = await openWindow(projects[3]);
    console.log(`Ninth project window opened in ${Math.round(performance.now() - opening)} ms`);
    await expect.poll(subscriptions).toEqual([[...projects].sort()]);
    await ninth.getByRole('button', { name: 'Activity', exact: true }).click();
    const activity = ninth.getByRole('region', { name: 'Activity', exact: true });
    for (const project of runningProjects) {
      const run = activity.getByRole('listitem').filter({ hasText: project });
      await expect(run.getByRole('button', { name: 'Stop', exact: true })).toBeEnabled();
    }
  } finally {
    try {
      await drainProjects(request, projects);
    } finally {
      await Promise.all(windows.map((window) => window.close()));
      await observer.close();
    }
  }
});
