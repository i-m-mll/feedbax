import { expect, test as base, type BrowserContext, type Page, type Response } from '@playwright/test';
import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { closeSync, openSync, readFileSync, rmSync } from 'node:fs';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startFixtureRunner } from './python';
import type { AnalysisDocument, StudioActivityJob, StudioJobSnapshot, StudioProjectSnapshot } from '../../../src/generated/studioContracts';

export const repo = fileURLToPath(new URL('../../../../', import.meta.url));

/** The e2e Python environment: the test Python task is installed, as a package registers one. */
export const pythonEnvironment = {
  ...process.env, PYTHONPATH: [`${repo}tests/support/plugins`, process.env.PYTHONPATH].filter(Boolean).join(':'),
};

/** The Python application each Studio instance serves. */
export type StudioBackend = 'app' | 'remote';
/** `preview` serves the memoized production build; `dev` serves source modules through Vite. */
export type StudioFrontend = 'preview' | 'dev';

const backendModules: Record<StudioBackend, string> = {
  app: 'scripts.studio_dev',
  remote: 'web.e2e.studio.fixtures.remote_runs_backend',
};

/**
 * Bounds a wait for a Job to reach a state. A Job reports only after its worker imports feedbax and, for programs the
 * shared JAX cache lacks, compiles: about 10 s alone, and up to several times that while the tier's other Jobs and
 * other work on the machine share the CPU.
 */
export const jobTimeout = 90_000;

/** Bounds a Studio's start: its backend imports feedbax and opens its home, about 10 s alone and longer under load. */
const launchTimeout = 60_000;

export type Studio = {
  origin: string;
  home: string;
  stopBackend(): Promise<void>;
  startBackend(): Promise<void>;
  serviceLog(): string;
};

function processId(text: string, owner: string): number {
  const pid = Number(text.trim());
  if (!Number.isSafeInteger(pid) || pid <= 1) throw new Error(`Invalid isolated ${owner} PID`);
  return pid;
}

async function loopbackPorts(count: number) {
  const servers = Array.from({ length: count }, () => createServer());
  try {
    return await Promise.all(servers.map(async (server) => {
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const address = server.address();
      if (address === null || typeof address === 'string') throw new Error('Expected a loopback TCP port');
      return address.port;
    }));
  } finally {
    for (const server of servers) server.close();
  }
}

async function stop(child: ChildProcess, log: () => string) {
  if (child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'exit');
    child.kill('SIGTERM');
    await exited;
  }
  if (child.exitCode !== 0 && child.exitCode !== 143) {
    throw new Error(`Studio launcher exited with ${child.exitCode ?? child.signalCode}\n${log().slice(-4000)}`);
  }
}

/** Lists the processes still working in `home`, which a stopped Studio must not leave behind. */
function processesIn(home: string) {
  try {
    return execFileSync('pgrep', ['-lf', home]).toString().trim();
  } catch {
    return '';
  }
}

/**
 * Starts one Studio (frontend, backend and lazily its job host) in a private home that is
 * removed after its jobs settle on close.
 */
export async function launchStudio(backend: StudioBackend, frontend: StudioFrontend) {
  const [webPort, backendPort] = await loopbackPorts(2);
  const origin = `http://127.0.0.1:${webPort}`;
  const home = join(tmpdir(), `feedbax-studio-test-${randomUUID()}`);
  // Launchers write to a file rather than to pipes, so a descendant that outlives its launcher
  // cannot hold this worker open.
  const logPath = `${home}.log`;
  const logFd = openSync(logPath, 'a');
  const log = () => readFileSync(logPath, 'utf8');
  const launchers: ChildProcess[] = [];
  const start = (...args: string[]) => {
    const child = spawn('uv', [
      'run', '--offline', '--no-sync', 'python', '-B', '-m', backendModules[backend],
      '--home', home, '--backend-port', String(backendPort), ...args,
    ], {
      cwd: repo,
      env: { ...pythonEnvironment, FEEDBAX_OWNER_PID: String(process.pid), FEEDBAX_STUDIO_TEST_HOME: home },
      stdio: ['ignore', logFd, logFd],
    });
    launchers.push(child);
    return child;
  };
  const healthy = async () => {
    try {
      const response = await fetch(`${origin}/api/health`, { signal: AbortSignal.timeout(1000) });
      await response.arrayBuffer();
      return response.ok;
    } catch {
      return false;
    }
  };
  const dist = process.env.FEEDBAX_STUDIO_E2E_DIST;
  if (frontend === 'preview' && dist === undefined) throw new Error('The Studio global setup did not build the frontend');
  const jaxCache = process.env.FEEDBAX_STUDIO_E2E_JAX_CACHE;
  if (jaxCache === undefined) throw new Error('The Studio global setup did not create the JAX cache');
  const main = start('--port', String(webPort), '--remove-test-home', '--jax-cache', jaxCache,
    ...(frontend === 'preview' ? ['--preview-from', dist!] : []));
  while (!await healthy()) {
    if (main.exitCode !== null || main.signalCode !== null) {
      throw new Error(`Studio launcher exited with ${main.exitCode ?? main.signalCode}\n${log()}`);
    }
    await new Promise((done) => setTimeout(done, 50));
  }
  const studio: Studio = {
    origin,
    home,
    stopBackend: async () => {
      const recorded = await readFile(join(home, 'cache', 'backend.pid'), 'utf8');
      process.kill(processId(recorded, 'backend'), 'SIGKILL');
      await expect.poll(healthy).toBe(false);
    },
    startBackend: async () => {
      start('--backend-only');
      await expect.poll(healthy, { timeout: launchTimeout }).toBe(true);
    },
    serviceLog: log,
  };
  return {
    studio,
    close: async () => {
      const stopped = await Promise.allSettled(launchers.map((child) => stop(child, log)));
      closeSync(logFd);
      for (const result of stopped) if (result.status === 'rejected') throw result.reason;
      const survivors = processesIn(home);
      if (survivors) throw new Error(`Studio processes outlived their launcher in ${home}:\n${survivors}\n${log().slice(-4000)}`);
      rmSync(logPath);
    },
  };
}

/**
 * Traces are off; `FEEDBAX_STUDIO_TRACE=<spec file>[,<spec file>…]` records one for each failing
 * test in those specs. Tracing the context here, not through the worker-scoped `trace` option, keeps the spec
 * in the shared workers and their Studio.
 *
 * `FEEDBAX_STUDIO_CPU_THROTTLE=<rate>` slows every page's CPU by that factor, to reproduce what a spec does under load.
 */
const traced = base.extend<{ namedTrace: void; cpuThrottle: void }, { fixtureRunner: void }>({
  cpuThrottle: [async ({ context }, use) => {
    const rate = Number(process.env.FEEDBAX_STUDIO_CPU_THROTTLE ?? 1);
    if (!(rate >= 1)) throw new Error('FEEDBAX_STUDIO_CPU_THROTTLE must be a number of at least 1.');
    const throttle = async (page: Page) => {
      const session = await context.newCDPSession(page);
      await session.send('Emulation.setCPUThrottlingRate', { rate });
    };
    if (rate > 1) {
      context.on('page', (page) => void throttle(page));
      await Promise.all(context.pages().map(throttle));
    }
    await use();
  }, { auto: true }],
  fixtureRunner: [async ({}, use) => {
    startFixtureRunner();
    await use();
  }, { scope: 'worker', auto: true }],
  namedTrace: [async ({ context }, use, info) => {
    const spec = process.env.FEEDBAX_STUDIO_TRACE;
    const on = spec !== undefined && spec.split(',').some((name) => info.file.endsWith(name));
    if (on) await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
    await use();
    if (!on) return;
    if (info.status === info.expectedStatus) {
      await context.tracing.stop();
    } else {
      const path = info.outputPath('trace.zip');
      await context.tracing.stop({ path });
      await info.attach('trace', { path, contentType: 'application/zip' });
    }
  }, { auto: true }],
});

/** Job statuses that hold a place in the Host's queue or a compute slot. */
const unsettled = new Set<StudioActivityJob['status']>([
  'queued', 'awaiting_authorization', 'blocked', 'waiting', 'starting', 'running', 'stopping', 'collecting',
]);

async function unsettledJobs(origin: string) {
  const response = await fetch(`${origin}/api/jobs`);
  if (!response.ok) throw new Error(`Listing the Studio's Jobs failed with ${response.status}: ${await response.text()}`);
  return (await response.json() as StudioJobSnapshot).jobs.filter((job) => unsettled.has(job.status));
}

/**
 * Force-stops every Job a test left in its worker's shared queue and waits until none holds a place,
 * so the worker's next test starts on an empty queue whether this one passed or failed.
 */
async function settleJobs(origin: string) {
  const left = await unsettledJobs(origin);
  if (!left.length) return;
  const stopping = left.filter((job) => job.status !== 'stopping');
  if (stopping.length) {
    const response = await fetch(`${origin}/api/jobs/stop`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ job_ids: stopping.map((job) => job.job_id), mode: 'force' }),
    });
    if (response.status !== 202) throw new Error(`Stopping leftover Jobs failed with ${response.status}: ${await response.text()}`);
  }
  await expect.poll(async () => (await unsettledJobs(origin)).map((job) => `${job.label}: ${job.status}`),
    { message: 'Jobs a test left behind settle after a force stop' }).toEqual([]);
}

const snapshotRoute = /^\/api\/projects(?:\/[^/]+(?:\/(?:open|rename|files|history\/replay|members\/command|model\/(?:edit|undo)|loaded-models\/import))?)?$/;

function memberNames(snapshot: StudioProjectSnapshot) {
  const keys = new Set<string>();
  const labels = new Set<string>();
  const layerNames = new Set<string>();
  function collect(members: Record<string, { label: string }> | undefined) {
    for (const [key, { label }] of Object.entries(members ?? {})) {
      keys.add(key);
      labels.add(label);
    }
  }
  function analysis(payload: unknown) {
    const { operations, views } = payload as AnalysisDocument;
    collect(operations);
    collect(views);
    for (const { figure } of Object.values(views ?? {})) {
      if ('layers' in figure) for (const key of Object.keys(figure.layers)) layerNames.add(key);
    }
  }

  collect(snapshot.project.documents);
  collect(snapshot.project.steps);
  for (const [key, document] of Object.entries(snapshot.project.documents ?? {})) {
    if (document.schema !== 'feedbax.analysis_document.v1') continue;
    const payload = 'payload' in document ? document.payload : snapshot.composed?.[key];
    if (payload && typeof payload === 'object') analysis(payload);
  }
  for (const step of Object.values(snapshot.project.steps ?? {})) {
    if (step.kind === 'analyze' && 'inline' in step.base) analysis(step.base.inline);
  }
  // A variant can introduce members of its own and label existing ones differently.
  for (const variant of Object.values(snapshot.variant_names)) {
    for (const group of [variant.documents, variant.steps, ...Object.values(variant.elements)]) {
      for (const [key, label] of Object.entries(group)) {
        keys.add(key);
        labels.add(label);
      }
    }
  }
  return { keys: [...keys].filter((key) => !labels.has(key)), layerNames: [...layerNames] };
}

/** Reads snapshots the browser already requested, then checks the visible text in each open Studio page. */
async function checkMemberLabels({ context, studio }: { context: BrowserContext; studio: Studio }, use: () => Promise<void>) {
  const reads: Promise<void>[] = [];
  const latest = new Map<string, { order: number; members: ReturnType<typeof memberNames> }>();
  const read = (response: Response) => {
    const { origin, pathname } = new URL(response.url());
    if (origin !== studio.origin || !response.ok() || !snapshotRoute.test(pathname)) return;
    const order = reads.length;
    reads.push(response.json().catch(() => undefined).then((body) => {
      // Some specs abort requests and some matching routes also return a list.
      const snapshot = (body?.snapshot ?? body) as StudioProjectSnapshot | undefined;
      if (!snapshot?.project || !snapshot.variant_names) return;
      if ((latest.get(snapshot.project_id)?.order ?? -1) < order) {
        latest.set(snapshot.project_id, { order, members: memberNames(snapshot) });
      }
    }));
  };
  context.on('response', read);
  try {
    await use();
  } finally {
    context.off('response', read);
    await Promise.all(reads);
    for (const page of context.pages()) {
      const url = new URL(page.url());
      const shown = url.searchParams.get('project');
      const members = url.origin === studio.origin && shown ? latest.get(shown)?.members : undefined;
      if (!members?.keys.length) continue;
      const leaked = await page.evaluate(({ keys, layerNames }) => {
        const members = new Set(keys);
        const layers = new Set(layerNames);
        const nodes = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const range = document.createRange();
        const found = new Set<string>();
        while (nodes.nextNode()) {
          const node = nodes.currentNode;
          const value = node.textContent?.trim();
          const parent = node.parentElement;
          if (!value || !members.has(value) || !parent || parent.closest('[data-member-key], input, textarea, .sr-only')) continue;
          // The title names the project itself, which can share a name with one of its members.
          if (parent.closest('.studio-window-bar__project h1')) continue;
          // A figure layer can have the same authored name as a different member's key.
          if (layers.has(value) && parent.closest('[data-node]')?.getAttribute('data-node') === JSON.stringify(['layer', value])) continue;
          if (getComputedStyle(parent).visibility !== 'visible') continue;
          range.selectNodeContents(node);
          if (Array.from(range.getClientRects()).some(({ width, height }) => width > 0 && height > 0)) {
            const place = [parent.localName, ...parent.classList].join('.');
            found.add(`${value} in ${place}`);
          }
        }
        return [...found];
      }, members);
      expect(leaked, 'Studio shows member names outside Change Key').toEqual([]);
    }
  }
}

type WorkerFixtures = { studioBackend: StudioBackend; studioFrontend: StudioFrontend; studio: Studio };

/** Each worker owns one Studio; tests in the worker share its home and job host, and each leaves the queue empty. */
export const test = traced.extend<{ settledJobs: void; memberLabels: void }, WorkerFixtures>({
  memberLabels: [checkMemberLabels, { auto: true }],
  settledJobs: [async ({ studio }, use) => {
    await use();
    await settleJobs(studio.origin);
  }, { auto: true }],
  studioBackend: ['app', { scope: 'worker', option: true }],
  studioFrontend: ['preview', { scope: 'worker', option: true }],
  studio: [async ({ studioBackend, studioFrontend }, use) => {
    const { studio, close } = await launchStudio(studioBackend, studioFrontend);
    try {
      await use(studio);
    } finally {
      await close();
    }
  }, { scope: 'worker', timeout: launchTimeout }],
  baseURL: async ({ studio }, use) => {
    await use(studio.origin);
  },
});

async function workerLogs(home: string) {
  const root = join(home, 'state', 'host', 'workers');
  const tokens = await readdir(root).catch(() => []);
  const logs = await Promise.all(tokens.map(async (token) =>
    `\n--- Host worker ${token}\n` + await readFile(join(root, token, 'startup.log'), 'utf8').catch(() => '')));
  return logs.join('');
}

/** Each test owns its Studio, for tests that stop and restart the backend. */
export const isolatedTest = traced.extend<{ studio: Studio; memberLabels: void }>({
  memberLabels: [checkMemberLabels, { auto: true }],
  studio: [async ({}, use, info) => {
    const { studio, close } = await launchStudio('app', 'preview');
    try {
      await use(studio);
    } finally {
      const failed = info.status !== info.expectedStatus;
      const logs = failed ? studio.serviceLog() + await workerLogs(studio.home) : '';
      try {
        await close();
      } finally {
        if (failed) {
          const path = info.outputPath('studio.log');
          await writeFile(path, logs);
          await info.attach('Studio process log', { path, contentType: 'text/plain' });
        }
      }
    }
  }, { timeout: launchTimeout }],
  baseURL: async ({ studio }, use) => {
    await use(studio.origin);
  },
});

export { expect };
