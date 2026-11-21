import { expect, test as base } from '@playwright/test';
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

export const repo = fileURLToPath(new URL('../../../../', import.meta.url));

/** The Python application each Studio instance serves. */
export type StudioBackend = 'app' | 'remote';
/** `preview` serves the memoized production build; `dev` serves source modules through Vite. */
export type StudioFrontend = 'preview' | 'dev';

const backendModules: Record<StudioBackend, string> = {
  app: 'scripts.studio_dev',
  remote: 'web.e2e.studio.fixtures.remote_runs_backend',
};

/**
 * Bounds the wait for a Studio's first job to report progress. Each Studio home starts with an
 * idle Host, so its first worker imports feedbax and, for programs the shared JAX cache lacks,
 * compiles (about 10 s, more under parallel load) before training starts; later jobs report
 * within about 3 s.
 */
export const firstJobTimeout = 20_000;

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
      env: { ...process.env, FEEDBAX_OWNER_PID: String(process.pid), FEEDBAX_STUDIO_TEST_HOME: home },
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
      await expect.poll(healthy, { timeout: 15_000 }).toBe(true);
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
 */
const traced = base.extend<{ namedTrace: void }, { fixtureRunner: void }>({
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

type WorkerFixtures = { studioBackend: StudioBackend; studioFrontend: StudioFrontend; studio: Studio };

/** Each worker owns one Studio; tests in the worker share its home and job host. */
export const test = traced.extend<{}, WorkerFixtures>({
  studioBackend: ['app', { scope: 'worker', option: true }],
  studioFrontend: ['preview', { scope: 'worker', option: true }],
  studio: [async ({ studioBackend, studioFrontend }, use) => {
    const { studio, close } = await launchStudio(studioBackend, studioFrontend);
    try {
      await use(studio);
    } finally {
      await close();
    }
  }, { scope: 'worker', timeout: 60_000 }],
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
export const isolatedTest = traced.extend<{ studio: Studio }>({
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
  }, { timeout: 60_000 }],
  baseURL: async ({ studio }, use) => {
    await use(studio.origin);
  },
});

export { expect };
