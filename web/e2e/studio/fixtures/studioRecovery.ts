import { expect, test } from '@playwright/test';
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const repo = fileURLToPath(new URL('../../../../', import.meta.url));
const run = promisify(execFile);

function processId(text: string, owner: string): number {
  const pid = Number(text.trim());
  if (!Number.isSafeInteger(pid) || pid <= 1) throw new Error(`Invalid isolated ${owner} PID`);
  return pid;
}

export async function hostPid(home: string): Promise<number> {
  const { stdout } = await run('uv', ['run', '--no-sync', 'python', '-B', '-c', `
from pathlib import Path
import sys
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient

paths = AppPaths.resolve(instance_root=Path(sys.argv[1]))
print(HostClient(paths.host).ping()["pid"])
`, home], { cwd: repo, timeout: 10_000 });
  return processId(stdout, 'host');
}

export function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ESRCH') return false;
    throw error;
  }
}

async function stop(child: ChildProcess) {
  if (child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'exit');
    child.kill('SIGTERM');
    await exited;
  }
  if (child.exitCode !== 0 && child.exitCode !== 143) {
    throw new Error(`Studio launcher exited with ${child.exitCode ?? child.signalCode}`);
  }
}

function studioRecovery(home: string, origin: string, backendPort: number) {
  const backendArgs = ['--home', home, '--backend-port', String(backendPort)];
  const log: string[] = [];
  const launchers: ChildProcess[] = [];
  const start = (...args: string[]) => {
    const child = spawn('./scripts/dev.sh', [...backendArgs, ...args], {
      cwd: repo,
      env: { ...process.env, FEEDBAX_STUDIO_TEST_OWNER_PID: String(process.pid) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    for (const stream of [child.stdout, child.stderr]) stream.on('data', (chunk) => log.push(String(chunk)));
    launchers.push(child);
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
  return {
    start,
    healthy,
    stopBackend: async () => {
      const recorded = await readFile(join(home, 'cache', 'backend.pid'), 'utf8');
      process.kill(processId(recorded, 'backend'), 'SIGKILL');
      await expect.poll(healthy).toBe(false);
    },
    startBackend: async () => {
      start('--backend-only');
      await expect.poll(healthy, { timeout: 10_000 }).toBe(true);
    },
    serviceLog: () => log.join(''),
    close: async () => {
      const stopped = await Promise.allSettled(launchers.map(stop));
      for (const result of stopped) if (result.status === 'rejected') throw result.reason;
    },
  };
}

async function loopbackPorts() {
  const servers = [createServer(), createServer()];
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

export type RecoveryStudio = {
  origin: string;
  home: string;
  stopBackend(): Promise<void>;
  startBackend(): Promise<void>;
  serviceLog(): string;
};

export const recoveryTest = test.extend<{ studio: RecoveryStudio }>({
  studio: [async ({}, use, info) => {
    const [webPort, backendPort] = await loopbackPorts();
    const origin = `http://127.0.0.1:${webPort}`;
    const suiteHome = String(info.config.metadata.home);
    const retain = info.config.metadata.retainHome;
    const home = retain
      ? join(suiteHome, 'recovery', randomUUID())
      : join(tmpdir(), `feedbax-studio-recovery-${randomUUID()}`);
    const recovery = studioRecovery(home, origin, backendPort);
    const preview = process.env.FEEDBAX_STUDIO_E2E_PREVIEW === '1'
      ? ['--preview-from', join(suiteHome, 'cache', 'dist')] : [];
    recovery.start('--port', String(webPort), ...preview, ...(retain ? [] : ['--remove-test-home']));
    try {
      await expect.poll(recovery.healthy, { timeout: 35_000 }).toBe(true);
      await use({ origin, home, ...recovery });
    } finally {
      try {
        await recovery.close();
      } finally {
        await info.attach('Studio process log', { body: recovery.serviceLog(), contentType: 'text/plain' });
      }
    }
  }, { timeout: 45_000 }],
  baseURL: async ({ studio }, use) => {
    await use(studio.origin);
  },
});
