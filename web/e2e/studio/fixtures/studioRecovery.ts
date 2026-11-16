import { expect, type APIRequestContext } from '@playwright/test';
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
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

export function studioRecovery(request: APIRequestContext, home: string, origin: string, backendPort: number) {
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
      return (await request.get(`${origin}/api/health`, { timeout: 1000 })).ok();
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
