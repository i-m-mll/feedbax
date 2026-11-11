import { expect, test } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../../', import.meta.url));

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

test('reports that the backend stopped and recovers in the same window', async ({ page }, testInfo) => {
  const [webPort, backendPort] = await loopbackPorts();
  const origin = `http://127.0.0.1:${webPort}`;
  const home = join(tmpdir(), `feedbax-studio-recovery-${randomUUID()}`);
  const backendArgs = ['--home', home, '--backend-port', String(backendPort)];
  const log: string[] = [];
  const start = (...args: string[]) => {
    const child = spawn('./scripts/dev.sh', [...backendArgs, ...args], {
      cwd: repo,
      env: { ...process.env, FEEDBAX_STUDIO_TEST_OWNER_PID: String(process.pid) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    for (const stream of [child.stdout, child.stderr]) {
      stream.on('data', (chunk) => log.push(String(chunk)));
    }
    return child;
  };
  const healthy = async () => {
    try {
      return (await page.request.get(`${origin}/api/provider/health`, { timeout: 1000 })).ok();
    } catch {
      return false;
    }
  };
  const studio = start('--port', String(webPort), '--remove-test-home');
  let restarted: ChildProcess | undefined;
  try {
    await expect.poll(healthy, { timeout: 15_000 }).toBe(true);
    const pid = Number(await readFile(join(home, 'cache', 'backend.pid'), 'utf8'));
    await page.goto(origin);
    const connection = page.getByRole('status', { name: 'Studio connection' });
    await expect(connection).toHaveText(/^connected$/i);
    process.kill(pid, 'SIGTERM');
    await expect.poll(healthy).toBe(false);

    await page.getByTitle('Open project', { exact: true }).click();
    const failure = page.getByRole('alert').filter({ hasText: /service unreachable/i });
    await expect(failure).toBeVisible();
    await expect(connection).toHaveText(/service unreachable/i);
    const retry = page.getByRole('button', { name: 'Retry', exact: true });
    await expect(retry).toBeEnabled();

    restarted = start('--backend-only');
    await expect.poll(healthy, { timeout: 10_000 }).toBe(true);
    await retry.click();
    await expect(failure).toHaveCount(0);
    await expect(connection).toHaveText(/^connected$/i);
  } finally {
    const launchers = restarted ? [restarted, studio] : [studio];
    const stopped = await Promise.allSettled(launchers.map(stop));
    await testInfo.attach('Studio process log', { body: log.join(''), contentType: 'text/plain' });
    for (const result of stopped) {
      if (result.status === 'rejected') throw result.reason;
    }
  }
});
