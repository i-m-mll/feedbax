import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { join } from 'node:path';
import { repo } from './studio';

export type PythonProgram = { module: string } | { path: string } | { code: string };

type Runner = { child: ChildProcess; ready: Promise<void>; stdout: Buffer[]; stderr: Buffer[] };

function startRunner(): Runner {
  // The runner exits when its parent does, so the worker is its direct parent rather than `uv run`.
  const child = spawn(join(repo, '.venv', 'bin', 'python'), ['-B', '-m', 'web.e2e.studio.fixtures.runner'], {
    cwd: repo, stdio: ['pipe', 'pipe', 'pipe'],
  });
  const runner: Runner = { child, ready: Promise.resolve(), stdout: [], stderr: [] };
  child.stdout!.on('data', (chunk: Buffer) => runner.stdout.push(chunk));
  runner.ready = new Promise((resolve, reject) => {
    let preamble = '';
    const preload = (chunk: Buffer) => {
      preamble += chunk.toString();
      const marker = preamble.indexOf('ready\n');
      if (marker < 0) return;
      child.stderr!.off('data', preload);
      child.stderr!.on('data', (data: Buffer) => runner.stderr.push(data));
      runner.stderr.push(Buffer.from(preamble.slice(marker + 'ready\n'.length)));
      resolve();
    };
    child.stderr!.on('data', preload);
    child.once('exit', (code, signal) => reject(new Error(`Fixture runner exited with ${code ?? signal}\n${preamble}`)));
  });
  runner.ready.catch(() => undefined);
  child.unref();
  for (const stream of [child.stdin, child.stdout, child.stderr]) (stream as unknown as { unref?: () => void }).unref?.();
  return runner;
}

let warm: Runner | undefined;

/**
 * Runs a Python program as `__main__` in this worker's warm fixture interpreter, which has
 * already imported the fixture modules, and returns its standard output. Each program gets a
 * fresh interpreter; the next one starts importing as soon as this one is taken.
 */
export async function runPython(program: PythonProgram, argv: readonly string[] = [], env: Record<string, string> = {}) {
  const runner = warm ?? startRunner();
  warm = startRunner();
  await runner.ready;
  const exited = once(runner.child, 'close') as Promise<[number | null, NodeJS.Signals | null]>;
  runner.child.stdin!.end(JSON.stringify({ program, argv, env: { ...process.env, ...env } }) + '\n');
  const [code, signal] = await exited;
  const stdout = Buffer.concat(runner.stdout).toString();
  if (code !== 0) {
    const stderr = Buffer.concat(runner.stderr).toString();
    throw new Error(`Fixture ${Object.values(program)[0].slice(0, 80)} ${argv.join(' ')} exited with ${code ?? signal}\n${stderr}`);
  }
  return stdout;
}
