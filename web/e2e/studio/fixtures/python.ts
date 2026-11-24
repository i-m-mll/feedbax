import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import type { Readable } from 'node:stream';
import { pythonEnvironment, repo } from './studio';

export type PythonProgram = { module: string } | { path: string } | { code: string };

type Reply = { code: number; stdout: string; stderr: string };
type Runner = { child: ChildProcess; replies: AsyncIterator<string>; exited: Promise<never>; ready: Promise<void> };

const python = () => join(repo, '.venv', 'bin', 'python');

function describe(program: PythonProgram, argv: readonly string[]) {
  return `${Object.values(program)[0].slice(0, 80)} ${argv.join(' ')}`;
}

function startRunner(): Runner {
  // The runner exits when its parent does, so the worker is its direct parent rather than `uv run`.
  const child = spawn(python(), ['-B', '-m', 'web.e2e.studio.fixtures.runner'], {
    cwd: repo, env: pythonEnvironment, stdio: ['pipe', 'ignore', 'pipe', 'pipe'],
  });
  const log: Buffer[] = [];
  child.stderr!.on('data', (chunk: Buffer) => log.push(chunk));
  const exited = once(child, 'exit').then(([code, signal]): never => {
    throw new Error(`Fixture runner exited with ${code ?? signal}\n${Buffer.concat(log).toString()}`);
  });
  exited.catch(() => undefined);
  const replies = createInterface({ input: child.stdio[3] as Readable })[Symbol.asyncIterator]();
  const started: Runner = { child, replies, exited, ready: Promise.resolve() };
  started.ready = next(started).then(() => undefined);
  started.ready.catch(() => undefined);
  child.unref();
  for (const stream of [child.stdin, child.stderr, child.stdio[3]]) (stream as unknown as { unref?: () => void }).unref?.();
  return started;
}

async function next(runner: Runner): Promise<string> {
  const line = await Promise.race([runner.replies.next(), runner.exited]);
  return line.done ? await runner.exited : line.value;
}

let runner: Runner | undefined;
let calls: Promise<unknown> = Promise.resolve();

/** Starts this worker's fixture runner, so its imports overlap the worker's other setup. */
export function startFixtureRunner() {
  runner ??= startRunner();
}

async function serve(program: PythonProgram, argv: readonly string[]) {
  startFixtureRunner();
  const current = runner!;
  try {
    await current.ready;
    current.child.stdin!.write(JSON.stringify({ program, argv }) + '\n');
    return JSON.parse(await next(current)) as Reply;
  } catch (error) {
    if (runner === current) runner = undefined;
    throw error;
  }
}

/**
 * Runs a Python program as `__main__` in this worker's fixture runner, which imported feedbax once
 * when the worker started, and returns its standard output. Calls run one at a time; each is a
 * function of its program and argv, so a fixture takes its home as an argument.
 */
export async function runPython(program: PythonProgram, argv: readonly string[] = []) {
  const reply = calls.then(() => serve(program, argv));
  calls = reply.catch(() => undefined);
  const { code, stdout, stderr } = await reply;
  if (code !== 0) throw new Error(`Fixture ${describe(program, argv)} exited with ${code}\n${stderr}`);
  return stdout;
}

/**
 * Runs a Python program in a fresh interpreter, for a fixture that starts or stops a process and so
 * must not leave it attached to the shared runner. The caller states why beside the call.
 */
export async function runPythonProcess(program: PythonProgram, argv: readonly string[] = []) {
  const source = 'module' in program ? ['-m', program.module] : 'path' in program ? [program.path] : ['-c', program.code];
  const child = spawn(python(), ['-B', ...source, ...argv], { cwd: repo, env: pythonEnvironment, stdio: ['ignore', 'pipe', 'pipe'] });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk));
  const [code, signal] = await once(child, 'close') as [number | null, NodeJS.Signals | null];
  if (code !== 0) throw new Error(`Fixture ${describe(program, argv)} exited with ${code ?? signal}\n${Buffer.concat(stderr).toString()}`);
  return Buffer.concat(stdout).toString();
}
