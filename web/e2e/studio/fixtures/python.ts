import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';
import { repo } from './studio';

type Reply = { id: number; code: number; stdout: string; stderr: string };

let server: { child: ChildProcess; ready: Promise<void>; replies: Map<number, (reply: Reply) => void> } | undefined;
let nextId = 0;

function start() {
  const child = spawn('uv', ['run', '--offline', '--no-sync', 'python', '-B', '-m', 'web.e2e.studio.fixtures.runner'], {
    cwd: repo, stdio: ['pipe', 'pipe', 'inherit'],
  });
  const replies = new Map<number, (reply: Reply) => void>();
  let ready!: () => void;
  const started = new Promise<void>((resolve) => { ready = resolve; });
  createInterface({ input: child.stdout! }).on('line', (line) => {
    const message = JSON.parse(line) as Reply | { ready: true };
    if ('ready' in message) return ready();
    replies.get(message.id)!(message);
    replies.delete(message.id);
  });
  child.on('exit', (code, signal) => {
    for (const reply of replies.values()) reply({ id: -1, code: -1, stdout: '', stderr: `Fixture server exited with ${code ?? signal}` });
    server = undefined;
  });
  child.unref();
  (child.stdin as unknown as { unref?: () => void }).unref?.();
  (child.stdout as unknown as { unref?: () => void }).unref?.();
  return { child, ready: started, replies };
}

export type PythonProgram = { module: string } | { path: string } | { code: string };

/**
 * Runs a Python program as `__main__` in a fork of this worker's preloaded fixture interpreter
 * and returns its standard output.
 */
export async function runPython(program: PythonProgram, argv: readonly string[] = [], env: Record<string, string> = {}) {
  server ??= start();
  const current = server;
  await current.ready;
  const id = nextId++;
  const reply = await new Promise<Reply>((resolve) => {
    current.replies.set(id, resolve);
    current.child.stdin!.write(JSON.stringify({ id, program, argv, env: { ...process.env, ...env } }) + '\n');
  });
  if (reply.code !== 0) {
    throw new Error(`Fixture ${Object.values(program)[0].slice(0, 80)} ${argv.join(' ')} exited with ${reply.code}\n${reply.stderr}`);
  }
  return reply.stdout;
}
