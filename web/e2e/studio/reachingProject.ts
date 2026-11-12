import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { StudioPersistenceDocument } from '../../src/generated/studioContracts';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);

export async function reachingDocument(batches?: number, name?: string): Promise<StudioPersistenceDocument> {
  const args = ['run', '--no-sync', 'python', '-B', 'web/e2e/studio/fixtures/reaching_project.py'];
  if (batches !== undefined) args.push('--batches', String(batches));
  if (name !== undefined) args.push('--name', name);
  const { stdout } = await run('uv', args, { cwd: repo, timeout: 10_000 });
  return JSON.parse(stdout);
}
