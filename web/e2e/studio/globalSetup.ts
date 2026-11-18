import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('../../', import.meta.url));
const cache = join(web, 'node_modules', '.cache', 'studio-e2e-dist');
const jaxCaches = join(web, 'node_modules', '.cache', 'studio-e2e-jax');
const fixtureCaches = join(web, 'node_modules', '.cache', 'studio-e2e-fixtures');
const repo = join(web, '..');

/** Hashes the tracked and untracked files under `paths`, relative to `root`. */
function inputHash(root: string, paths: readonly string[]) {
  const files = execFileSync('git', ['ls-files', '-co', '--exclude-standard', '-z', '--', ...paths], { cwd: root })
    .toString().split('\0').filter((file) => file && existsSync(join(root, file))).sort();
  const hash = createHash('sha256');
  for (const file of files) hash.update(file).update('\0').update(readFileSync(join(root, file))).update('\0');
  return hash.digest('hex').slice(0, 16);
}

/** Keeps the newest `key` directory under `root` and removes the others. */
function keepOnly(root: string, key: string) {
  for (const entry of readdirSync(root)) if (entry !== key) rmSync(join(root, entry), { recursive: true, force: true });
}

/** Builds the production frontend once per distinct source tree and shares it with every worker. */
export default function globalSetup() {
  const key = inputHash(web, ['.', ':!e2e']);
  const dist = join(cache, key);
  if (!existsSync(join(dist, 'index.html'))) {
    const staging = `${dist}.${process.pid}`;
    const build = spawnSync('bun', ['--bun', 'vite', 'build', '--outDir', staging, '--emptyOutDir', '--logLevel', 'error'],
      { cwd: web, stdio: 'inherit' });
    if (build.status !== 0) throw new Error(`Studio frontend build failed with ${build.status ?? build.signal}`);
    rmSync(dist, { recursive: true, force: true });
    renameSync(staging, dist);
  }
  keepOnly(cache, key);
  process.env.FEEDBAX_STUDIO_E2E_DIST = dist;
  // Project-file generators are pure functions of their arguments and the Python they import.
  const python = inputHash(repo, ['feedbax', 'tests', 'examples', 'uv.lock', 'web/e2e/studio/fixtures']);
  mkdirSync(join(fixtureCaches, python), { recursive: true });
  keepOnly(fixtureCaches, python);
  process.env.FEEDBAX_STUDIO_E2E_FIXTURES = join(fixtureCaches, python);
  // Homes share JAX compilations; a dependency change starts a new cache rather than growing the old one.
  const lock = createHash('sha256').update(readFileSync(join(web, '..', 'uv.lock'))).digest('hex').slice(0, 16);
  mkdirSync(join(jaxCaches, lock), { recursive: true, mode: 0o700 });
  keepOnly(jaxCaches, lock);
  process.env.FEEDBAX_STUDIO_E2E_JAX_CACHE = join(jaxCaches, lock);
  // JAX otherwise keys entries by a GPU autotune directory inside each home's cache, so no home could reuse another's.
  process.env.JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES = 'none';
  // Workers run many small JAX programs at once; one thread each avoids oversubscribing the CPU.
  process.env.XLA_FLAGS = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1';
}
