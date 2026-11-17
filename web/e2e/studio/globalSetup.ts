import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, renameSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('../../', import.meta.url));
const cache = join(web, 'node_modules', '.cache', 'studio-e2e-dist');

function inputHash() {
  const files = execFileSync('git', ['ls-files', '-co', '--exclude-standard', '-z', '--', '.', ':!e2e'], { cwd: web })
    .toString().split('\0').filter((file) => file && existsSync(join(web, file))).sort();
  const hash = createHash('sha256');
  for (const file of files) hash.update(file).update('\0').update(readFileSync(join(web, file))).update('\0');
  return hash.digest('hex').slice(0, 16);
}

/** Builds the production frontend once per distinct source tree and shares it with every worker. */
export default function globalSetup() {
  const key = inputHash();
  const dist = join(cache, key);
  if (!existsSync(join(dist, 'index.html'))) {
    const staging = `${dist}.${process.pid}`;
    const build = spawnSync('bun', ['--bun', 'vite', 'build', '--outDir', staging, '--emptyOutDir', '--logLevel', 'error'],
      { cwd: web, stdio: 'inherit' });
    if (build.status !== 0) throw new Error(`Studio frontend build failed with ${build.status ?? build.signal}`);
    rmSync(dist, { recursive: true, force: true });
    renameSync(staging, dist);
  }
  for (const entry of readdirSync(cache)) if (entry !== key) rmSync(join(cache, entry), { recursive: true, force: true });
  process.env.FEEDBAX_STUDIO_E2E_DIST = dist;
}
