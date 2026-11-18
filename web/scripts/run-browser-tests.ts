const allowed = [
  'PATH',
  'TMPDIR',
  'LANG',
  'CI',
  'PLAYWRIGHT_BROWSERS_PATH',
  'PLAYWRIGHT_JSON_OUTPUT_FILE',
  'FEEDBAX_E2E_PORT',
  'FEEDBAX_STUDIO_E2E_WORKERS',
  'FEEDBAX_STUDIO_TIER',
  'FEEDBAX_STUDIO_TRACE',
] as const;

const env: Record<string, string> = { FEEDBAX_BROWSER_TEST_RUNNER: '1' };
for (const name of allowed) {
  const value = process.env[name];
  if (value !== undefined) env[name] = value;
}

// The browser suite runs only on a tree that passes the static checks.
const check = Bun.spawn([process.execPath, 'run', 'check'], { stdin: 'inherit', stdout: 'inherit', stderr: 'inherit' });
if (await check.exited !== 0) process.exit(1);

const runner = Bun.spawn(
  [process.execPath, '--no-env-file', '--bun', 'playwright', 'test', ...process.argv.slice(2)],
  { env, stdin: 'inherit', stdout: 'inherit', stderr: 'inherit' },
);
process.exitCode = await runner.exited;
