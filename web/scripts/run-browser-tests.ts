const allowed = [
  'PATH',
  'TMPDIR',
  'LANG',
  'CI',
  'PLAYWRIGHT_BROWSERS_PATH',
  'PLAYWRIGHT_JSON_OUTPUT_FILE',
  'FEEDBAX_E2E_PORT',
  'FEEDBAX_STUDIO_E2E_WORKERS',
  'FEEDBAX_STUDIO_MEASURE',
] as const;

const env: Record<string, string> = {};
for (const name of allowed) {
  const value = process.env[name];
  if (value !== undefined) env[name] = value;
}

const runner = Bun.spawn(
  [process.execPath, '--no-env-file', '--bun', 'playwright', 'test', ...process.argv.slice(2)],
  { env, stdin: 'inherit', stdout: 'inherit', stderr: 'inherit' },
);
process.exitCode = await runner.exited;
