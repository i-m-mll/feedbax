/**
 * Browser tests run through `scripts/run-browser-tests.ts`, which runs Playwright under Bun with an
 * allowlisted environment; Playwright started directly runs under Node with the caller's environment.
 */
export function requireBrowserTestRunner() {
  if (process.env.FEEDBAX_BROWSER_TEST_RUNNER !== '1') {
    throw new Error('Run browser tests with `bun run test:e2e` or `bun run test:studio` (arguments pass through to Playwright), not `playwright test`.');
  }
}
