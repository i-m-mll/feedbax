import { expect, test } from '@playwright/test';

test.use({ serviceWorkers: 'allow' });

test('does not present a cached local session when the production browser loses its connection', async ({
  context,
  page,
}) => {
  test.skip(process.env.FEEDBAX_STUDIO_E2E_PREVIEW !== '1', 'Requires the production service worker');

  await page.goto('/');
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null);

  const path = '/api/sessions';
  const request = page.waitForResponse((response) => new URL(response.url()).pathname === path);
  const online = await page.evaluate(async (url) => (await fetch(url)).ok, path);
  expect(online).toBe(true);
  expect((await request).fromServiceWorker()).toBe(true);

  await context.setOffline(true);
  try {
    const available = await page.evaluate(async (url) => {
      const response = await fetch(url).catch(() => null);
      return response?.ok === true;
    }, path);
    expect(available).toBe(false);
  } finally {
    await context.setOffline(false);
  }
});
