import { expect, test } from '@playwright/test';

test.use({ serviceWorkers: 'allow' });

const installed = `
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open('workbox-precache').then((cache) => cache.put('/index.html', new Response('stale'))));
});
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') event.respondWith(caches.match('/index.html'));
});`;

// Catches: a browser that installed Studio's old offline worker keeps serving its cached build after an update.
test('an installed Studio worker removes itself and its caches on its next update', async ({ context, page }) => {
  await context.route('**/sw.js', (route) => route.fulfill({ contentType: 'text/javascript', body: installed }));
  await page.goto('/');
  await page.evaluate(async () => {
    await navigator.serviceWorker.register('/sw.js');
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await expect(page.locator('body')).toHaveText('stale');
  await context.unroute('**/sw.js');

  await page.evaluate(async () => (await navigator.serviceWorker.getRegistration())?.update());
  await expect.poll(() => page.evaluate(async () =>
    [(await navigator.serviceWorker.getRegistrations()).length, (await caches.keys()).length])).toEqual([0, 0]);
  await expect(page.locator('#root')).toBeAttached();
});
