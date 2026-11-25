// Studio once installed an offline worker at this path. Its successor removes it: it clears the worker's caches,
// unregisters, and reloads the pages it controlled so they load from the Service.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    await Promise.all((await caches.keys()).map((name) => caches.delete(name)));
    await self.registration.unregister();
    for (const client of await self.clients.matchAll({ type: 'window' })) client.navigate(client.url);
  })());
});
