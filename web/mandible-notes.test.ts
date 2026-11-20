import { createServer as createHttpServer, type Server } from 'node:http';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createServer, type ViteDevServer } from 'vite';
import { afterAll, beforeAll, expect, test } from 'vitest';
import { mandibleInterfaceNotes } from './mandible-notes';

let directory: string;
let socketPath: string;
let daemon: Server;
let vite: ViteDevServer;
let origin: string;

beforeAll(async () => {
  directory = mkdtempSync(path.join(tmpdir(), 'mandible-notes-'));
  socketPath = path.join(directory, 'daemon.sock');
  writeFileSync(path.join(directory, 'index.html'), '<!doctype html><html><head></head><body></body></html>');
  daemon = createHttpServer((request, response) => {
    if (request.url !== '/api/instance') return response.writeHead(404).end();
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ instance: { public_origin: 'https://app.mandible.localhost:19999' } }));
  });
  await new Promise<void>((resolve) => daemon.listen(socketPath, resolve));
  vite = await createServer({
    configFile: false, root: directory, logLevel: 'silent',
    plugins: [mandibleInterfaceNotes(socketPath)],
    server: { port: 0, hmr: false },
  });
  await vite.listen();
  origin = vite.resolvedUrls!.local[0];
});

afterAll(async () => {
  await vite.close();
  rmSync(directory, { recursive: true, force: true });
});

test('the page loads the embed from the daemon only while its socket exists', async () => {
  const withSocket = await (await fetch(origin)).text();
  expect(withSocket).toContain('<script type="module" src="https://app.mandible.localhost:19999/embed/notes.js"></script>');
  await new Promise<void>((resolve) => daemon.close(() => resolve()));
  const withoutSocket = await (await fetch(origin)).text();
  expect(withoutSocket).not.toContain('embed/notes.js');
});
