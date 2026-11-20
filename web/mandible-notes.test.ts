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
    if (request.url === '/api/events') {
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write('event: interface_note:changed\ndata: {}\n\n');
      return;
    }
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ url: request.url, headers: request.headers }));
  });
  await new Promise<void>((resolve) => daemon.listen(socketPath, resolve));
  vite = await createServer({
    configFile: false, root: directory, logLevel: 'silent',
    plugins: [mandibleInterfaceNotes(socketPath)],
    server: { port: 0, strictPort: false, hmr: false },
  });
  await vite.listen();
  origin = vite.resolvedUrls!.local[0].replace(/\/$/, '');
});

afterAll(async () => {
  await vite.close();
  daemon.close();
  rmSync(directory, { recursive: true, force: true });
});

test('the proxy forwards to the socket without its prefix and with the embed header set', async () => {
  const response = await fetch(`${origin}/__mandible/api/interface-notes?app=feedbax-studio`, {
    headers: { 'x-mandible-embed': '0' },
  });
  const echoed = await response.json();
  expect(echoed.url).toBe('/api/interface-notes?app=feedbax-studio');
  expect(echoed.headers['x-mandible-embed']).toBe('1');
});

test('an event stream arrives before the upstream ends', async () => {
  const controller = new AbortController();
  const response = await fetch(`${origin}/__mandible/api/events`, { signal: controller.signal });
  const { value } = await response.body!.getReader().read();
  expect(new TextDecoder().decode(value)).toContain('interface_note:changed');
  controller.abort();
});

test('the page carries the embed tag only while the socket exists', async () => {
  const withSocket = await (await fetch(`${origin}/`)).text();
  expect(withSocket).toMatch(/<script defer src="\/__mandible\/dist\/interface-notes-embed\.js" data-app="feedbax-studio" data-project="feedbax"><\/script>/);
  await new Promise<void>((resolve) => daemon.close(() => resolve()));
  const withoutSocket = await (await fetch(`${origin}/`)).text();
  expect(withoutSocket).not.toContain('__mandible');
});
