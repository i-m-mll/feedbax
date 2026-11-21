import { createServer as createHttpServer, type Server } from 'node:http';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createServer, preview, type PreviewServer, type ViteDevServer } from 'vite';
import { afterAll, beforeAll, expect, test } from 'vitest';
import { mandibleInterfaceNotes } from './mandible-notes';

// Longer than preview's compression threshold, so preview would compress it.
const page = `<!doctype html><html><head></head><body>${'<p>Studio</p>'.repeat(200)}</body></html>`;
const tag = '<script type="module" src="https://app.mandible.localhost:19999/embed/notes.js"></script>';

let directory: string;
let socketPath: string;
let daemon: Server;
let dev: ViteDevServer;
let built: PreviewServer;
let devOrigin: string;
let previewOrigin: string;

beforeAll(async () => {
  directory = mkdtempSync(path.join(tmpdir(), 'mandible-notes-'));
  socketPath = path.join(directory, 'daemon.sock');
  writeFileSync(path.join(directory, 'index.html'), page);
  mkdirSync(path.join(directory, 'dist/assets'), { recursive: true });
  writeFileSync(path.join(directory, 'dist/index.html'), page);
  writeFileSync(path.join(directory, 'dist/assets/app.js'), 'export {};');
  daemon = createHttpServer((request, response) => {
    if (request.url !== '/api/instance') return response.writeHead(404).end();
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ instance: { public_origin: 'https://app.mandible.localhost:19999' } }));
  });
  await new Promise<void>((resolve) => daemon.listen(socketPath, resolve));
  const shared = { configFile: false, root: directory, logLevel: 'silent', plugins: [mandibleInterfaceNotes(socketPath)] } as const;
  dev = await createServer({ ...shared, server: { port: 0, hmr: false } });
  await dev.listen();
  devOrigin = dev.resolvedUrls!.local[0];
  built = await preview({ ...shared, preview: { port: 0 } });
  previewOrigin = built.resolvedUrls!.local[0];
});

afterAll(async () => {
  await dev.close();
  await new Promise<void>((resolve) => built.httpServer.close(() => resolve()));
  rmSync(directory, { recursive: true, force: true });
});

const text = async (url: string, headers?: Record<string, string>) => (await fetch(url, { headers: { 'accept-encoding': 'gzip', ...headers } })).text();

test('the dev page loads the embed from the daemon', async () => {
  expect(await text(devOrigin)).toContain(tag);
});

test('preview pages load the embed on every load, assets are untouched', async () => {
  const first = await fetch(previewOrigin);
  expect(await first.text()).toContain(tag);
  expect(await text(`${previewOrigin}index.html`, { 'if-none-match': first.headers.get('etag') ?? '*' })).toContain(tag);
  expect(await text(`${previewOrigin}projects/a/steps`)).toContain(tag);
  expect(await text(`${previewOrigin}assets/app.js`)).toBe('export {};');
});

test('no page carries the embed once the socket is gone', async () => {
  await new Promise<void>((resolve) => daemon.close(() => resolve()));
  for (const url of [devOrigin, previewOrigin, `${previewOrigin}projects/a/steps`]) {
    expect(await text(url)).not.toContain('embed/notes.js');
  }
});
