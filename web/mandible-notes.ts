import { existsSync } from 'node:fs';
import { get, type OutgoingHttpHeaders, type ServerResponse } from 'node:http';
import { homedir } from 'node:os';
import path from 'node:path';
import type { Connect, Logger, Plugin } from 'vite';

export const defaultMandibleSocket = process.env.MANDIBLE_SOCKET
  ?? path.join(homedir(), '.cache/mandible/local-api/production.sock');

type Origin = { ok: true; origin: string } | { ok: false; reason: string };

function daemonOrigin(socketPath: string): Promise<Origin> {
  return new Promise((resolve) => {
    const request = get({ socketPath, path: '/api/instance', timeout: 1000 }, (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        if (response.statusCode !== 200) return resolve({ ok: false, reason: `GET /api/instance answered ${response.statusCode}` });
        try {
          const origin = JSON.parse(body).instance?.public_origin;
          resolve(typeof origin === 'string' ? { ok: true, origin } : { ok: false, reason: 'GET /api/instance has no instance.public_origin' });
        } catch (thrown) {
          resolve({ ok: false, reason: `GET /api/instance is not JSON: ${thrown}` });
        }
      });
    });
    request.on('timeout', () => request.destroy(new Error('timed out')));
    request.on('error', (error) => resolve({ ok: false, reason: error.message }));
  });
}

async function withEmbed(html: string, socketPath: string, logger: Logger): Promise<string> {
  if (!existsSync(socketPath)) return html;
  const daemon = await daemonOrigin(socketPath);
  if (!daemon.ok) {
    logger.warn(`Mandible interface notes are off: ${socketPath}: ${daemon.reason}`);
    return html;
  }
  return html.replace('</head>', `<script type="module" src="${daemon.origin}/embed/notes.js"></script></head>`);
}

const pagePath = /(?:\.html|\/[^./]*)$/;

// Vite preview serves pages from three middlewares (static files, the SPA fallback, the HTML middleware), so pages are rewritten on their way out.
function rewritePages(rewrite: (html: string) => Promise<string>): Connect.NextHandleFunction {
  return (request, response, next) => {
    if (request.method !== 'GET' || !pagePath.test(request.url!.split('?')[0])) return next();
    // Every load must ask the daemon again, so the page is never answered from the browser's cache.
    delete request.headers['if-none-match'];
    delete request.headers['if-modified-since'];
    // Preview's compression runs between this rewrite and the file, so pages are sent uncompressed.
    delete request.headers['accept-encoding'];
    const { write, end } = response;
    const chunks: Buffer[] = [];
    let buffering: boolean | undefined;
    const isPage = () => (buffering ??= response.statusCode === 200 && String(response.getHeader('content-type')).startsWith('text/html'));
    response.writeHead = ((status: number, headers?: OutgoingHttpHeaders) => {
      response.statusCode = status;
      for (const [name, value] of Object.entries(headers ?? {})) if (value !== undefined) response.setHeader(name, value);
      return response;
    }) as ServerResponse['writeHead'];
    response.write = ((...args: [Buffer | string, ...unknown[]]) => {
      if (!isPage()) return write.apply(response, args as never);
      chunks.push(Buffer.from(args[0]));
      return true;
    }) as ServerResponse['write'];
    response.end = ((...args: [unknown?, ...unknown[]]) => {
      if (!isPage()) return end.apply(response, args as never);
      if (typeof args[0] === 'string' || args[0] instanceof Uint8Array) chunks.push(Buffer.from(args[0]));
      rewrite(Buffer.concat(chunks).toString('utf8')).then((html) => {
        response.removeHeader('etag');
        response.removeHeader('last-modified');
        response.setHeader('content-length', Buffer.byteLength(html));
        response.setHeader('cache-control', 'no-cache');
        end.call(response, html, 'utf8');
      }, next);
      return response;
    }) as ServerResponse['end'];
    next();
  };
}

// Studio's local Vite server, dev or preview, loads Mandible's interface-notes toolbar from the daemon whose local socket exists.
export function mandibleInterfaceNotes(socketPath = defaultMandibleSocket): Plugin {
  let logger: Logger;
  return {
    name: 'mandible-interface-notes',
    // Vite resolves `vite preview` as `serve` too, so only `vite build` leaves the plugin out.
    apply: 'serve',
    configResolved(config) { logger = config.logger; },
    transformIndexHtml(html) { return withEmbed(html, socketPath, logger); },
    configurePreviewServer(server) {
      server.middlewares.use(rewritePages((html) => withEmbed(html, socketPath, logger)));
    },
  };
}
