import { existsSync } from 'node:fs';
import { get } from 'node:http';
import { homedir } from 'node:os';
import path from 'node:path';
import type { Plugin } from 'vite';

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

// Dev-only: Mandible's interface-notes toolbar, loaded from the daemon whose local socket exists.
export function mandibleInterfaceNotes(socketPath = defaultMandibleSocket): Plugin {
  return {
    name: 'mandible-interface-notes',
    apply: 'serve',
    async transformIndexHtml(_html, context) {
      if (!existsSync(socketPath)) return [];
      const daemon = await daemonOrigin(socketPath);
      if (!daemon.ok) {
        context.server?.config.logger.warn(`Mandible interface notes are off: ${socketPath}: ${daemon.reason}`);
        return [];
      }
      return [{ tag: 'script', attrs: { type: 'module', src: `${daemon.origin}/embed/notes.js` }, injectTo: 'head' }];
    },
  };
}
