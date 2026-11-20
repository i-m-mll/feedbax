import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import type { Plugin, ProxyOptions } from 'vite';

type Proxy = Parameters<NonNullable<ProxyOptions['configure']>>[0];

const prefix = '/__mandible';

export const defaultMandibleSocket = process.env.MANDIBLE_SOCKET
  ?? path.join(homedir(), '.cache/mandible/local-api/production.sock');

// An event stream whose upstream ends early must end for the page too, so the page reconnects.
export function endStreamsWithUpstream(proxy: Proxy) {
  proxy.on('proxyRes', (upstream, _request, downstream) => {
    if (!upstream.headers['content-type']?.startsWith('text/event-stream')) return;
    upstream.once('close', () => {
      if (!upstream.complete && !downstream.writableEnded) downstream.destroy();
    });
  });
}

// Dev-only: Mandible's interface-notes toolbar, reached through a same-origin proxy onto the daemon's local socket.
export function mandibleInterfaceNotes(socketPath = defaultMandibleSocket): Plugin {
  return {
    name: 'mandible-interface-notes',
    apply: 'serve',
    config: () => ({
      server: {
        proxy: {
          [`${prefix}/`]: {
            // http-proxy copies socketPath from the target; host and port are unused on a socket.
            target: { socketPath } as ProxyOptions['target'],
            rewrite: (url) => url.slice(prefix.length),
            headers: { 'x-mandible-embed': '1' },
            configure: endStreamsWithUpstream,
          },
        },
      },
    }),
    transformIndexHtml() {
      if (!existsSync(socketPath)) return [];
      return [{
        tag: 'script',
        attrs: {
          defer: true,
          src: `${prefix}/dist/interface-notes-embed.js`,
          'data-app': 'feedbax-studio',
          'data-project': 'feedbax',
        },
        injectTo: 'head',
      }];
    },
  };
}
