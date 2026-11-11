/// <reference types="vitest/config" />

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import type { Plugin } from 'vite';

const isE2E = process.env.VITE_FEEDBAX_E2E === '1';
const isDesktop = process.env.VITE_FEEDBAX_DESKTOP === '1';
const apiAddress = `127.0.0.1:${process.env.VITE_FEEDBAX_API_PORT ?? 8000}`;

const plotlyPackage = 'plotly.js-dist-min';
const desktopPlotlyAsset = 'vendor/plotly.mjs';

function desktopResources(): Plugin {
  return {
    name: 'desktop-resources',
    enforce: 'pre',
    resolveId(source) {
      if (source === plotlyPackage) {
        return { id: `/${desktopPlotlyAsset}`, external: true };
      }
    },
    generateBundle(_options, bundle) {
      const packages = new Map<string, { license?: string; notices: string[] }>();
      const visited = new Set<string>();
      function recordPackage(directory: string) {
        if (visited.has(directory)) return;
        visited.add(directory);
        const { name, version, license } = JSON.parse(readFileSync(path.join(directory, 'package.json'), 'utf8'));
        if (!name || !version) return;
        const notices = readdirSync(directory, { withFileTypes: true })
          .filter((entry) => entry.isFile() && /^(licen[cs]e|notice|copying)([-._]|$)/i.test(entry.name))
          .sort((a, b) => a.name.localeCompare(b.name))
          .map((entry) => readFileSync(path.join(directory, entry.name), 'utf8'));
        packages.set(`${name}@${version}`, { license, notices });
      }
      const plotly = path.join(__dirname, 'node_modules', plotlyPackage);
      recordPackage(plotly);
      this.emitFile({
        type: 'asset',
        fileName: desktopPlotlyAsset,
        source: Buffer.concat([
          Buffer.from('const module = { exports: {} };\n'),
          readFileSync(path.join(plotly, 'plotly.min.js')),
          Buffer.from('\nexport const { newPlot, purge } = module.exports;\n'),
        ]),
      });
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue;
        for (const [id, rendered] of Object.entries(output.modules)) {
          if (rendered.renderedLength === 0 || !id.includes('/node_modules/')) continue;
          let directory = path.dirname(id);
          while (directory.includes('/node_modules/') && !existsSync(path.join(directory, 'package.json'))) {
            directory = path.dirname(directory);
          }
          if (directory.includes('/node_modules/')) recordPackage(directory);
        }
      }
      this.emitFile({
        type: 'asset',
        fileName: 'third-party-notices.json',
        source: JSON.stringify(Object.fromEntries(Array.from(packages.entries()).sort()), null, 2) + '\n',
      });
    },
  };
}

export default defineConfig({
  cacheDir: process.env.FEEDBAX_VITE_CACHE_DIR,
  test: {
    // Transform Zod's ESM re-exports in Vitest instead of loading them through Bun's SSR interop.
    server: { deps: { inline: ['zod'] } },
    globals: true,
    environment: 'node',
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**'],
  },
  plugins: [
    react(),
    isDesktop ? desktopResources() : VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        skipWaiting: true,
        navigateFallback: '/index.html',
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        runtimeCaching: [
          {
            // Project state, job snapshots, and event streams must reach the live service.
            urlPattern: /^https?:\/\/[^/]+\/api\/(?:graphs|jobs|submissions|events|sessions|provider\/health)(?:[/?].*)?$/i,
            handler: 'NetworkOnly',
          },
          {
            // Training status, probes, etc. can tolerate brief staleness
            urlPattern: /^https?:\/\/.*\/api\/.*/i,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60, // 1 hour
              },
            },
          },
        ],
      },
      manifest: {
        name: 'Feedbax Studio',
        short_name: 'Feedbax',
        description: 'Neural network model builder and training environment',
        theme_color: '#18181b',
        background_color: '#f4f5f7',
        display: 'standalone',
        icons: [
          {
            src: '/icon.svg',
            sizes: 'any',
            type: 'image/svg+xml',
            purpose: 'any maskable',
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/recharts/')) {
            return 'recharts';
          }
          if (id.includes('/node_modules/@xyflow/')) {
            return 'xyflow';
          }
          if (id.includes('/node_modules/lucide-react/')) {
            return 'icons';
          }
          return undefined;
        },
      },
    },
  },
  server: {
    port: 3008,
    proxy: isE2E
      ? undefined
      : {
          '/api': {
            target: `http://${apiAddress}`,
            changeOrigin: true,
            configure(proxy) {
              proxy.on('proxyRes', (upstream, _request, downstream) => {
                if (!upstream.headers['content-type']?.startsWith('text/event-stream')) return;
                upstream.once('close', () => {
                  if (!upstream.complete && !downstream.writableEnded) downstream.destroy();
                });
              });
            },
          },
          '/ws': {
            target: `ws://${apiAddress}`,
            ws: true,
          },
        },
  },
});
