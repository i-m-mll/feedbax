/// <reference types="vitest/config" />

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { Agent } from 'node:http';
import type { Plugin } from 'vite';
import { buildSync } from 'esbuild';
import { appearanceBootEntry } from './entries';
import { mandibleInterfaceNotes } from './mandible-notes';
import { sourceStamps } from './source-stamps';

const isE2E = process.env.VITE_FEEDBAX_E2E === '1';
const isDesktop = process.env.VITE_FEEDBAX_DESKTOP === '1';
/**
 * A build or server for the owner's walkthrough, chosen by its launcher: each element says where it is written, and
 * pages load Mandible's notes toolbar. Every other build and server, the tests' included, has neither.
 */
const isWalkthrough = process.env.FEEDBAX_WALKTHROUGH === '1';
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

// The theme and density must be on the root before the first paint, so the boot script is inlined into the page.
function appearanceBoot(): Plugin {
  return {
    name: 'appearance-boot',
    transformIndexHtml() {
      const { outputFiles } = buildSync({
        entryPoints: [path.join(__dirname, appearanceBootEntry)],
        bundle: true, write: false, minify: true, format: 'iife', target: 'es2020',
      });
      return [{ tag: 'script', children: outputFiles[0].text, injectTo: 'head' }];
    },
  };
}

export default defineConfig({
  cacheDir: process.env.FEEDBAX_VITE_CACHE_DIR,
  // The standalone folder controller can load before a shell imports it.
  optimizeDeps: { include: ['immer'] },
  test: {
    // Transform Zod's ESM re-exports in Vitest instead of loading them through Bun's SSR interop.
    server: { deps: { inline: ['zod'] } },
    globals: true,
    environment: 'node',
    setupFiles: ['./src/test/setup.ts'],
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**'],
  },
  plugins: [
    appearanceBoot(),
    react(isWalkthrough ? { babel: { plugins: [sourceStamps] } } : {}),
    isWalkthrough ? mandibleInterfaceNotes() : [],
    isDesktop ? desktopResources() : [],
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      input: isDesktop
        ? path.resolve(__dirname, 'index.html')
        : { main: path.resolve(__dirname, 'index.html'), gallery: path.resolve(__dirname, 'gallery.html') },
      output: {
        manualChunks(id) {
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
            // Reuse backend connections; without an agent every proxied request closes both legs.
            agent: new Agent({ keepAlive: true }),
            configure(proxy) {
              // A stopped backend answers nothing; the page must see that, not the proxy's 500.
              proxy.on('error', (_error, request) => { request.socket.destroy(); });
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
