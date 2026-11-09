/// <reference types="vitest/config" />

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import type { Plugin } from 'vite';

const isE2E = process.env.VITE_FEEDBAX_E2E === '1';
const isDesktop = process.env.VITE_FEEDBAX_DESKTOP === '1';

function desktopNotices(): Plugin {
  return {
    name: 'desktop-notices',
    generateBundle(_options, bundle) {
      const packages = new Map<string, { license?: string; notices: string[] }>();
      const visited = new Set<string>();
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue;
        for (const [id, rendered] of Object.entries(output.modules)) {
          if (rendered.renderedLength === 0 || !id.includes('/node_modules/')) continue;
          let directory = path.dirname(id);
          while (directory.includes('/node_modules/') && !existsSync(path.join(directory, 'package.json'))) {
            directory = path.dirname(directory);
          }
          if (!directory.includes('/node_modules/') || visited.has(directory)) continue;
          visited.add(directory);
          const { name, version, license } = JSON.parse(readFileSync(path.join(directory, 'package.json'), 'utf8'));
          if (!name || !version) continue;
          const notices = readdirSync(directory, { withFileTypes: true })
            .filter((entry) => entry.isFile() && /^(licen[cs]e|notice|copying)(\.|$)/i.test(entry.name))
            .sort((a, b) => a.name.localeCompare(b.name))
            .map((entry) => readFileSync(path.join(directory, entry.name), 'utf8'));
          packages.set(`${name}@${version}`, { license, notices });
        }
      }
      this.emitFile({
        type: 'asset',
        fileName: 'third-party-notices.json',
        source: JSON.stringify(Object.fromEntries([...packages].sort()), null, 2) + '\n',
      });
    },
  };
}

export default defineConfig({
  test: {
    // Transform Zod's ESM re-exports in Vitest instead of loading them through Bun's SSR interop.
    server: { deps: { inline: ['zod'] } },
    globals: true,
    environment: 'node',
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**'],
  },
  plugins: [
    react(),
    isDesktop ? desktopNotices() : VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        skipWaiting: true,
        navigateFallback: '/index.html',
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        runtimeCaching: [
          {
            // Graph state must never be served from cache (Cache-Control: no-store is ignored by NetworkFirst)
            urlPattern: /^https?:\/\/.*\/api\/graphs(\/.*)?$/i,
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
            target: 'http://localhost:8000',
            changeOrigin: true,
          },
          '/ws': {
            target: 'ws://localhost:8000',
            ws: true,
          },
        },
  },
});
