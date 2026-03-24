import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
  },
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        skipWaiting: true,
        navigateFallback: '/index.html',
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
  server: {
    port: 3008,
    proxy: {
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
