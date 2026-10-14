import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vitest/config';

// GitHub Pages serves the site from /<repo>/.
const BASE = '/Movie-Recommendation/';

// GitHub Pages cannot set response headers, so the policy ships as a meta tag.
// It is added only to production builds because the dev server injects inline
// scripts for hot reload. frame-ancestors is ignored in meta tags, which is why
// it is absent here.
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
  'upgrade-insecure-requests',
].join('; ');

function contentSecurityPolicy(): Plugin {
  return {
    name: 'content-security-policy',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler: () => [
        {
          tag: 'meta',
          attrs: { 'http-equiv': 'Content-Security-Policy', content: CSP },
          injectTo: 'head-prepend',
        },
      ],
    },
  };
}

export default defineConfig({
  base: BASE,
  plugins: [react(), contentSecurityPolicy()],
  build: {
    target: 'es2022',
    sourcemap: false,
    // Keep everything as external files; inlined assets would need a looser CSP.
    assetsInlineLimit: 0,
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
