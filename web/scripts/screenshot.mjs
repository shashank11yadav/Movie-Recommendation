// Captures docs/screenshot.png for the README from the production build.
// Usage: npm run build && npm run preview (in another shell) && node scripts/screenshot.mjs [outfile]
import { chromium } from '@playwright/test';

const url = process.env.SCREENSHOT_URL ?? 'http://localhost:5101/Movie-Recommendation/';
const out = process.argv[2] ?? '../docs/screenshot.png';
const scheme = process.env.SCREENSHOT_SCHEME === 'dark' ? 'dark' : 'light';
const width = Number(process.env.SCREENSHOT_WIDTH ?? 1100);

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width, height: 1000 },
  deviceScaleFactor: 1,
  colorScheme: scheme,
  reducedMotion: 'reduce',
});
await page.goto(`${url}#liked=1214,541`);
await page.getByRole('heading', { level: 3 }).first().waitFor();
await page.screenshot({ path: out, fullPage: process.env.SCREENSHOT_FULL === '1' });
await browser.close();
