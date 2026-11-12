import { expect, test } from '@playwright/test';

test.use({
  headless: true,
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  trace: 'off',
});

test('uses the pinned shell surfaces and keeps Studio’s accent and native inset', async ({ page }) => {
  await page.route((url) => url.pathname.startsWith('/api/'), (route) => route.fulfill({ status: 503 }));
  const revision = 'a'.repeat(64);
  await page.route('**/api/projects/shell/open', (route) => route.fulfill({ json: {
    configuration: { project: 'Shell', store: 'results' },
    snapshot: {
      project_id: 'shell', revision, presentation_revision: revision, semantic_root: revision,
      project: {}, presentation: {}, file_revisions: [],
    },
  }}));
  await page.route(/^https:\/\/fonts\./, (route) => route.abort());
  await page.goto('/?project=shell');

  const frame = page.locator('.shell-frame.studio-project-window');
  const header = frame.locator('[data-studio-focus-region="top"]');
  const controls = header.locator('.studio-window-bar__controls');
  const status = frame.locator('[data-studio-focus-region="bottom"]');
  const accent = () => header.evaluate((element) =>
    getComputedStyle(element).getPropertyValue('--studio-shell-accent').trim());
  await expect(frame).toHaveCSS('background-color', 'rgb(242, 242, 242)');
  await expect(status).toBeVisible();
  await expect(header).toHaveCSS('height', '48px');
  await expect(controls).toHaveCSS('padding-left', '8px');
  expect(await accent()).toBe('#5d4b87');

  await page.evaluate(() => {
    document.documentElement.dataset.theme = 'dark';
    document.documentElement.style.setProperty('--studio-native-controls', '80px');
  });
  await expect(frame).toHaveCSS('background-color', 'rgb(34, 34, 34)');
  await expect(controls).toHaveCSS('padding-left', '88px');
  expect(await accent()).toBe('#c1aedf');
});
