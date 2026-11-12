import { expect, test } from '@playwright/test';

test.use({
  headless: true,
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  trace: 'off',
});

test('uses the pinned shell surfaces and keeps Studio’s accent and native inset', async ({ page }) => {
  await page.route('**/api/**', (route) => route.fulfill({ status: 503 }));
  await page.route(/^https:\/\/fonts\./, (route) => route.abort());
  await page.goto('/');

  const header = page.locator('header').filter({ has: page.getByAltText('feedbax studio logo') });
  const status = page.locator('footer').filter({ hasText: 'Connected' });
  const accent = () => header.evaluate((element) =>
    getComputedStyle(element).getPropertyValue('--studio-shell-accent').trim());
  await expect(header).toHaveCSS('background-color', 'rgb(242, 242, 242)');
  await expect(status).toHaveCSS('background-color', 'rgb(242, 242, 242)');
  await expect(header).toHaveCSS('height', '48px');
  await expect(header).toHaveCSS('padding-left', '12px');
  expect(await accent()).toBe('#5d4b87');

  await page.evaluate(() => {
    document.documentElement.dataset.theme = 'dark';
    document.documentElement.style.setProperty('--studio-native-controls', '80px');
  });
  await expect(header).toHaveCSS('background-color', 'rgb(34, 34, 34)');
  await expect(status).toHaveCSS('background-color', 'rgb(34, 34, 34)');
  await expect(header).toHaveCSS('padding-left', '92px');
  expect(await accent()).toBe('#c1aedf');
});
