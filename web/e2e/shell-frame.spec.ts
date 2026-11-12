import { expect, test } from '@playwright/test';

test.use({
  headless: true,
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  trace: 'off',
  viewport: { width: 1280, height: 720 },
});

test('restores authored pane sizes after responsive folding and reload', async ({ page }) => {
  const storageKey = 'feedbax-studio-layout';
  await page.addInitScript((key) => {
    if (!localStorage.getItem(key)) {
      localStorage.setItem(key, JSON.stringify({
        version: 3,
        state: { bottomHeight: 320, leftSidebarWidth: 250, leftSidebarVisible: true },
      }));
    }
  }, storageKey);
  await page.route((url) => url.pathname.startsWith('/api/'), (route) => route.fulfill({ status: 503 }));
  await page.route(/^https:\/\/fonts\./, (route) => route.abort());
  await page.goto('/');

  const bottom = page.locator('#studio-bottom-pane');
  const palette = page.getByRole('complementary', { name: 'Component palette' });
  const separator = page.getByRole('separator', { name: 'Resize bottom pane' });
  const stored = () => page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).state, storageKey);
  await expect(bottom).toHaveCSS('height', '320px');
  await expect(palette).toHaveCSS('width', '250px');
  await expect(separator).toHaveAttribute('aria-orientation', 'horizontal');
  await separator.press('ArrowUp');
  await expect(separator).toHaveAttribute('aria-valuenow', '328');

  const bounds = (await separator.boundingBox())!;
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y - 24);
  await expect.poll(() => separator.getAttribute('aria-valuenow')).not.toBe('328');
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(separator).toHaveAttribute('aria-valuenow', '328');
  expect((await stored()).bottomHeight).toBe(328);

  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;
  await page.mouse.move(centerX, centerY);
  await page.mouse.down();
  await page.mouse.move(centerX, centerY - 20);
  expect((await stored()).bottomHeight).toBe(328);
  await page.mouse.up();
  await expect(separator).toHaveAttribute('aria-valuenow', '348');
  expect((await stored()).bottomHeight).toBe(348);

  const url = page.url();
  await page.setViewportSize({ width: 360, height: 180 });
  await expect(bottom).toHaveCSS('height', '44px');
  await expect(palette).toBeHidden();
  expect(await stored()).toMatchObject({ bottomHeight: 348, leftSidebarWidth: 250, leftSidebarVisible: true });
  expect(page.url()).toBe(url);
  await page.reload();
  await expect(bottom).toHaveCSS('height', '44px');
  await page.setViewportSize({ width: 1280, height: 720 });
  await expect(bottom).toHaveCSS('height', '348px');
  await expect(palette).toHaveCSS('width', '250px');

  await page.getByRole('button', { name: 'Open project' }).focus();
  await expect(page.getByRole('tooltip')).toHaveText('Open project');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toBeHidden();
});
