import { expect, test } from '@playwright/test';

test.use({
  headless: true,
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  trace: 'off',
});

test('uses the pinned shell surfaces, Studio’s accent and the shell’s title bar', async ({ page }) => {
  await page.route((url) => url.pathname.startsWith('/api/'), (route) => route.fulfill({ status: 503 }));
  const revision = 'a'.repeat(64);
  await page.route('**/api/projects/shell/open', (route) => route.fulfill({ json: {
    configuration: { project: 'Shell', store: 'results' },
    snapshot: {
      project_id: 'shell', revision, presentation_revision: revision, semantic_root: revision,
      project: {}, presentation: {}, file_revisions: [],
    },
  }}));
  await page.goto('/?project=shell');

  const frame = page.locator('.shell-frame.studio-project-window');
  const header = frame.locator('[data-studio-focus-region="top"]');
  const logo = header.locator('.shell-titlebar__logo > img');
  const firstState = header.locator('.shell-titlebar__leading > button').first();
  const status = frame.locator('[data-studio-focus-region="bottom"]');
  const accent = () => header.evaluate((element) =>
    getComputedStyle(element).getPropertyValue('--studio-shell-accent').trim());
  const geometry = () => frame.evaluate((root) => {
    const style = getComputedStyle(root);
    const rail = Number.parseFloat(style.gridTemplateColumns);
    const header = root.querySelector('[data-studio-focus-region="top"]')!.getBoundingClientRect();
    const box = root.querySelector('.shell-titlebar__logo')!.getBoundingClientRect();
    const logo = root.querySelector('.shell-titlebar__logo > img')!.getBoundingClientRect();
    const leading = root.querySelector('.shell-titlebar__leading')!;
    const first = leading.querySelector('button')!.getBoundingClientRect();
    const leftmost = Math.min(...Array.from(root.querySelectorAll('[data-studio-focus-region="top"] :is(button, img, h1)'))
      .map((element) => element.getBoundingClientRect()).filter((rect) => rect.width > 0).map((rect) => rect.left));
    return {
      headerHeight: header.height, railCentre: rail / 2, logoLeft: logo.left, logoCentre: logo.left + logo.width / 2,
      logoBoxRight: box.right, leftmost, firstLeft: first.left,
      offset: Number.parseFloat(style.getPropertyValue('--shell-titlebar-offset')),
    };
  });

  await expect(frame).toHaveCSS('background-color', 'rgb(242, 242, 242)');
  await expect(status).toBeVisible();
  await expect(logo).toBeVisible();
  await expect(firstState).toBeVisible();
  expect(await accent()).toBe('#5d4b87');

  const browser = await geometry();
  expect(browser.headerHeight).toBe(42);
  expect(Math.abs(browser.logoCentre - browser.railCentre)).toBeLessThanOrEqual(0.5);
  expect(browser.leftmost).toBe(browser.logoLeft);
  expect(browser.firstLeft).toBeCloseTo(browser.logoBoxRight + browser.offset, 1);

  await page.evaluate(() => {
    document.documentElement.dataset.theme = 'dark';
    document.documentElement.dataset.shellWindowControls = 'overlay';
    document.documentElement.style.setProperty('--shell-window-controls-reservation', '88px');
  });
  await expect(frame).toHaveCSS('background-color', 'rgb(34, 34, 34)');
  expect(await accent()).toBe('#c1aedf');
  const overlay = await geometry();
  expect(overlay.logoLeft).toBeCloseTo(88, 1);
  expect(overlay.firstLeft).toBeCloseTo(overlay.logoBoxRight + overlay.offset, 1);

  await page.evaluate(() => {
    document.documentElement.dataset.shellWindowControls = 'none';
  });
  const restored = await geometry();
  expect(Math.abs(restored.logoCentre - restored.railCentre)).toBeLessThanOrEqual(0.5);
});
