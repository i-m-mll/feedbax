import { expect, test, type Page } from '@playwright/test';

test.use({
  headless: true,
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  trace: 'off',
});

async function openShell(page: Page) {
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
  // The window drawn while the project opens is replaced by the opened project's.
  await expect(page.getByRole('heading', { level: 1, name: 'Shell' })).toBeVisible();
}

test('uses the pinned shell surfaces, Studio’s accent and the shell’s title bar', async ({ page }) => {
  await openShell(page);

  const frame = page.locator('.shell-frame.studio-project-window');
  const header = frame.locator('[data-studio-focus-region="top"]');
  const logo = header.locator('.shell-titlebar__logo > svg');
  const firstControl = header.locator('.shell-titlebar__leading > :nth-child(2)');
  const status = frame.locator('[data-studio-focus-region="bottom"]');
  const accent = () => header.evaluate((element) =>
    getComputedStyle(element).getPropertyValue('--studio-shell-accent').trim());
  const geometry = () => frame.evaluate((root) => {
    const style = getComputedStyle(root);
    const rail = Number.parseFloat(style.gridTemplateColumns);
    const header = root.querySelector('[data-studio-focus-region="top"]')!.getBoundingClientRect();
    const box = root.querySelector('.shell-titlebar__logo')!.getBoundingClientRect();
    const logo = root.querySelector('.shell-titlebar__logo > svg')!.getBoundingClientRect();
    const leading = root.querySelector('.shell-titlebar__leading')!;
    const first = leading.children[1].getBoundingClientRect();
    const leftmost = Math.min(...Array.from(root.querySelectorAll('[data-studio-focus-region="top"] :is(button, svg, h1)'))
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
  await expect(firstControl).toBeVisible();
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

test('the map drawer meets the main area with the divider, not the chrome edge', async ({ page }) => {
  await openShell(page);
  const drawer = page.locator('.studio-window-side').first();

  for (const theme of ['light', 'dark']) {
    const tokens = await page.evaluate((theme) => {
      document.documentElement.dataset.theme = theme;
      const resolve = (token: string) => {
        const probe = document.createElement('div');
        probe.style.color = `var(${token})`;
        document.body.append(probe);
        const colour = getComputedStyle(probe).color;
        probe.remove();
        return colour;
      };
      return { divider: resolve('--border-muted'), edge: resolve('--app-boundary-line-color') };
    }, theme);
    expect(tokens.divider).not.toBe(tokens.edge);
    await expect(drawer).toHaveCSS('border-inline-end-color', tokens.divider);
  }
});

test('the chrome edge stands out from the chrome in both themes', async ({ page }) => {
  await openShell(page);
  const edge = page.locator('.shell-frame.studio-project-window > .shell-frame__content');
  await expect(edge).toBeVisible();

  for (const theme of ['light', 'dark']) {
    const { width, contrast } = await edge.evaluate((element, theme) => {
      document.documentElement.dataset.theme = theme;
      const pixel = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!;
      const luminance = (colour: string) => {
        pixel.fillStyle = colour;
        pixel.fillRect(0, 0, 1, 1);
        const [r, g, b] = Array.from(pixel.getImageData(0, 0, 1, 1).data.slice(0, 3), (value) => {
          const channel = value / 255;
          return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      };
      const after = getComputedStyle(element, '::after');
      const line = luminance(after.borderTopColor);
      const chrome = luminance(getComputedStyle(element.parentElement!).backgroundColor);
      return { width: after.borderTopWidth, contrast: (Math.max(line, chrome) + 0.05) / (Math.min(line, chrome) + 0.05) };
    }, theme);
    expect(width, theme).toBe('1px');
    expect(contrast, theme).toBeGreaterThanOrEqual(1.25);
  }
});
