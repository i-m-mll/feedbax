import { expect, type Locator, type Page, type TestInfo } from '@playwright/test';
import axe from 'axe-core';
import type { AxeResults, Result } from 'axe-core';

function findings(results: Result[]) {
  return results.map(({ id, impact, nodes }) => ({
    id, impact,
    nodes: nodes.map(({ target, failureSummary, any, all }) => ({
      target, failureSummary,
      checks: [...any, ...all].map(({ id, data, message }) => ({ id, data, message })),
    })),
  }));
}

export async function scanStudio(page: Page, info: TestInfo, name: string) {
  if (!await page.evaluate(() => 'axe' in window)) await page.addScriptTag({ content: axe.source });
  const result = await page.evaluate(async (): Promise<AxeResults> => {
    const { axe } = window as Window & { axe: typeof import('axe-core') };
    return axe.run(document, {
      runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
      resultTypes: ['violations', 'incomplete'],
    });
  });
  const violations = findings(result.violations);
  await info.attach(`axe-${name}`, {
    body: JSON.stringify({
      engine: result.testEngine, tags: result.toolOptions.runOnly,
      violations, incomplete: findings(result.incomplete),
      passed: result.passes.map(({ id }) => id),
    }, null, 2),
    contentType: 'application/json',
  });
  expect.soft(violations.map(({ id, nodes }) => ({ id, targets: nodes.map(({ target }) => target) })), name).toEqual([]);
}

export async function expectVisibleFocus(control: Locator) {
  await expect(control).toBeFocused();
  await expect(control).toBeInViewport({ ratio: 0.75 });
  await expect(control).toHaveCSS('outline-style', 'solid');
  await expect(control).toHaveCSS('outline-width', '2px');
}

export async function expectStudioMapContrast(page: Page) {
  const ratios = await page.locator('[data-map-canvas]').evaluate((map) => {
    const probe = document.createElement('span');
    map.append(probe);
    const context = document.createElement('canvas').getContext('2d')!;
    function luminance(color: string) {
      probe.style.color = color;
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = getComputedStyle(probe).color;
      context.fillRect(0, 0, 1, 1);
      const [red, green, blue, alpha] = context.getImageData(0, 0, 1, 1).data;
      if (alpha !== 255) throw new Error(`Expected an opaque Studio color: ${color}`);
      const channels = [red, green, blue].map((byte) => {
        const value = byte / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    }
    const surfaces = ['surface-main', 'surface-chrome'].map((name) => luminance(`var(--${name})`));
    const tokens = [
      ...['idle', 'neutral', 'warning', 'running', 'success', 'failure'].map((status) => `studio-status-${status}`),
      'studio-kind-train', 'studio-kind-evaluate', 'shell-focus-color',
    ];
    const result = Object.fromEntries(tokens.map((name) => {
      const color = luminance(`var(--${name})`);
      const ratio = (surface: number) => (Math.max(color, surface) + 0.05) / (Math.min(color, surface) + 0.05);
      return [name, Math.min(...surfaces.map(ratio))];
    }));
    probe.remove();
    return result;
  });
  for (const [token, ratio] of Object.entries(ratios)) expect(ratio, token).toBeGreaterThanOrEqual(3);
}
