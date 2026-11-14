import { expect, type Page, type TestInfo } from '@playwright/test';
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
  expect.soft(violations, name).toEqual([]);
}
