import { expect, test, type Page } from '@playwright/test';
import type {
  GraphSpec,
  NodeUIState,
  StudioPersistenceDocument,
} from '../../src/generated/studioContracts';
import { openSavedProject } from './projectControls';

const measure = process.env.FEEDBAX_STUDIO_MEASURE === '1';

async function scriptSampler(page: Page) {
  const session = await page.context().newCDPSession(page);
  await session.send('Performance.enable');

  async function scriptTimeMs() {
    const { metrics } = await session.send('Performance.getMetrics');
    const duration = metrics.find(({ name }) => name === 'ScriptDuration');
    if (!duration) throw new Error('Chromium did not report ScriptDuration');
    return duration.value * 1000;
  }

  return async (action: () => Promise<void>) => {
    const before = await scriptTimeMs();
    await action();
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));
    return (await scriptTimeMs()) - before;
  };
}

function summarize(samples: number[]) {
  const sorted = samples.toSorted((left, right) => left - right);
  return {
    samples: sorted.length,
    median: sorted[Math.floor(sorted.length / 2)],
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
    max: sorted.at(-1),
  };
}

async function openMeasurementGraph(page: Page, componentCount: number) {
  await page.goto('/');
  await page.getByRole('button', { name: 'New project', exact: true }).click();
  const name = page.getByPlaceholder('Tab name...');
  await name.fill('Measurement seed');
  await name.press('Enter');
  await page.getByRole('button', { name: 'Runtime', exact: true }).click();
  const save = page.waitForRequest((request) =>
    new URL(request.url()).pathname === '/api/graphs' && request.method() === 'POST'
      && Object.values((request.postDataJSON() as StudioPersistenceDocument).graph?.nodes ?? {})
        .some((node) => node.type === 'feedbax.Linear'),
  );
  await page.getByTitle('Add feedbax.Linear', { exact: true }).click();
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  const seed = (await save).postDataJSON() as StudioPersistenceDocument;
  const component = Object.values(seed.graph?.nodes ?? {})[0];
  if (!component || !seed.graph?.metadata || !seed.workspace_document) {
    throw new Error('The authored Linear did not produce a complete save request');
  }

  const label = `Responsiveness ${componentCount}`;
  const nodes: NonNullable<GraphSpec['nodes']> = {};
  const nodeStates: Record<string, NodeUIState> = {};
  for (let index = 0; index < componentCount; index += 1) {
    const id = `linear_${index + 1}`;
    nodes[id] = component;
    nodeStates[id] = {
      position: { x: 90 + (index % 5) * 210, y: 75 + Math.floor(index / 5) * 180 },
    };
  }
  const project: StudioPersistenceDocument = {
    ...seed,
    graph: {
      ...seed.graph,
      nodes,
      metadata: { ...seed.graph.metadata, name: label },
    },
    workspace_document: {
      ...seed.workspace_document,
      graph_ui_state: { node_states: nodeStates, viewport: { x: 0, y: 0, zoom: 1 } },
    },
  };
  const created = await page.request.post('/api/graphs', { data: project });
  expect(created.ok(), `Project creation returned ${created.status()}`).toBe(true);
  const { data: { id } } = await created.json() as { data: { id: string } };
  await openSavedProject(page, label);
  await expect(page.locator('.react-flow__node')).toHaveCount(componentCount);
  return id;
}

async function visibleComponentId(page: Page) {
  const id = await page.locator('.react-flow').evaluate((canvas) => {
    const view = canvas.getBoundingClientRect();
    const nodes = [...canvas.querySelectorAll<HTMLElement>('.react-flow__node[data-id]')];
    return nodes.find((node) => {
      const bounds = node.getBoundingClientRect();
      return bounds.left >= view.left && bounds.top >= view.top
        && bounds.right + 48 < view.right && bounds.bottom + 24 < view.bottom;
    })?.dataset.id;
  });
  if (!id) throw new Error('The graph has no component with room to drag inside the canvas');
  return id;
}

test.describe('Studio responsiveness baseline', () => {
  test.skip(!measure, 'Run with FEEDBAX_STUDIO_MEASURE=1');

  for (const componentCount of [20, 100]) {
    test(`edit and drag with ${componentCount} components`, async ({ page }, testInfo) => {
      const id = await openMeasurementGraph(page, componentCount);
      const nodeId = await visibleComponentId(page);
      const node = page.getByTestId(`rf__node-${nodeId}`);
      await node.click();
      const properties = page.getByRole('button', { name: 'Show properties panel', exact: true });
      if (await properties.isVisible()) await properties.click();
      const size = page.getByRole('textbox', { name: /^output size\b/i });
      await expect(size).toBeVisible();

      const sample = await scriptSampler(page);
      await size.fill('');
      const keystrokeMs = [];
      for (const digit of '123') keystrokeMs.push(await sample(() => size.press(digit)));
      await expect(size).toHaveValue('123');

      const committed = page.waitForResponse((response) => {
        const request = response.request();
        return new URL(response.url()).pathname === `/api/graphs/${id}`
          && request.method() === 'PUT'
          && request.postDataJSON().graph?.nodes?.[nodeId]?.params?.output_size === 123;
      });
      const commitMs = await sample(() => size.press('Enter'));
      await page.getByRole('button', { name: 'Save', exact: true }).click();
      const saved = await committed;
      expect(saved.ok(), `Project save returned ${saved.status()}`).toBe(true);

      const origin = await node.boundingBox();
      if (!origin) throw new Error('Linear is outside the canvas');
      await page.mouse.move(origin.x + origin.width / 2, origin.y + 16);
      await page.mouse.down();
      const dragStepMs = [];
      try {
        for (let step = 1; step <= 12; step += 1) {
          dragStepMs.push(await sample(() =>
            page.mouse.move(origin.x + origin.width / 2 + step * 4, origin.y + 16 + step * 2),
          ));
        }
      } finally {
        await page.mouse.up();
      }
      await expect.poll(async () => (await node.boundingBox())?.x).toBeGreaterThan(origin.x);
      await testInfo.attach('script-duration-ms', {
        contentType: 'application/json',
        body: JSON.stringify({
          componentCount,
          keystroke: summarize(keystrokeMs),
          commit: commitMs,
          dragStep: summarize(dragStepMs),
        }, null, 2),
      });
    });
  }

  test('measure playback frame intervals', async ({ page, headless }, testInfo) => {
    test.skip(headless, 'Playback timing requires a visible browser (--headed)');
    await page.goto('/');
    await page.getByRole('button', { name: 'Workspace', exact: true }).click();
    await page.getByRole('button', { name: 'playback', exact: true }).click();
    const play = page.getByTitle('Play', { exact: true });
    await expect(play).toBeEnabled();
    await play.click();
    const intervals = await page.evaluate(() => new Promise<number[]>((resolve, reject) => {
      if (document.visibilityState !== 'visible') {
        reject(new Error('Show the playback window before measuring'));
        return;
      }
      const frameTimes: number[] = [];
      const collect = (now: number) => {
        frameTimes.push(now);
        if (frameTimes.length <= 60) requestAnimationFrame(collect);
        else resolve(frameTimes.slice(1).map((time, index) => time - frameTimes[index]));
      };
      requestAnimationFrame(collect);
    }));
    await expect(page.getByTitle('Pause', { exact: true })).toBeVisible();
    await expect.poll(async () => Number(
      await page.getByRole('slider', { name: 'Playback position' }).inputValue(),
    )).toBeGreaterThan(0);
    await testInfo.attach('playback-frame-interval-ms', {
      contentType: 'application/json',
      body: JSON.stringify(summarize(intervals), null, 2),
    });
  });
});
