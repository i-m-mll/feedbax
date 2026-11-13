import { expect, test } from '@playwright/test';
import type { ModelDocument, ModelEditResponse } from '../../src/generated/studioContracts';
import { mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

function modelFixture(cards: number): ModelDocument {
  const components: NonNullable<ModelDocument['interior']['components']> = {};
  const wires: NonNullable<ModelDocument['interior']['wires']> = {};
  for (let index = 0; index < cards; index++) {
    const component = `sum-${index}`;
    components[component] = { kind: 'leaf', name: `Signal aggregation ${index + 1}`, type: 'feedbax.Sum' };
    for (const [offset, port] of ['a', 'b', 'c'].entries()) {
      const previous = index - offset - 1;
      wires[`wire-${index}-${port}`] = {
        source: previous < 0 ? { owner: 'inputs', port: 'target' }
          : { owner: 'component', component: `sum-${previous}`, port: 'output' },
        target: { owner: 'component', component, port },
      };
    }
  }
  return {
    step_seconds: 0.01, ports: { inputs: [{ id: 'target', name: 'reach_target', type: {} }] },
    interior: { components, wires },
  };
}

type Samples = { callbacks: number[]; updates: number[]; longTasks: number[]; stop(): void };
type SampleWindow = Window & { dragSamples?: Samples };

function gaps(samples: number[]) {
  return samples.slice(1).map((time, index) => time - samples[index]);
}

function statistics(samples: number[]) {
  const sorted = [...samples].sort((left, right) => left - right);
  return {
    samples: sorted.length,
    p50: sorted[Math.ceil(sorted.length * 0.5) - 1],
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
    maximum: sorted.at(-1),
    above16: sorted.filter((time) => time > 16).length,
    above20: sorted.filter((time) => time > 20).length,
    above50: sorted.filter((time) => time > 50).length,
  };
}

test('measures drag updates in the mounted Model at 20 and 100 cards', async ({ page, request, browser, browserName }, info) => {
  test.setTimeout(55_000);
  const result = [];
  for (const cards of [20, 100]) {
    const { snapshot, projectPath } = await seedManagedProject(request, [
      { kind: 'document', id: 'model', document: { schema: 'feedbax.spec.model', payload: modelFixture(cards) } },
      { kind: 'step', id: 'train', step: { kind: 'train', base: { inline: { graph: { document: 'model' } } } } },
    ]);
    await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Model', exact: true }).click();
    const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
    const selector = '.react-flow__node[data-id="sum-5"]';
    await expect(model.locator('.react-flow__node')).toHaveCount(cards + 2);
    await expect(model.locator('.react-flow__edge')).toHaveCount(cards * 3);
    const card = model.locator(selector);
    const edge = model.locator('.react-flow__edge[data-id="wire-5-a"] .react-flow__edge-path');
    const context = await page.context().newCDPSession(page);
    await context.send('Performance.enable');
    const samplesByDrag = [];
    for (let repetition = 0; repetition < 4; repetition++) {
      const prior = await edge.getAttribute('d');
      const box = await card.locator('header').boundingBox();
      if (!box) throw new Error('The mounted card is not measurable');
      const start = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
      await page.mouse.move(start.x, start.y);
      await page.evaluate(async (selector) => {
        const card = document.querySelector<HTMLElement>(selector);
        if (!card) throw new Error('The mounted card is missing');
        let current = card.style.transform;
        let frame: number;
        const callbacks: number[] = [], updates: number[] = [], longTasks: number[] = [];
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) longTasks.push(entry.duration);
        });
        observer.observe({ type: 'longtask' });
        function sample(time: number) {
          callbacks.push(time);
          if (card!.style.transform !== current) {
            current = card!.style.transform;
            updates.push(time);
          }
          frame = requestAnimationFrame(sample);
        }
        (window as SampleWindow).dragSamples = {
          callbacks, updates, longTasks,
          stop() { cancelAnimationFrame(frame); observer.disconnect(); },
        };
        frame = requestAnimationFrame(sample);
        await new Promise<void>((done) => requestAnimationFrame(() => done()));
      }, selector);
      const saved = page.waitForResponse((response) => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === `${projectPath}/model/edit`);
      await page.mouse.down();
      const before = (await context.send('Performance.getMetrics')).metrics;
      const roundTrip: number[] = [];
      const direction = repetition % 2 === 0 ? -1 : 1;
      let due = performance.now();
      for (let move = 1; move <= 50; move++) {
        await new Promise((done) => setTimeout(done, Math.max(0, due - performance.now())));
        const began = performance.now();
        await page.mouse.move(start.x + direction * move * 1.6, start.y + direction * move * 0.6);
        roundTrip.push(performance.now() - began);
        due = Math.max(due + 1000 / 60, began + 1000 / 60);
      }
      const after = (await context.send('Performance.getMetrics')).metrics;
      const timing = await page.evaluate(() => {
        const { dragSamples } = window as SampleWindow;
        if (!dragSamples) throw new Error('The drag sampler did not start');
        dragSamples.stop();
        return { callbacks: dragSamples.callbacks, updates: dragSamples.updates, longTasks: dragSamples.longTasks };
      });
      await page.mouse.up();
      const response = await saved;
      expect(response.ok(), await response.text()).toBe(true);
      const { result: edit } = await response.json() as ModelEditResponse;
      expect(edit.status).toBe('applied');
      expect(timing.updates.length).toBeGreaterThan(25);
      await expect(edge).not.toHaveAttribute('d', prior!);
      const chromeWork = Object.fromEntries(['ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration', 'TaskDuration'].map((name) => [
        name, 1000 * (after.find((item) => item.name === name)!.value - before.find((item) => item.name === name)!.value),
      ]));
      if (repetition > 0) samplesByDrag.push({ ...timing, roundTrip, chromeWork });
    }
    await context.detach();
    result.push({
      cards, connections: cards * 3,
      callbacks: statistics(samplesByDrag.flatMap(({ callbacks, updates }) =>
        gaps(callbacks.filter((time) => time >= updates[0] && time <= updates.at(-1)!)))),
      changedCard: statistics(samplesByDrag.flatMap(({ updates }) => gaps(updates))),
      roundTrip: statistics(samplesByDrag.flatMap((sample) => sample.roundTrip)),
      longTasks: samplesByDrag.flatMap((sample) => sample.longTasks),
      raw: samplesByDrag,
    });
  }
  const report = { browserName, version: browser.version(), slots: 1, result };
  await info.attach('mounted-model-drag', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
  console.log(JSON.stringify({ ...report, result: result.map(({ raw, ...summary }) => summary) }));
});
