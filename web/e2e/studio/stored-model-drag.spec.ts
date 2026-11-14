import { expect, test, type APIRequestContext, type CDPSession, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, ModelFieldResponse, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

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

type EventSample = { name: string; duration: number; inputDelay: number; processing: number };
type Samples = { timeOrigin: number; callbacks: number[]; updates: number[]; longTasks: number[]; events: EventSample[] };
type SampleWindow = Window & { finishStudioSample?: () => Samples };

async function startSamples(target: Locator, kind: 'drag' | 'field' | 'scroll', label: string) {
  await target.evaluate(async (element, { kind, label }) => {
    const state = window as SampleWindow;
    if (state.finishStudioSample) throw new Error('A Studio measurement is already running');
    const node = element as HTMLElement;
    const value = () => {
      if (kind === 'drag') return node.style.transform;
      if (kind === 'field') return (node as HTMLInputElement).value;
      return node.scrollTop;
    };
    const samples: Samples = { timeOrigin: performance.timeOrigin, callbacks: [], updates: [], longTasks: [], events: [] };
    const tasks = (entries: PerformanceEntry[]) => entries.forEach(({ duration }) => samples.longTasks.push(duration));
    const events = (entries: PerformanceEntry[]) => {
      for (const entry of entries as PerformanceEventTiming[]) {
        if (entry.target === element && entry.name === 'keydown') samples.events.push({
          name: entry.name, duration: entry.duration, inputDelay: entry.processingStart - entry.startTime,
          processing: entry.processingEnd - entry.processingStart,
        });
      }
    };
    const taskObserver = new PerformanceObserver((list) => tasks(list.getEntries()));
    taskObserver.observe({ type: 'longtask' });
    let eventObserver: PerformanceObserver | undefined;
    if (kind === 'field' && PerformanceObserver.supportedEntryTypes.includes('event')) {
      eventObserver = new PerformanceObserver((list) => events(list.getEntries()));
      const options: PerformanceObserverInit & { durationThreshold: number } = { type: 'event', durationThreshold: 16 };
      eventObserver.observe(options);
    }
    performance.mark(`studio-${label}-start`);
    let current = value();
    let frame = requestAnimationFrame(function sample(time) {
      samples.callbacks.push(time);
      const next = value();
      if (next !== current) {
        samples.updates.push(time);
        current = next;
      }
      frame = requestAnimationFrame(sample);
    });
    state.finishStudioSample = () => {
      cancelAnimationFrame(frame);
      performance.mark(`studio-${label}-end`);
      tasks(taskObserver.takeRecords());
      taskObserver.disconnect();
      if (eventObserver) {
        events(eventObserver.takeRecords());
        eventObserver.disconnect();
      }
      delete state.finishStudioSample;
      return samples;
    };
    await new Promise<void>((done) => requestAnimationFrame(() => done()));
  }, { kind, label });
}

async function endSamples(page: Page) {
  return page.evaluate(() => {
    const { finishStudioSample } = window as SampleWindow;
    if (!finishStudioSample) throw new Error('Studio measurement did not start');
    return finishStudioSample();
  });
}

async function chromeMetrics(context: CDPSession) {
  const { metrics } = await context.send('Performance.getMetrics');
  return new Map(metrics.map(({ name, value }) => [name, value]));
}

function chromeWork(before: Map<string, number>, after: Map<string, number>) {
  return Object.fromEntries(['ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration', 'TaskDuration'].map((name) => [
    name, 1000 * (after.get(name)! - before.get(name)!),
  ]));
}

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

async function measureField(page: Page, context: CDPSession, projectPath: string) {
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const field = inspector.getByRole('textbox', { name: 'Step', exact: true });
  await expect(field).toBeEditable({ timeout: 8_000 });
  await field.selectText();
  const value = '0.02 s';
  await startSamples(field, 'field', 'typing');
  const beforeTyping = await chromeMetrics(context);
  await field.pressSequentially(value, { delay: 35 });
  await expect(field).toHaveValue(value);
  await field.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const typingWork = chromeWork(beforeTyping, await chromeMetrics(context));
  const typing = await endSamples(page);
  expect(typing.updates.length).toBeGreaterThan(0);

  const prepared = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === `${projectPath}/model/fields` &&
    Boolean(response.request().postDataJSON().edit), { timeout: 8_000 });
  const saved = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === `${projectPath}/model/edit`, { timeout: 8_000 });
  const savedEdit = saved.then(async (response) => {
    const acknowledgedAt = performance.now();
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    if (result.status !== 'applied') throw new Error(result.problem.detail);
    return { snapshot: result.snapshot, acknowledgedAt };
  });
  const settled = page.waitForResponse(async (response) => {
    const url = new URL(response.url());
    return response.ok() && response.request().method() === 'GET' &&
      url.pathname === `${projectPath}/train-runs` && url.searchParams.get('window_map') === 'true' &&
      url.searchParams.get('revision') === (await savedEdit).snapshot.revision;
  }, { timeout: 12_000 });
  void savedEdit.catch(() => {});
  void settled.catch(() => {});
  const projected = page.waitForResponse(async (response) => {
    const request = response.request();
    if (!response.ok() || request.method() !== 'POST' ||
      new URL(response.url()).pathname !== `${projectPath}/model/fields`) return false;
    const body = request.postDataJSON();
    return !body.edit && body.revision === (await savedEdit).snapshot.revision;
  }, { timeout: 12_000 });
  void projected.catch(() => {});
  await startSamples(field, 'field', 'commit');
  const beforeCommit = await chromeMetrics(context);
  const submittedAt = performance.now();
  await field.press('Enter');
  const preparation = await prepared;
  expect(preparation.ok(), await preparation.text()).toBe(true);
  const fields = await preparation.json() as ModelFieldResponse;
  expect(fields.problems).toHaveLength(0);
  expect(fields.command?.operation.op).toBe('set_field');
  const { snapshot, acknowledgedAt } = await savedEdit;
  const model = snapshot.project.documents.model.payload as unknown as ModelDocument;
  expect(model.step_seconds).toBe(0.02);
  const plan = await (await settled).json() as StudioSavedTrainRuns;
  expect(plan.revision).toBe(snapshot.revision);
  expect(plan.window_map?.rows).toHaveLength(1);
  await projected;
  await expect(field).toBeEditable();
  await expect(field).toHaveValue(value);
  const visibleAt = performance.now();
  const commitWork = chromeWork(beforeCommit, await chromeMetrics(context));
  const commit = await endSamples(page);
  return { keys: value.length, typing, typingWork, commit, commitWork,
    submitToAcknowledgementMs: acknowledgedAt - submittedAt,
    acknowledgementToEditableFieldMs: visibleAt - acknowledgedAt };
}

async function measureScroll(page: Page, request: APIRequestContext, context: CDPSession) {
  const rowCount = 1_000;
  const rates = Array.from({ length: rowCount }, (_, index) => ((index + 1) / 100_000).toFixed(5));
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2', ...rates));
  const projected = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return response.request().method() === 'GET' && url.pathname === `${projectPath}/train-runs` &&
      url.searchParams.get('revision') === snapshot.revision && url.searchParams.get('window_map') === 'true';
  }, { timeout: 60_000 });
  void projected.catch(() => {});
  await page.goto(`/?project=${snapshot.project_id}`);
  const plan = await projected;
  expect(plan.ok(), plan.ok() ? undefined : await plan.text()).toBe(true);
  const { rows: planned } = await plan.json() as StudioSavedTrainRuns;
  expect(planned).toHaveLength(rowCount);
  expect(planned.every((row) => row.status === 'not_run' && row.request_id)).toBe(true);
  expect(new Set(planned.map((row) => row.request_id)).size).toBe(rowCount);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  await expect(table).toHaveAttribute('aria-rowcount', String(rowCount + 1));
  const rows = table.locator('tbody tr[data-index]');
  await expect(rows.first()).toHaveAttribute('data-index', '0');
  const viewport = table.locator('..');
  const estimatedDistance = await viewport.evaluate((node) => node.scrollHeight - node.clientHeight);
  expect(estimatedDistance).toBeGreaterThan(0);
  const inputs = 100;
  await viewport.hover();
  await startSamples(viewport, 'scroll', 'idle');
  await viewport.evaluate(() => new Promise<void>((done) => {
    let remaining = 20;
    requestAnimationFrame(function frame() {
      if (--remaining) requestAnimationFrame(frame);
      else done();
    });
  }));
  const idle = await endSamples(page);
  await startSamples(viewport, 'scroll', 'scroll');
  const before = await chromeMetrics(context);
  const roundTrip = await paceInputs(inputs, () => page.mouse.wheel(0, Math.ceil(estimatedDistance / inputs)));
  const end = await viewport.evaluate(async (node, lastIndex) => {
    await new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done())));
    const viewportTop = node.getBoundingClientRect().top + node.clientTop;
    const viewportBottom = viewportTop + node.clientHeight;
    const row = node.querySelector(`tbody tr[data-index="${lastIndex}"]`)?.getBoundingClientRect();
    return {
      offset: node.scrollTop,
      distance: node.scrollHeight - node.clientHeight,
      lastRowVisible: row !== undefined && row.top >= viewportTop - 1 && row.bottom <= viewportBottom + 1,
    };
  }, rowCount - 1);
  const work = chromeWork(before, await chromeMetrics(context));
  const samples = await endSamples(page);
  return { rowCount, inputs, estimatedDistance, end, work, idle, samples, roundTrip,
    scrollUpdates: statistics(gaps(samples.updates)) };
}

async function paceInputs(count: number, send: (index: number) => Promise<void>) {
  const roundTrip: number[] = [];
  let due = performance.now();
  for (let index = 1; index <= count; index++) {
    await new Promise((done) => setTimeout(done, Math.max(0, due - performance.now())));
    const began = performance.now();
    await send(index);
    roundTrip.push(performance.now() - began);
    due = Math.max(due + 1000 / 60, began + 1000 / 60);
  }
  return roundTrip;
}

test('measures drag updates in the mounted Model at 20 and 100 cards', async ({ page, request, browser, browserName }, info) => {
  const measureInteractions = process.env.FEEDBAX_STUDIO_MEASURE === '1';
  test.setTimeout(measureInteractions ? 100_000 : 55_000);
  const result = [];
  const context = await page.context().newCDPSession(page);
  await context.send('Performance.enable');
  const trace = info.outputPath('studio-interactions-chrome.json');
  if (measureInteractions) await browser.startTracing(undefined, { path: trace, categories: [
    'blink.user_timing', 'devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'cc', 'viz',
  ] });
  try {
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
      const header = card.locator('header');
      const lock = model.getByRole('button', { name: 'Lock', exact: true });
      await expect(lock).toBeEnabled();
      await expect(lock).toHaveAttribute('aria-pressed', 'false');
      await model.getByRole('button', { name: 'Fit View', exact: true }).click();
      const samplesByDrag = [];
      for (let repetition = 0; repetition < 4; repetition++) {
        await expect(lock).toBeEnabled();
        await header.hover();
        const prior = await edge.getAttribute('d');
        const box = await header.boundingBox();
        if (!box) throw new Error('The mounted card is not measurable');
        const start = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
        await page.mouse.move(start.x, start.y);
        await startSamples(card, 'drag', `drag-${cards}-${repetition}`);
        const hitCard = await header.evaluate((element, point) =>
          element.contains(document.elementFromPoint(point.x, point.y)), start);
        expect(hitCard, 'The drag start must hit the Model card header').toBe(true);
        const saved = page.waitForResponse((response) => response.request().method() === 'POST' &&
          new URL(response.url()).pathname === `${projectPath}/model/edit`);
        await page.mouse.down();
        const before = await chromeMetrics(context);
        const direction = repetition % 2 === 0 ? -1 : 1;
        const roundTrip = await paceInputs(50, (move) =>
          page.mouse.move(start.x + direction * move * 1.6, start.y + direction * move * 0.6));
        const work = chromeWork(before, await chromeMetrics(context));
        const timing = await endSamples(page);
        const releasedAt = performance.now();
        await page.mouse.up();
        const response = await saved;
        const releaseToAcknowledgementMs = performance.now() - releasedAt;
        expect(response.ok(), await response.text()).toBe(true);
        const { result: edit } = await response.json() as ModelEditResponse;
        expect(edit.status).toBe('applied');
        expect(timing.updates.length).toBeGreaterThan(1);
        await expect(edge).not.toHaveAttribute('d', prior!);
        if (repetition > 0) samplesByDrag.push({ ...timing, roundTrip, chromeWork: work, releaseToAcknowledgementMs });
      }
      result.push({
        cards, connections: cards * 3,
        callbacks: statistics(samplesByDrag.flatMap(({ callbacks, updates }) =>
          gaps(callbacks.filter((time) => time >= updates[0] && time <= updates.at(-1)!)))),
        changedCard: statistics(samplesByDrag.flatMap(({ updates }) => gaps(updates))),
        roundTrip: statistics(samplesByDrag.flatMap((sample) => sample.roundTrip)),
        longTasks: samplesByDrag.flatMap((sample) => sample.longTasks),
        raw: samplesByDrag,
      });
      if (cards === 100) {
        const drag = { browserName, version: browser.version(), slots: 1, viewport: page.viewportSize(), result };
        await info.attach('mounted-model-drag', { body: JSON.stringify(drag, null, 2), contentType: 'application/json' });
        console.log(JSON.stringify({ ...drag, result: result.map(({ raw, ...summary }) => summary) }));
        if (measureInteractions) {
          await model.locator('.react-flow__pane').click({ position: { x: 10, y: 10 } });
          const field = await measureField(page, context, projectPath);
          await info.attach('mounted-model-field', { body: JSON.stringify(field, null, 2), contentType: 'application/json' });
        }
      }
    }
    if (measureInteractions) {
      const scroll = await measureScroll(page, request, context);
      await info.attach('mounted-runs-scroll', { body: JSON.stringify(scroll, null, 2), contentType: 'application/json' });
      expect(scroll.samples.updates.length).toBeGreaterThan(1);
      expect(scroll.end.lastRowVisible).toBe(true);
      expect(Math.abs(scroll.end.offset - scroll.end.distance)).toBeLessThanOrEqual(1);
    }
  } finally {
    if (measureInteractions) {
      await browser.stopTracing();
      await info.attach('studio-interactions-chrome', { path: trace, contentType: 'application/json' });
    }
    await context.detach();
  }
});
