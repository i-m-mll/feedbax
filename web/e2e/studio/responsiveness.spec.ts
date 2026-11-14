import { expect, test, type APIRequestContext, type CDPSession, type TestInfo, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, ModelFieldResponse, StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns, WorkspaceReplayTrial } from '../../src/generated/studioContracts';
import { summarizeStudioTrace } from './fixtures/studioTrace';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ headless: true, trace: 'off', viewport: { width: 1440, height: 900 } });

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

type Samples = { callbacks: number[]; updates: number[]; longTasks: { duration: number; startTime: number }[] };
type SampleWindow = Window & {
  finishStudioSample?: () => Samples;
  settledPlanStatus?: { visibleAt: number; text: string };
};

async function startSamples(target: Locator, kind: 'drag' | 'field' | 'scroll' | 'replay', label: string) {
  await target.evaluate(async (element, { kind, label }) => {
    const state = window as SampleWindow;
    if (state.finishStudioSample) throw new Error('A Studio measurement is already running');
    const node = element as HTMLElement;
    const value = () => {
      if (kind === 'drag') return node.style.transform;
      if (kind === 'field') return (node as HTMLInputElement).value;
      if (kind === 'replay') return [node.getAttribute('cx'), node.getAttribute('cy')].join(',');
      return node.scrollTop;
    };
    const samples: Samples = { callbacks: [], updates: [], longTasks: [] };
    const tasks = (entries: PerformanceEntry[]) => entries.forEach(({ duration, startTime }) => samples.longTasks.push({ duration, startTime }));
    if (!PerformanceObserver.supportedEntryTypes.includes('longtask')) {
      throw new Error('Chromium Long Tasks instrumentation is unavailable');
    }
    const taskObserver = new PerformanceObserver((list) => tasks(list.getEntries()));
    taskObserver.observe({ type: 'longtask' });
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

async function measureField(page: Page, context: CDPSession, projectPath: string, modelDocument: string) {
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
    return { snapshot: result.snapshot, changed: result.changed, acknowledgedAt };
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
  const planCard = page.getByRole('article', { name: 'Train train', exact: true });
  const planStatus = planCard.getByRole('button', { name: /^\d+ Can't run$/ });
  await expect(planStatus).toBeVisible();
  await planStatus.evaluate((status) => {
    const state = window as SampleWindow;
    const card = status.closest<HTMLElement>('[data-map-step]')!;
    let pending = false;
    const observer = new MutationObserver(() => {
      const restored = Array.from(card.querySelectorAll<HTMLButtonElement>('button[aria-haspopup="dialog"]'))
        .find((button) => /^\d+ Can't run$/.test(button.textContent?.trim() ?? '') && button.checkVisibility());
      if (!restored) pending = true;
      if (!pending || !restored) return;
      state.settledPlanStatus = { visibleAt: performance.timeOrigin + performance.now(), text: restored.textContent!.trim() };
      performance.mark('studio-commit-plan-status-visible');
      observer.disconnect();
    });
    observer.observe(card, { subtree: true, childList: true, characterData: true, attributes: true });
  });
    await startSamples(field, 'field', 'commit');
    const beforeCommit = await chromeMetrics(context);
    const submittedAt = performance.now();
    await field.press('Enter');
    const preparation = await prepared;
    expect(preparation.ok(), await preparation.text()).toBe(true);
    const fields = await preparation.json() as ModelFieldResponse;
    expect(fields.problems).toHaveLength(0);
    expect(fields.command?.operation.op).toBe('set_field');
    const { snapshot, changed, acknowledgedAt } = await savedEdit;
    const addressed = changed.find(({ element }) => element.kind === 'field' && element.path.at(-1) === 'step_seconds');
    if (!addressed) throw new Error('The acknowledgement did not address the Step field');
    expect(addressed.document).toBe(modelDocument);
    const model = snapshot.project.documents[addressed.document].payload as unknown as ModelDocument;
    expect(model.step_seconds).toBe(0.02);
    const plan = await (await settled).json() as StudioSavedTrainRuns;
    expect(plan.revision).toBe(snapshot.revision);
    expect(plan.window_map?.rows).toHaveLength(1);
    const cannotRun = plan.window_map?.steps.find(({ step_id }) => step_id === 'train')?.status_counts.cannot_run;
    expect(cannotRun).toBe(1);
    await projected;
    await expect(field).toBeEditable();
    await expect(field).toHaveValue(value);
    const visibleAt = performance.now();
    const commitWork = chromeWork(beforeCommit, await chromeMetrics(context));
    const commit = await endSamples(page);
    const planVisible = await page.evaluate(() => (window as SampleWindow).settledPlanStatus);
    if (!planVisible) throw new Error('The Plan status did not return from the pending state');
    expect(planVisible.text).toBe(`${cannotRun} Can't run`);
    return { keys: value.length, typing, typingWork, commit, commitWork,
      submitToAcknowledgementMs: acknowledgedAt - submittedAt,
      acknowledgementToEditableFieldMs: visibleAt - acknowledgedAt,
      acknowledgementToVisiblePlanStatusMs: planVisible.visibleAt - (performance.timeOrigin + acknowledgedAt) };

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

async function measureDrag(page: Page, model: Locator, context: CDPSession, projectPath: string) {
  const lock = model.getByRole('button', { name: 'Lock', exact: true });
  const card = model.locator('.react-flow__node[data-id="sum-5"]');
  const header = card.locator('header');
  const edge = model.locator('.react-flow__edge[data-id="wire-5-a"] .react-flow__edge-path');
  await expect(lock).toBeEnabled();
  await model.getByRole('button', { name: 'Fit View', exact: true }).click();
  const observations = [];
  for (let repetition = 0; repetition < 4; repetition++) {
    await expect(lock).toBeEnabled();
    await header.hover();
    const beforeEdge = await edge.getAttribute('d');
    const box = await header.boundingBox();
    if (!box) throw new Error('Model card is missing its bounding box');
    const start = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    await page.mouse.move(start.x, start.y);
    const hit = await header.evaluate((element, point) => element.contains(document.elementFromPoint(point.x, point.y)), start);
    expect(hit).toBe(true);
    const saved = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === projectPath + '/model/edit');
    void saved.catch(() => {});
    await startSamples(card, 'drag', 'drag-100-' + repetition);
    await page.mouse.down();
    const metrics = await chromeMetrics(context);
    const direction = repetition % 2 ? 1 : -1;
    const roundTrip = await paceInputs(50, (move) => page.mouse.move(start.x + direction * move * 1.6, start.y + direction * move * 0.6));
    const work = chromeWork(metrics, await chromeMetrics(context));
    const samples = await endSamples(page);
    const released = performance.now();
    await page.mouse.up();
    const response = await saved;
    const acknowledgedMs = performance.now() - released;
    expect(response.ok(), await response.text()).toBe(true);
    expect((await response.json() as ModelEditResponse).result.status).toBe('applied');
    expect(samples.updates.length).toBeGreaterThan(1);
    await expect(edge).not.toHaveAttribute('d', beforeEdge!);
    observations.push({ samples, roundTrip, work, acknowledgedMs, changed: statistics(gaps(samples.updates)) });
  }
  return observations;
}

async function measureReplay(page: Page, request: APIRequestContext, context: CDPSession) {
  const project = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed', String(info.config.metadata.home), 'playback-cadence',
  ));
  const path = '/api/projects/' + project;
  const opened = await request.get(path);
  expect(opened.ok()).toBe(true);
  const saved = await opened.json() as StudioProjectSnapshot;
  await page.goto('/?project=' + encodeURIComponent(project));
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  const tabs = page.getByRole('tablist', { name: 'Details' });
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  async function selectStep(step: string) {
    await mapStep(page, step).click();
    await tabs.getByRole('tab', { name: 'Runs', exact: true }).click();
  }
  async function run(step: string) {
    await selectStep(step);
    const accepted = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === path + '/train-runs');
    void accepted.catch(() => {});
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await accepted;
    expect(response.status(), await response.text()).toBe(202);
    const [item] = (await response.json() as StudioJobSubmission).items;
    expect(item.disposition).toBe('admitted');
    await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 24_000 });
    return item;
  }
  const train = await run('train');
  const evaluation = await run('baseline');
  const projected = await request.get(path + '/train-runs', { params: { revision: saved.revision, window_map: 'true' } });
  expect(projected.ok()).toBe(true);
  const plan = await projected.json() as StudioSavedTrainRuns;
  const result = plan.window_map?.evaluations?.find(({ key }) => key.step === 'baseline');
  expect(result?.result?.job_id).toBe(evaluation.job_id);
  expect(result?.result?.checkpoint_source?.job_id).toBe(train.job_id);
  const trial = result!.trials[1];
  const row = plan.window_map!.rows.find(({ key }) => key.step === 'baseline')!;
  await table.getByRole('button', { name: row.label, exact: true }).click();
  await tabs.getByRole('tab', { name: 'Scene', exact: true }).click();
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  const cursor = scene.locator('[data-scene-cursor]');
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  if (!await drawer.isVisible()) await footer.getByRole('button', { name: 'Show Drawer' }).click();
  const trials = drawer.getByRole('combobox', { name: 'Trials', exact: true });
  const responded = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === path + '/evaluate-replay' && response.request().postDataJSON().trial === trial);
  void responded.catch(() => {});
  await trials.selectOption(trial);
  const loaded = await responded;
  expect(loaded.ok(), await loaded.text()).toBe(true);
  expect(loaded.request().postDataJSON()).toMatchObject({ key: row.key, job_id: evaluation.job_id, trial });
  const replay = await loaded.json() as WorkspaceReplayTrial;
  expect(replay.identity.stable_id).toBe(trial);
  expect(replay.manifest_refs.producer_manifest).toMatchObject({
    kind: 'EvaluationRunManifest', id: result?.result?.receipt.manifest_id,
    metadata: { manifest_sha256: result?.result?.receipt.manifest_sha256 },
  });
  expect(replay.time.length).toBe(81);
  await expect(cursor).toBeVisible();
  const clock = footer.getByRole('group', { name: 'Clock', exact: true });
  const scrubber = clock.getByRole('slider', { name: 'Playback Position' });
  await expect(scrubber).toHaveValue('0');
  await startSamples(cursor, 'replay', 'replay');
  const before = await chromeMetrics(context);
  const play = clock.getByRole('button', { name: 'Play', exact: true });
  await play.evaluate((button) => {
    button.addEventListener('click', () => performance.mark('studio-playing-start'), { once: true });
  });
  await play.click();
  await expect(clock.getByRole('button', { name: 'Pause', exact: true })).toBeVisible();
  await cursor.evaluate(() => new Promise<void>((done) => {
    let frames = 120;
    requestAnimationFrame(function sample() { if (--frames) requestAnimationFrame(sample); else done(); });
  }));
  const pause = clock.getByRole('button', { name: 'Pause', exact: true });
  await pause.evaluate((button) => {
    button.addEventListener('click', () => performance.mark('studio-playing-end'), { once: true });
  });
  await pause.click();
  await expect(clock.getByRole('button', { name: 'Play', exact: true })).toBeVisible();
  const work = chromeWork(before, await chromeMetrics(context));
  const raw = await endSamples(page);
  const [start, end] = await page.evaluate(() => ['studio-playing-start', 'studio-playing-end'].map(
    (mark) => performance.getEntriesByName(mark)[0].startTime,
  ));
  const samples = { ...raw,
    callbacks: raw.callbacks.filter((time) => time >= start && time <= end),
    updates: raw.updates.filter((time) => time >= start && time <= end),
    longTasks: raw.longTasks.filter(({ startTime, duration }) => startTime <= end && startTime + duration >= start),
  };
  const paused = await scrubber.inputValue();
  await cursor.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
  expect(samples.updates.length).toBeGreaterThanOrEqual(60);
  await expect(scrubber).toHaveValue(paused);
  return { receipt: result?.result?.receipt, trial, sampleCount: replay.time.length,
    paused, samples, work, cursorUpdates: statistics(gaps(samples.updates)),
    callbackGaps: statistics(gaps(samples.callbacks)) };
}

test('measures mounted Model, Runs and Scene responsiveness @responsiveness', async ({ page, request, browser }, info) => {
  test.setTimeout(100_000);
  expect(process.env.FEEDBAX_STUDIO_E2E_PREVIEW).toBe('1');
  const files = await fixture('ordinary-files', '2');
  const modelFile = files.find((file: { document?: { schema: string } }) => file.document?.schema === 'feedbax.spec.model');
  const stepFile = files.find((file: { kind: string }) => file.kind === 'step');
  const trial = stepFile.step.base.inline.trial_documents;
  delete trial.task;
  trial.python_task = { document: 'profile-task' };
  trial.objective = { document: 'profile-objective' };
  modelFile.document.payload = modelFixture(100);
  const { snapshot, projectPath } = await seedManagedProject(request, [
    modelFile,
    { kind: 'document', id: 'profile-task', document: {
      schema: 'feedbax.spec.python_task', payload: { type: 'feedbax.task.reaching.v1', params: {} },
    } },
    { kind: 'document', id: 'profile-objective', document: {
      schema: 'feedbax.spec.objective_document', payload: { id: 'profile-objective', terms: [] },
    } },
    stepFile,
  ]);
  await page.goto('/?project=' + encodeURIComponent(snapshot.project_id));
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.locator('.react-flow__node')).toHaveCount(102);
  await expect(model.locator('.react-flow__edge')).toHaveCount(300);
  await expect(model.getByRole('button', { name: 'Lock', exact: true })).toBeEnabled();
  const context = await page.context().newCDPSession(page);
  await context.send('Performance.enable');
  const trace = info.outputPath('studio-responsiveness-chrome.json');
  await browser.startTracing(undefined, { path: trace, categories: ['blink.user_timing', 'devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'cc', 'viz', 'toplevel'] });
  let fieldResult: Awaited<ReturnType<typeof measureField>>;
  try {
    fieldResult = await measureField(page, context, projectPath, modelFile.id);
    const field = fieldResult;
    await info.attach('mounted-model-field', { body: JSON.stringify(field), contentType: 'application/json' });
    const drag = await measureDrag(page, model, context, projectPath);
    await info.attach('mounted-model-drag', { body: JSON.stringify(drag), contentType: 'application/json' });
    const scroll = await measureScroll(page, request, context);
    await info.attach('mounted-runs-scroll', { body: JSON.stringify(scroll), contentType: 'application/json' });
    expect(scroll.end.lastRowVisible).toBe(true);
    expect(Math.abs(scroll.end.offset - scroll.end.distance)).toBeLessThanOrEqual(1);
    const replay = await measureReplay(page, request, context, info);
    await info.attach('mounted-scene-playback', { body: JSON.stringify(replay), contentType: 'application/json' });
    const longTasks = [...field.commit.longTasks, ...drag.flatMap(({ samples }) => samples.longTasks), ...replay.samples.longTasks];
    expect.soft(longTasks.filter(({ duration }) => duration > 50), 'drag, commit and active playback main-thread tasks over 50 ms').toEqual([]);
    expect.soft(field.acknowledgementToVisiblePlanStatusMs, 'revision-matched DOM status after edit acknowledgement').toBeGreaterThanOrEqual(0);
    expect.soft(field.acknowledgementToVisiblePlanStatusMs, 'revision-matched DOM status after edit acknowledgement').toBeLessThanOrEqual(500);

  } finally {
    await browser.stopTracing();
    await info.attach('studio-step-chrome', { path: trace, contentType: 'application/json' });
    await context.detach();
  }
  const summary = await summarizeStudioTrace(trace);
  await info.attach('studio-responsiveness-summary', {
    body: JSON.stringify({ browser: browser.version(), ...summary }, null, 2), contentType: 'application/json',
  });
  expect(summary.keydownMs).toHaveLength(fieldResult.keys);
  expect(Math.max(...summary.keydownMs), 'headless Chromium keydown EventTiming').toBeLessThanOrEqual(50);
  for (const frame of summary.frames) {
    expect(frame.beginFrames, frame.name).toBeGreaterThan(0);
    expect(frame.drawFrames, frame.name).toBeGreaterThan(0);
    if (frame.name === 'scroll') {
      expect(frame.rendererDrops.filter(drop => drop.partialUpdate !== true), 'full or unclassified renderer drops during scroll').toEqual([]);
    }
  }
});
