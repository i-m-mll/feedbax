import { expect, test, jobTimeout } from './fixtures/studio';
import type { APIRequestContext, Locator, Page } from '@playwright/test';
import { analysisDocumentSchemaId, type StudioJobSubmission, type StudioProjectDocument, type StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { appendManagedFiles, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pickRecordedTrial, showTrials, trialPicker } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const panel = (page: Page, name: string) => page.getByRole('tabpanel', { name, exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });
const footer = (page: Page) => page.locator('[data-studio-focus-region="bottom"]');
const scrubber = (page: Page) => footer(page).getByRole('slider', { name: 'Playback Position' });
const velocity = '/v1/model/state/plant/effector~1velocity';

/** The evaluation project with an Analyze drawing Speed by trial from the baseline's 64 recorded trials, all run. */
async function speedFigure(page: Page, request: APIRequestContext, home: string, scenario: string) {
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-sixty-four-trials', home, scenario,
  ));
  const path = `/api/projects/${project}`;
  const opened = await request.get(path + '/open');
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  const { snapshot: seeded } = await appendManagedFiles(request, snapshot, [
    { kind: 'document', id: 'analysis', document: {
      schema: analysisDocumentSchemaId, label: 'Analysis', payload: {
        inputs: { states: { kind: 'dataset' } },
        operations: { speed: {
          label: 'Speed', operation: 'feedbax.trial_speed', params: { velocity }, inputs: { states: { input: 'states' } },
        } },
        views: { profile: {
          label: 'Profile', figure: { template: 'feedbax.SpeedByTrial' }, inputs: { speed: { element: 'speed', product: 'speed' } },
        } },
      },
    } },
    { kind: 'step', id: 'analyze', step: {
      label: 'Analyze', kind: 'analyze', base: { document: 'analysis' }, inputs: { states: { step: 'baseline' } },
    } },
  ]);
  const plans = page.getByRole('table', { name: 'Runs', exact: true });
  const run = async (step: string) => {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const accepted = page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === path + '/train-runs');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await accepted;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0].disposition).toBe('admitted');
    await expect(plans.getByText('Done', { exact: true }).first()).toBeVisible({ timeout: jobTimeout });
  };
  await page.goto('/?project=' + encodeURIComponent(project));
  await run('train');
  await run('baseline');
  await run('analyze');
  const projected = await request.get(path + '/train-runs', { params: { revision: seeded.revision, window_map: 'true' } });
  expect(projected.ok(), await projected.text()).toBe(true);
  const { window_map } = await projected.json() as StudioSavedTrainRuns;
  const trials = window_map!.evaluations!.find(({ key }) => key.step === 'baseline')!.trials;
  expect(trials).toHaveLength(64);
  return { trials };
}

async function showFigure(page: Page) {
  await mapStep(page, 'analyze').click();
  await panel(page, 'Runs').click();
  await tab(page, 'Figures').click();
  return drawnFigure(page);
}

/** The shown Analyze's Speed by trial figure, once drawn. */
async function drawnFigure(page: Page) {
  const figures = panel(page, 'Figures');
  await figures.getByRole('button', { name: 'Profile', exact: true }).click();
  const plot = figures.locator('.js-plotly-plot');
  await expect(plot.locator('svg.main-svg').first()).toBeVisible();
  await expect.poll(() => plot.evaluate((node) => (node as unknown as { data?: unknown[] }).data?.length)).toBe(64);
  return plot;
}

interface DrawnTrace { trial?: string; visible: unknown; width: number }

/** The traces as the figure asks Plotly to draw them: each one's trial, whether it is drawn, and its line width. */
const drawnTraces = (plot: Locator) => plot.evaluate((node) =>
  (node as unknown as { data: { meta?: { group?: { trial?: string } }; visible?: unknown; line?: { width?: number } }[] })
    .data.map((trace): DrawnTrace => ({ trial: trace.meta?.group?.trial, visible: trace.visible ?? true, width: trace.line?.width ?? 2 })));

const shownTrials = async (plot: Locator) => (await drawnTraces(plot)).filter(({ visible }) => visible === true).map(({ trial }) => trial);

async function heavyTrial(plot: Locator) {
  const traces = await drawnTraces(plot);
  const heaviest = Math.max(...traces.map(({ width }) => width));
  return traces.filter(({ width }) => width === heaviest).map(({ trial }) => trial);
}

/** The playhead's x and the pixel the x axis draws `time` at, in the plot's own frame. */
const playheadAt = (plot: Locator, time: number) => plot.evaluate((node, time) => {
  const axis = (node as unknown as { _fullLayout: { xaxis: { _offset: number; l2p(value: number): number } } })._fullLayout.xaxis;
  const line = node.parentElement!.querySelector<HTMLElement>('[data-figure-playhead]')!;
  return { drawn: new DOMMatrix(getComputedStyle(line).transform).m41, expected: axis._offset + axis.l2p(time) };
}, time);

/** The time the x axis draws at a fraction of its width, with that point in page coordinates at mid-height. */
const axisPoint = (plot: Locator, fraction: number) => plot.evaluate((node, fraction) => {
  const { xaxis, yaxis } = (node as unknown as { _fullLayout: Record<'xaxis' | 'yaxis', { _offset: number; _length: number; p2l(pixel: number): number }> })._fullLayout;
  const box = node.getBoundingClientRect();
  const pixel = xaxis._length * fraction;
  return { time: xaxis.p2l(pixel), x: box.left + xaxis._offset + pixel, y: box.top + yaxis._offset + yaxis._length / 2 };
}, fraction);

// Catches a figure that ignores the clock's time or its picked trial, a pin that freezes the trial the figure and the
// Scene share, a figure whose press does not move the clock, a click on a trace that picks no trial, a legend or hover
// label listing every trial over the plot, and a playhead drawn over a figure that declares no time panel.
test('with the Scene pinned on an Evaluate and its Analyze selected, scrubbing moves both and the trial is marked', { tag: '@round' }, async ({ page, request, studio }) => {
  const { trials } = await speedFigure(page, request, studio.home, 'figure-clock');
  await mapStep(page, 'baseline').click();
  await tab(page, 'Runs').click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await tab(page, 'Scene').click();
  const scene = panel(page, 'Scene');
  await expect(scene.getByRole('group', { name: 'Scene', exact: true })).toBeVisible();
  await scene.getByRole('button', { name: 'Pin', exact: true }).click();
  const plot = await showFigure(page);
  const playhead = page.locator('[data-figure-playhead]');
  await expect(playhead).toHaveCount(1);

  await expect(trialPicker(page)).toHaveText('trial 1 of 64');
  await expect.poll(() => shownTrials(plot)).toEqual([trials[0]]);
  await pickRecordedTrial(page, 3);
  await expect.poll(() => shownTrials(plot)).toEqual([trials[3]]);

  const plant = scene.locator('[data-scene-layer="bodies"] [data-scene-shown]').first();
  const pose = () => plant.evaluate((circle: SVGCircleElement) => [circle.cx.baseVal.value, circle.cy.baseVal.value]);
  const start = await pose();
  await scrubber(page).press('End');
  const end = Number(await scrubber(page).getAttribute('aria-valuenow'));
  expect(end).toBeGreaterThan(0);
  await expect.poll(async () => { const { drawn, expected } = await playheadAt(plot, end); return Math.abs(drawn - expected); })
    .toBeLessThan(1);
  await expect.poll(pose).not.toEqual(start);

  const quarter = await axisPoint(plot, 0.25);
  await page.mouse.move(quarter.x, quarter.y);
  await page.mouse.down();
  const half = await axisPoint(plot, 0.5);
  await page.mouse.move(half.x, half.y, { steps: 4 });
  await page.mouse.up();
  await expect.poll(async () => Math.abs(Number(await scrubber(page).getAttribute('aria-valuenow')) - half.time)).toBeLessThan(0.02);
  await expect.poll(async () => { const { drawn, expected } = await playheadAt(plot, half.time); return Math.abs(drawn - expected); })
    .toBeLessThan(1);

  await showTrials(page, 'All');
  await expect(trialPicker(page)).toHaveText('64 trials');
  await expect.poll(() => shownTrials(plot)).toHaveLength(64);
  await expect.poll(() => heavyTrial(plot)).toEqual([trials[3]]);
  await expect(plot.locator('g.legend .traces')).toHaveCount(0);
  expect(await plot.evaluate((node) => (node as unknown as { _fullLayout: { yaxis: { _length: number } } })._fullLayout.yaxis._length /
    node.getBoundingClientRect().height)).toBeGreaterThan(0.5);
  await page.mouse.move(half.x, half.y);
  await page.mouse.move(half.x + 4, half.y);
  await expect(plot.locator('.hoverlayer text')).toHaveCount(0);

  const top = await plot.evaluate((node, x) => {
    const gd = node as unknown as {
      _fullData: { x: ArrayLike<number>; y: ArrayLike<number>; meta: { group: { trial: string } } }[];
      _fullLayout: Record<'xaxis' | 'yaxis', { _offset: number; l2p(value: number): number }>;
    };
    const at = Array.from(gd._fullData[0].x).findIndex((time) => time >= x);
    const highest = gd._fullData.reduce((best, trace) => trace.y[at] > best.y[at] ? trace : best);
    const box = node.getBoundingClientRect();
    const { xaxis, yaxis } = gd._fullLayout;
    return { trial: highest.meta.group.trial, x: box.left + xaxis._offset + xaxis.l2p(highest.x[at]), y: box.top + yaxis._offset + yaxis.l2p(highest.y[at]) };
  }, half.time);
  expect(top.trial).not.toBe(trials[3]);
  await page.mouse.click(top.x, top.y);
  await expect(trialPicker(page)).toHaveText('64 trials');
  await expect.poll(() => heavyTrial(plot)).toEqual([top.trial]);

  await page.route('**/light-results/**/figure/figure.json', async (route) => {
    const product = await (await route.fetch()).json();
    for (const declared of Object.values(product.layout.meta.panels) as { time?: unknown }[]) delete declared.time;
    await route.fulfill({ json: product });
  });
  await page.reload();
  const untimed = await drawnFigure(page);
  await expect(untimed.locator('svg.main-svg').first()).toBeVisible();
  await expect(playhead).toHaveCount(0);
});

// Catches a playhead that redraws the figure's 64 traces to move, which cannot keep up with playback: it must move on
// nearly every frame.
test('the playhead of Speed by trial over 64 trials holds 60 Hz in playback @responsiveness', async ({ page, request, studio }, info) => {
  test.setTimeout(120_000);
  await speedFigure(page, request, studio.home, 'figure-clock-rate');
  await showFigure(page);
  const line = page.locator('[data-figure-playhead]');
  await expect(line).toHaveCount(1);
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  const duration = Number(await scrubber(page).getAttribute('max'));
  expect(duration).toBeGreaterThanOrEqual(0.8);
  await clock.getByRole('button', { name: 'Play', exact: true }).click();
  // Sampled for 45 frames, which end before the trial does at any display rate of 60 Hz or more.
  const { frames, updates } = await line.evaluate((node) => new Promise<{ frames: number[]; updates: number[] }>((done) => {
    const frames: number[] = [];
    const updates: number[] = [];
    let last = node.style.transform;
    requestAnimationFrame(function sample(time) {
      frames.push(time);
      if (node.style.transform !== last) { updates.push(time); last = node.style.transform; }
      if (frames.length < 45) requestAnimationFrame(sample); else done({ frames, updates });
    });
  }));
  const gaps = updates.slice(1).map((time, index) => time - updates[index]).sort((a, b) => a - b);
  const p95 = gaps[Math.ceil(gaps.length * 0.95) - 1];
  await info.attach('figure-playhead-gaps', {
    body: JSON.stringify({ frames: frames.length, updates: updates.length, p95, maximum: gaps.at(-1) }), contentType: 'application/json',
  });
  expect(updates.length).toBeGreaterThanOrEqual(0.9 * frames.length);
  expect(p95).toBeLessThanOrEqual(1000 / 60 * 1.2);
});

// Catches a window reopened on the Figures view that draws no figure and offers no trial clock until a figure is
// pressed, since the selection that chose the figure is not kept across a reopen.
test('a window reopened on the Figures view draws its figure on the clock without a press', { tag: '@round' }, async ({ page, request, studio }) => {
  await speedFigure(page, request, studio.home, 'figure-reopen');
  await showFigure(page);
  await expect(trialPicker(page)).toHaveText('trial 1 of 64');
  await page.reload();
  const plot = panel(page, 'Figures').locator('.js-plotly-plot');
  await expect.poll(() => plot.evaluate((node) => (node as unknown as { data?: unknown[] }).data?.length)).toBe(64);
  await expect(trialPicker(page)).toHaveText('trial 1 of 64');
  await expect(page.locator('[data-figure-playhead]')).toHaveCount(1);
});
