import { expect, test, jobTimeout } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StudioJobSubmission, StudioTrialReplayResponse, TrialMean } from '../../src/generated/studioContracts';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { alignMean, pick, showTrials, trialPicker, trialPopup } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const footer = (page: Page) => page.locator('[data-studio-focus-region="bottom"]');
const scene = (page: Page) => page.getByRole('tabpanel', { name: 'Scene', exact: true });
const element = (page: Page, name: string) => scene(page).getByRole('button', { name: `Select ${name}`, exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

/** A Train step's files with its batch size set, which bounds how many samples the clock draws. */
async function trainFiles(batchSize: number, ...scenario: string[]) {
  const files = await fixture(...scenario) as { kind: string; step?: { base: { inline: { training_config: Record<string, unknown> } } } }[];
  files.find(({ kind }) => kind === 'step')!.step!.base.inline.training_config.batch_size = batchSize;
  return files;
}

async function openScene(page: Page, projectId: string, step: string) {
  await page.goto(`/?project=${projectId}`);
  await mapStep(page, step).click();
  await tab(page, 'Scene').click();
  await expect(scene(page).getByRole('group', { name: 'Scene', exact: true })).toBeVisible();
}

/** Where the shown trial's mark is drawn, in the canvas's coordinates. */
const shown = (mark: Locator) => mark.locator('[data-scene-shown]').evaluate((drawn: SVGGraphicsElement) => {
  const box = drawn.getBBox();
  return [box.x + box.width / 2, box.y + box.height / 2].map((coordinate) => Math.round(coordinate * 100) / 100);
});

// Catches a Scene that draws only the mechanism before a run, a task Point placed by name rather than by type, a
// mechanism not posed by the trial's initial values (the point mass apart from the linked start), the start and goal
// drawn alike, and hollow marks that are not one per sample of the picker's count.
test('with nothing run, the Scene draws the nominal trial\'s start and goal and each sample hollow', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await trainFiles(8, 'ordinary-files', '2'));
  await openScene(page, snapshot.project_id, 'train');
  await expect(trialPicker(page)).toHaveText('nominal');
  const start = element(page, 'start');
  const target = element(page, 'target');
  await expect(start.locator('[data-scene-shown]')).toHaveCount(1);
  await expect(target.locator('[data-scene-shown]')).toHaveCount(1);
  await expect(start.locator('[data-scene-shown]')).toHaveAttribute('stroke', 'var(--studio-scene-start)');
  await expect(target.locator('[data-scene-shown]')).toHaveAttribute('stroke', 'var(--studio-scene-goal)');
  await expect(target.locator('[data-scene-possible]')).toHaveCount(8);
  const plant = element(page, 'Point mass');
  await expect.poll(() => shown(plant)).toEqual(await shown(start));
  await page.getByRole('complementary', { name: 'Details Drawer', exact: true }).getByRole('tree', { name: 'Outline', exact: true })
    .getByRole('treeitem', { name: 'target Marker', exact: true }).click();
  await expect(target).toHaveAttribute('aria-pressed', 'true');
});

// Catches a signal drawn whether or not it is present at the playhead, and a Scene that does not follow the clock.
test('a Point signal appears in the Scene once the playhead passes its start', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('delayed-reaches-files'));
  await openScene(page, snapshot.project_id, 'train');
  await expect(trialPicker(page)).toHaveText('nominal');
  const target = element(page, 'target position');
  await expect(element(page, 'start').locator('[data-scene-shown]')).toHaveCount(1);
  await expect(target).toHaveCount(0);
  await footer(page).getByRole('slider', { name: 'Playback Position' }).press('End');
  await expect(target.locator('[data-scene-shown]')).toHaveCount(1);
});

/**
 * Runs the evaluation fixture's Train, its baseline Evaluate and then each of `after`, and opens the Scene of the
 * baseline's Done run.
 */
async function doneEvaluationScene(page: Page, home: string, seed: string, scenario: string, after: readonly string[] = []) {
  const projectId = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', seed, home, scenario));
  const runs = `/api/projects/${projectId}/train-runs`;
  const plans = page.getByRole('table', { name: 'Runs', exact: true });
  const run = async (step: string) => {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const accepted = page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === runs);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await accepted;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0].disposition).toBe('admitted');
    await expect(plans.getByText('Done', { exact: true }).first()).toBeVisible({ timeout: jobTimeout });
  };
  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await run('train');
  await run('baseline');
  for (const step of after) await run(step);
  if (after.length) await mapStep(page, 'baseline').click();
  await plans.getByRole('button', { name: /^\d+ Baseline$/ }).click();
  await tab(page, 'Scene').click();
}

// Catches a recorded trial drawn at its initial pose rather than at the playhead's recorded state, a recorded pose
// drawn without the rest pose beneath it, and a swept Evaluate whose clock offers no recorded trial until one of its
// runs is picked.
test('with a Done evaluation the mechanism at the last step is at its path\'s end', { tag: '@round' }, async ({ page, studio }) => {
  await doneEvaluationScene(page, studio.home, 'seed', 'scene-trial', ['conditions']);
  await expect(trialPicker(page)).toHaveText(/^trial 1 of \d+$/);
  const track = scene(page).locator('[data-scene-track]');
  await expect(track).toHaveCount(1);
  await footer(page).getByRole('slider', { name: 'Playback Position' }).press('End');
  const end = await track.evaluate((line: SVGPolylineElement) => {
    const last = line.points.getItem(line.points.numberOfItems - 1);
    return [last.x, last.y];
  });
  const plant = scene(page).locator('[data-scene-layer="bodies"]').first();
  await expect.poll(async () => {
    const [x, y] = await shown(plant);
    return Math.hypot(x - end[0], y - end[1]);
  }).toBeLessThan(0.5);
  await expect(scene(page).locator('[data-scene-rest] > *')).not.toHaveCount(0);

  await mapStep(page, 'conditions').click();
  await expect(trialPicker(page)).toHaveText(/^trial 1 of \d+$/);
  await trialPicker(page).click();
  await expect(trialPopup(page).getByRole('group', { name: 'Trials', exact: true }).getByRole('button', { name: 'All', exact: true })).toBeEnabled();
  await page.keyboard.press('Escape');
});

// Catches the Scene of a multi-replicate subject never playing (a replay sent without a replicate is refused), and a
// Scene that plays another replicate than the one picked.
test('an Evaluate of a two-replicate Train plays in the Scene, and picking replicate 2 draws that replicate', { tag: '@round' }, async ({ page, studio }) => {
  const replays: { replicate?: number | null; ok: boolean }[] = [];
  page.on('response', (response) => {
    if (response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith('/evaluate-replay')) {
      replays.push({ replicate: response.request().postDataJSON().replicate, ok: response.ok() });
    }
  });
  await doneEvaluationScene(page, studio.home, 'seed-two-replicates', 'scene-replicates');
  const replicate = footer(page).getByRole('combobox', { name: 'Replicate', exact: true });
  await expect(replicate).toHaveText('replicate 1 of 2');
  const track = scene(page).locator('[data-scene-track]');
  await expect(track).toHaveCount(1);
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toHaveCount(0);
  const points = () => track.getAttribute('points');
  const first = await points();
  await pick(replicate, 'replicate 2 of 2');
  await expect.poll(points).not.toBe(first);
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toHaveCount(0);
  expect(replays.length).toBeGreaterThan(1);
  expect(replays.every(({ ok }) => ok)).toBe(true);
  expect(new Set(replays.map(({ replicate }) => replicate))).toEqual(new Set([0, 1]));
});

// Catches All that does not reach recorded trials, a Scene that draws only the picked trial's path under All or draws
// every path alike, and Trails that leave a path drawn past the playhead.
test('All on a Done evaluation draws every trial\'s path, the picked one heavier, and Trails end them at the playhead', { tag: '@round' }, async ({ page, studio }) => {
  await doneEvaluationScene(page, studio.home, 'seed-eight-trials', 'scene-trials');
  await expect(trialPicker(page)).toHaveText('trial 1 of 8');
  await footer(page).getByRole('button', { name: 'Shelf', exact: true }).click();
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  await showTrials(page, 'All');
  await expect(trialPicker(page)).toHaveText('8 trials');
  const tracks = scene(page).locator('[data-scene-track]');
  await expect(tracks).toHaveCount(8);
  const widths = await tracks.evaluateAll((lines) => lines.map((line) => Number(line.getAttribute('stroke-width'))));
  expect(widths.filter((width) => width === Math.max(...widths))).toHaveLength(1);
  await shelf.getByRole('button', { name: 'Trails', exact: true }).click();
  await expect.poll(() => tracks.evaluateAll((lines) => (lines as SVGPolylineElement[]).map((line) => line.points.numberOfItems)))
    .toEqual(Array(8).fill(1));
});

/** The next evaluate-replay reply that answers a Mean request for `align`, with its request. */
const meanReply = (page: Page, align: string | null) => page.waitForResponse((response) => {
  if (new URL(response.url()).pathname.split('/').at(-1) !== 'evaluate-replay') return false;
  const body = response.request().postDataJSON() as { mean?: { align?: string | null } | null };
  return !!body.mean && (body.mean.align ?? null) === align;
}).then(async (response) => ({ request: response.request().postDataJSON(), reply: await response.json() as StudioTrialReplayResponse }));

/** Each condition's distinct world-frame mean paths, as the Scene draws them once each. */
const meanPathCount = (mean: TrialMean) => mean.conditions
  .reduce((count, condition) => count + new Set(Object.values(condition.states).map(({ mean }) => JSON.stringify(mean))).size, 0);

// Catches Mean that does not reach the Service's Mean, a Scene that draws no mean path or spread per condition or
// still poses the mechanism, a clock that stays on the trial's own time rather than Align's or does not reach the
// earliest aligned trial, a mean path that is not the mean of the trials aligned to Align's epoch, a shelf under Mean
// that lists fewer Task rows than one trial's or draws a row empty, By that does not change the conditions, By and
// Align that Escape leaves open, an Align control left on the bar, a clock that overflows the bottom bar of a narrow
// window, and an Evaluate bar whose buttons share the scrubber's line.
test('Mean by condition aligned to an epoch draws one mean path with its spread per condition, on time from that epoch', { tag: '@round' }, async ({ page, studio }) => {
  await doneEvaluationScene(page, studio.home, 'seed-mean-conditions', 'scene-mean');
  const clock = footer(page).getByRole('group', { name: 'Clock', exact: true });
  await footer(page).getByRole('button', { name: 'Shelf', exact: true }).click();
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const rows = shelf.locator('[data-shelf-rows] > [role="group"]');
  await expect(rows).not.toHaveCount(0);
  const oneTrialRows = await rows.evaluateAll((groups) => groups.map((group) => group.getAttribute('aria-label')));
  expect(oneTrialRows.length).toBeGreaterThan(1);
  const opened = meanReply(page, null);
  await showTrials(page, 'Mean');
  const { reply: first } = await opened;
  const options = first.mean_options!;
  const epoch = options.align.at(-1)!;
  const aligned = meanReply(page, epoch.id);
  await alignMean(page, epoch.label);
  const { reply } = await aligned;
  const mean = reply.mean!;
  expect(mean.conditions.length).toBeGreaterThan(1);
  await expect(scene(page).locator('[data-scene-mean]')).toHaveCount(meanPathCount(mean));
  await expect(scene(page).locator('[data-scene-spread]')).toHaveCount(meanPathCount(mean));
  await expect(scene(page).locator('[data-scene-layer="bodies"]')).toHaveCount(0);
  await expect(scene(page).locator('[data-scene-rest] > *')).not.toHaveCount(0);
  const start = Math.min(...mean.conditions.flatMap(({ trials }) => Object.values(trials).map(({ span }) => span.start)));
  expect(start).toBeLessThan(0);
  await expect(footer(page).getByRole('slider', { name: 'Playback Position' })).toHaveAttribute('min', String(start));
  const epochs = shelf.getByRole('group', { name: 'Epochs', exact: true });
  await expect(epochs.getByRole('button', { name: epoch.label, exact: true })).toHaveAttribute('data-start', '0');
  await expect.poll(() => rows.evaluateAll((groups) => groups.map((group) => group.getAttribute('aria-label')))).toEqual(oneTrialRows);
  for (const name of oneTrialRows.slice(1)) await expect(shelf.getByRole('group', { name: name!, exact: true }).locator('[data-share]')).not.toHaveCount(0);

  if (reply.drawn.kind !== 'trials') throw new Error('Mean answered without the trials.');
  const records = new Map(reply.drawn.trials.map((record) => [record.id, record]));
  const [address, summary] = Object.entries(mean.conditions[0].states)[0];
  const at = mean.conditions[0].time.findIndex((time) => Math.abs(time) < 1e-9);
  const samples = Object.entries(mean.conditions[0].trials).map(([id, { origin }]) => {
    const frame = reply.state_time.findIndex((time) => Math.abs(time - origin) < 1e-9);
    return records.get(id)!.states![address].samples[frame] as number[];
  });
  const reference = [0, 1].map((axis) => samples.reduce((sum, sample) => sum + sample[axis], 0) / samples.length);
  expect(summary.mean[at][0]).toBeCloseTo(reference[0], 9);
  expect(summary.mean[at][1]).toBeCloseTo(reference[1], 9);

  const pooled = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith('/evaluate-replay') &&
    (response.request().postDataJSON() as { mean?: { by?: string[] } }).mean?.by?.length === 0);
  const byMenu = trialPopup(page).getByRole('group', { name: 'By', exact: true });
  await trialPicker(page).click();
  for (const { label } of options.by) await byMenu.getByRole('checkbox', { name: label, exact: true }).click();
  const pooledMean = (await (await pooled).json() as StudioTrialReplayResponse).mean!;
  expect(pooledMean.conditions).toHaveLength(1);
  await expect(scene(page).locator('[data-scene-mean]')).toHaveCount(meanPathCount(pooledMean));

  await expect(byMenu).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(byMenu).toBeHidden();
  await expect(footer(page).getByRole('combobox', { name: 'Align', exact: true })).toHaveCount(0);
  await page.setViewportSize({ width: 900, height: 800 });
  await expect.poll(() => clock.evaluate((group) => [...group.querySelectorAll('*')]
    .filter((node) => node.getBoundingClientRect().right > group.closest('footer')!.getBoundingClientRect().right + 0.5).length)).toBe(0);
  const track = (await clock.locator('[data-clock-track]').boundingBox())!;
  for (const button of await footer(page).getByRole('button').all()) {
    expect((await button.boundingBox())!.y).toBeGreaterThanOrEqual(track.y + track.height);
  }
});

// Catches a figure pane that draws its saved product under Mean, or asks for a Mean other than the clock's, a By menu
// that closes, or that Escape leaves open, while the figure is drawn again, a clock whose span is not the figure's
// aligned data, and a hover label whose text is hard to read on its box.
test('under Mean a figure draws the Service\'s Mean rendering for the clock\'s By and Align', { tag: '@round' }, async ({ page, studio }) => {
  await doneEvaluationScene(page, studio.home, 'seed-mean-figure', 'figure-mean', ['analyze']);
  await footer(page).getByRole('button', { name: 'Shelf', exact: true }).click();
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const rows = shelf.locator('[data-shelf-rows] > [role="group"]');
  await expect(rows).not.toHaveCount(0);
  const oneTrialRows = await rows.evaluateAll((groups) => groups.map((group) => group.getAttribute('aria-label')));
  expect(oneTrialRows.length).toBeGreaterThan(1);
  const opened = meanReply(page, null);
  await showTrials(page, 'Mean');
  const epoch = (await opened).reply.mean_options!.align.at(-1)!;
  const aligned = meanReply(page, epoch.id);
  await alignMean(page, epoch.label);
  await aligned;

  const meanFigure = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith('/figures/mean'));
  await mapStep(page, 'analyze').click();
  await tab(page, 'Figures').click();
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  await figures.getByRole('button', { name: 'By trial', exact: true }).click();
  const response = await meanFigure;
  expect(response.status(), await response.text()).toBe(200);
  expect(response.request().postDataJSON()).toMatchObject({ mean: { by: null, align: epoch.id } });
  const product = await response.json() as { data: unknown[] };
  await expect(figures.locator('g.scatterlayer g.trace')).toHaveCount(product.data.length);

  // Pooled, the trials whose epochs start at different times share one condition, whose mean covers only their
  // common time; the clock still spans every trial the figure draws.
  const byMenu = trialPopup(page).getByRole('group', { name: 'By', exact: true });
  await trialPicker(page).click();
  await expect(byMenu.getByRole('checkbox').first()).toBeVisible();
  let pooled: Awaited<typeof meanFigure> | undefined;
  for (const item of await byMenu.getByRole('checkbox').all()) {
    const redrawn = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith('/figures/mean'));
    await item.click();
    pooled = await redrawn;
  }
  expect(pooled!.request().postDataJSON()).toMatchObject({ mean: { by: [], align: epoch.id } });
  await expect(byMenu).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(byMenu).toBeHidden();
  const times = (await pooled!.json() as { data: { x: number[] }[] }).data.flatMap(({ x }) => x);
  const slider = footer(page).getByRole('slider', { name: 'Playback Position' });
  await expect.poll(async () => Number(await slider.getAttribute('min'))).toBeCloseTo(Math.min(...times), 6);
  expect(Number(await slider.getAttribute('max'))).toBeCloseTo(Math.max(...times), 6);

  const plot = figures.locator('.js-plotly-plot');
  const centre = await plot.evaluate((node) => {
    const { xaxis, yaxis } = (node as unknown as { _fullLayout: Record<'xaxis' | 'yaxis', { _offset: number; _length: number }> })._fullLayout;
    const box = node.getBoundingClientRect();
    return { x: box.left + xaxis._offset + xaxis._length / 2, y: box.top + yaxis._offset + yaxis._length / 2 };
  });
  await page.mouse.move(centre.x, centre.y);
  await page.mouse.move(centre.x + 4, centre.y);
  const label = plot.locator('.hoverlayer g.hovertext, .hoverlayer g.legend').first();
  await expect(label).toBeVisible();
  const contrast = await label.evaluate((node) => {
    const luminance = (color: string) => {
      const [r, g, b] = color.match(/[\d.]+/g)!.slice(0, 3).map((value) => {
        const channel = Number(value) / 255;
        return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const box = node.querySelector('path, rect')!;
    const text = node.querySelector('text')!;
    const [light, dark] = [luminance(getComputedStyle(text).fill), luminance(getComputedStyle(box).fill)].sort((a, b) => b - a);
    return (light + 0.05) / (dark + 0.05);
  });
  expect(contrast).toBeGreaterThan(4.5);
});
