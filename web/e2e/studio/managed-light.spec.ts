import { expect, isolatedTest, test, type Studio, jobTimeout } from './fixtures/studio';
import { type APIRequestContext, type Locator, type Page, type TestInfo } from '@playwright/test';
import type {
  AnalysisDocument, ReportDocument, StudioJobSnapshot, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot,
  StudioProjectWindowMap, StudioSavedLightResult, StudioSavedTrainKey, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { authorEvaluateConditions } from './fixtures/authorEvaluateConditions';
import { enter, mapStep, runName, shownRunLabel, runProjectFixture, mapCard } from './fixtures/canonicalTrain';
import { authorAndRecoverTrain } from './fixtures/firstTrainLifecycle';
import { authorWalkthroughSweep, selectModelComponent, verifyWalkthroughSuccessors, verifyWalkthroughVariant } from './fixtures/managedLightWalkthrough';
import { activate, chooseOption, chooseStepKind, number, openView, palette, tabTo, typeValue } from './fixtures/studioKeyboard';

import { applyTo, pick, popupListbox } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

type WalkthroughContext = { page: Page; request: APIRequestContext; home: string };
type PlotSamples = number[] | { dtype: string; bdata: string };
type PlotTrace = { name: string; x: PlotSamples; y: PlotSamples; showlegend?: boolean };

function traceSamples(encoded: PlotSamples) {
  if (Array.isArray(encoded)) return encoded;
  const data = Buffer.from(encoded.bdata, 'base64');
  expect(encoded.dtype).toMatch(/^f[48]$/);
  const width = encoded.dtype === 'f4' ? 4 : 8;
  return Array.from({ length: data.length / width }, (_, index) =>
    width === 4 ? data.readFloatLE(index * width) : data.readDoubleLE(index * width));
}

async function expectSeparateVelocityLines(figure: Locator) {
  await expect.poll(() => figure.locator('g.scatterlayer g.lines path.js-line').evaluateAll((lines) => {
    const visible = lines.filter((line) => {
      const style = getComputedStyle(line);
      const bounds = line.getBoundingClientRect();
      return style.stroke !== 'none' && Number(style.strokeOpacity) > 0 && bounds.width > 0 && bounds.height > 0;
    });
    return { count: visible.length, shapes: new Set(visible.map((line) => line.getAttribute('d'))).size };
  })).toEqual({ count: 2, shapes: 2 });
}

async function savedLightGate(context: WalkthroughContext, info: TestInfo, studio?: Studio) {
  const start = performance.now();
  const first = studio ? await test.step('Author, Stop, Resume and recover the first Train', () =>
    authorAndRecoverTrain(context.page, context.request, info, studio)) : undefined;
  await exerciseSavedResults(context, info, first);
  await info.attach('Walkthrough duration', {
    body: JSON.stringify({ milliseconds: performance.now() - start }), contentType: 'application/json',
  });
}

async function exerciseSavedResults({ page, request, home: suiteHome }: WalkthroughContext, info: TestInfo,
  first?: Awaited<ReturnType<typeof authorAndRecoverTrain>>,
) {
  const paired = first !== undefined;
  const home = first?.home ?? suiteHome;
  const fixture = 'web.e2e.studio.fixtures.managed_light';
  const project: string = first?.project ?? JSON.parse(await runProjectFixture(fixture, 'seed', home, 'managed-light'));
  const scope = paired ? 'base' : 'default';
  const path = first?.projectPath ?? '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const mounted = '/?project=' + encodeURIComponent(project);
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const document = page.getByRole('tabpanel', { name: 'Document', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const boundsScale = inspector.getByRole('textbox', { name: 'Bounds Scale', exact: true });
  const mutation = (method: string, route: string) => page.waitForResponse((response) =>
    response.request().method() === method && new URL(response.url()).pathname === route);

  async function snapshot() {
    const response = await request.get(path + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  }
  async function plan(revision: string) {
    const response = await request.get(runPath, { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  async function edit(action: () => Promise<unknown>) {
    const saving = mutation('PATCH', path);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return await response.json() as StudioProjectSnapshot;
  }
  async function attempts() {
    const response = await request.get('/api/jobs', { params: { project } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioJobSnapshot).jobs.map(({ job_id }) => job_id).sort();
  }
  async function runSaved(reused?: readonly string[]) {
    const admitted = mutation('POST', runPath);
    await palette(page, 'run', 'Run');
    const response = await admitted;
    expect(response.status(), await response.text()).toBe(202);
    expect(Boolean(response.request().postDataJSON().reuse_completed)).toBe(reused !== undefined);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.filter(({ disposition }) => disposition === 'rejected')).toEqual([]);
    if (reused) {
      expect(items.every(({ disposition }) => disposition === 'reused')).toBe(true);
      expect(items.map(({ job_id }) => job_id).sort()).toEqual(reused);
    }
    return items.filter(({ disposition }) => disposition === 'admitted').map(({ job_id }) => job_id!);
  }
  async function runStep(step: string, doneCount = 1, expected?: readonly StudioSavedTrainKey[], selected: readonly string[] = []) {
    await activate(page, mapStep(page, step));
    await openView(page, 'runs');
    const table = page.getByRole('table', { name: 'Runs', exact: true });
    for (const [index, label] of selected.entries()) {
      await activate(page, table.getByRole('button', { name: runName(label) }), index > 0 ? 'Shift+Enter' : 'Enter');
    }
    const admitted = mutation('POST', runPath);
    await palette(page, 'run', 'Run');
    const response = await admitted;
    expect(response.status(), await response.text()).toBe(202);
    if (expected) expect(response.request().postDataJSON().run_keys).toEqual(expected);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.map(({ disposition, reason }) => [disposition, reason ?? null])).toEqual(
      Array.from({ length: expected?.length ?? doneCount }, () => ['admitted', null]),
    );
    const watched = new Set(items.map(({ job_id }) => job_id));
    await expect.poll(async () => {
      const jobs = await request.get('/api/jobs', { params: { project } });
      const history = await jobs.json() as StudioJobSnapshot;
      return history.jobs.filter(({ job_id }) => watched.has(job_id)).map(({ status }) => status);
    }, { timeout: jobTimeout + 8_000 * doneCount }).toEqual(Array(items.length).fill('succeeded'));
    await expect(table.getByText('Done', { exact: true })).toHaveCount(doneCount);
  }

  const walkthrough = { page, request, projectPath: path, attempts };
  const pairedInput = first ? await test.step('Author two Evaluate conditions from the resumed checkpoint', async () => {
    const [producer, ...otherRows] = await authorWalkthroughSweep(walkthrough);
    expect(producer.result).toEqual(first.trained);
    await runStep('train', 3, otherRows.map(({ key }) => key), otherRows.map(shownRunLabel));
    await activate(page, page.getByRole('table', { name: 'Runs', exact: true })
      .getByRole('button', { name: runName(shownRunLabel(producer)) }));
    return authorEvaluateConditions({
      page, request, projectPath: path, train: producer.key, field: 'Mass',
      conditions: [1, 2],
      selectComponent: () => selectModelComponent(page, first.plantId),
    });
  }) : undefined;
  if (!first) {
    await page.goto(mounted);
    await test.step('Run Train', () => runStep('train'));
  }
  const evaluateStep = pairedInput?.stepId ?? 'conditions';
  await test.step('Run both Evaluate conditions', () => runStep(evaluateStep, 2, pairedInput?.rows.map(({ key }) => key)));
  const initial = await snapshot();
  const initialPlan = await plan(initial.revision);
  const settledEvaluate = initialPlan.evaluations!.filter(({ key }) => key.variant === scope && key.step === evaluateStep);
  if (pairedInput) expect(settledEvaluate.map(({ key }) => key)).toEqual(pairedInput.rows.map(({ key }) => key));
  else expect(settledEvaluate.map(({ key }) => key.row).sort()).toEqual(['mass=heavy', 'mass=nominal']);
  expect(new Set(settledEvaluate.map(({ request_id }) => request_id)).size).toBe(2);
  expect(new Set(settledEvaluate.map(({ result }) => result?.job_id)).size).toBe(2);
  expect(settledEvaluate.map(({ status }) => status)).toEqual(['succeeded', 'succeeded']);
  const [firstCondition, secondCondition] = settledEvaluate;
  expect(firstCondition.trials.length).toBeGreaterThan(0);
  expect(firstCondition.trials).toEqual(secondCondition.trials);
  const producer = await request.get(runPath, { params: { revision: initial.revision } });
  expect(producer.ok(), await producer.text()).toBe(true);
  const training = (await producer.json() as StudioSavedTrainRuns).rows.find(({ key, result }) =>
    key.variant === scope && key.step === 'train' && (first ? result?.job_id === first.trained.job_id : key.row === 'default'))!;
  for (const condition of settledEvaluate) {
    expect(condition.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(condition.result?.checkpoint_source).toMatchObject({
      job_id: training.result!.job_id, request_id: training.request_id,
      completed_updates: training.result!.progress!.completed_updates,
    });
  }
  const sources = paired ? settledEvaluate : [settledEvaluate.find(({ key }) => key.row === 'mass=heavy')!];
  const admittedJobs = await attempts();
  expect(admittedJobs).toHaveLength((first ? 4 : 1) + settledEvaluate.length);
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  const rowLabel = (key: StudioSavedTrainKey) =>
    shownRunLabel(initialPlan.rows.find((row) => JSON.stringify(row.key) === JSON.stringify(key))!);
  const sourceLabels = pairedInput?.rows.map(shownRunLabel) ?? [rowLabel(sources[0].key)];
  await activate(page, table.getByRole('button', { name: runName(sourceLabels[0]) }));
  if (paired) await activate(page, table.getByRole('button', { name: runName(sourceLabels[1]) }), 'Shift+Enter');
  await palette(page, 'newStep', 'New Step');
  await chooseStepKind(page, 'Analyze', true);
  await expect(analysis).toBeVisible();
  const created = await snapshot();
  expect(created.project.steps?.analyze).toMatchObject({
    kind: 'analyze', base: { document: 'analysis' }, inputs: { states: {
      step: evaluateStep,
      ...(!paired && { where: { clauses: [{ mass: [{ id: 'heavy', label: expect.any(String) }] }] } }),
    } },
  });
  function expectAnalyzeSources(projection: StudioProjectWindowMap) {
    const edges = projection.edges.filter(({ to_key }) => to_key.variant === scope && to_key.step === 'analyze');
    expect(edges.map(({ from_key }) => from_key)).toEqual(sources.map(({ key }) => key));
  }
  expectAnalyzeSources(await plan(created.revision));

  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  /** Searches the Library and places the item: Tab reaches the first match, the arrows the others. */
  async function place(label: string) {
    await typeValue(page, drawer.getByRole('textbox', { name: 'Search Library', exact: true }), label);
    const options = drawer.getByRole('option');
    const option = drawer.getByRole('option', { name: label, exact: true });
    await tabTo(page, options.and(drawer.locator('[tabindex="0"]')));
    for (let index = await options.count(); index > 0 && !await option.evaluate((element) => element === element.ownerDocument.activeElement); index -= 1) {
      await page.keyboard.press('ArrowDown');
    }
    await expect(option).toBeFocused();
    return edit(() => page.keyboard.press('Enter'));
  }
  const selectElement = (name: string) => activate(page, analysis.getByRole('group', { name: 'Select ' + name, exact: true }));
  await place('Aligned velocity');
  await selectElement('Aligned velocity');
  if (!await inspector.isVisible()) await palette(page, 'showInspector', 'Show Inspector');
  // The walkthrough's Task names its target signal; the selected-row project's Task leaves it unnamed, shown by its ID.
  for (const [field, label] of [
    ['Position', / · effector position$/], ['Velocity', / · effector velocity$/], ['Target', /^target( position|_pos)$/],
  ] as const) {
    const select = inspector.getByRole('combobox', { name: field, exact: true });
    await edit(() => chooseOption(page, select, label));
  }
  const roleField = (name: string) => inspector.getByRole('combobox', { name, exact: true });
  await edit(() => chooseOption(page, roleField('States'), 'states'));

  await place('Trial summary');
  await selectElement('Trial summary');
  await edit(() => number(page, boundsScale, '1.5'));
  await edit(() => number(page, inspector.getByRole('textbox', { name: 'Ddof', exact: true }), '0'));
  // A role offers only the sources that fit it, by keyboard too: typing `states` finds nothing for a velocity.
  await tabTo(page, roleField('Velocity'));
  await page.keyboard.press('ArrowDown');
  // The popup's search field takes focus after it opens; keys typed before then are lost.
  await expect(page.locator('input').and(page.getByRole('combobox', { name: 'Velocity', exact: true }))).toBeFocused();
  await page.keyboard.type('states');
  await expect(popupListbox(page).getByRole('option')).toHaveCount(0);
  await page.keyboard.press('Escape');
  // Closing the popup gives focus back to its field once it has closed; the next choice starts from there.
  await expect(popupListbox(page)).toBeHidden();
  await expect(roleField('Velocity')).toBeFocused();
  await edit(() => chooseOption(page, roleField('Velocity'), 'Aligned velocity · velocity'));

  await place('Velocity Summary');
  await selectElement('Velocity Summary');
  const wired = await edit(() => chooseOption(page, roleField('Summary'), 'Trial summary · summary'));
  await expect(analysis.locator('.react-flow__edge')).toHaveCount(3);
  if (first) {
    const authored = rootPayload(wired.project.documents!.analysis) as unknown as AnalysisDocument;
    const state = '/v1/model/state/' + first.plantId + '/';
    expect(authored.operations).toMatchObject({
      'aligned-velocity': {
        operation: 'feedbax.aligned_velocity',
        params: { position: state + 'effector~1position', velocity: state + 'effector~1velocity', target: '/v1/task/output/target_pos' },
        inputs: { states: { input: 'states' } },
      },
      'trial-summary': {
        operation: 'feedbax.summarize_velocity', params: { bounds_scale: 1.5, ddof: 0 },
        inputs: { velocity: { element: 'aligned-velocity', product: 'velocity' } },
      },
    });
  }
  const original = await plan(wired.revision);
  const requests = (projection: typeof original) => Object.fromEntries(
    projection.analysis_elements!.filter(({ key }) => key.variant === scope && key.step === 'analyze')
      .map(({ name, requests: [{ request_id }] }) => [name, request_id]),
  );
  expect(Object.keys(requests(original)).sort()).toEqual(['aligned-velocity', 'trial-summary', 'velocity-summary']);
  expect(original.problems.filter(({ step }) => step === 'analyze')).toEqual([]);
  expectAnalyzeSources(original);
  await openView(page, 'figures');
  await activate(page, figures.getByRole('button', { name: 'Velocity Summary', exact: true }));
  await palette(page, 'newStep', 'New Step');
  await chooseStepKind(page, 'Report', true);
  await expect(document.getByRole('textbox', { name: 'Report Title', exact: true })).toBeVisible();
  await palette(page, 'addText', 'Add Text');
  const addText = page.getByRole('dialog', { name: 'Add Text', exact: true });
  await typeValue(page, addText.getByRole('textbox', { name: 'Text', exact: true }), '**Velocity** by condition.');
  await edit(() => activate(page, addText.getByRole('button', { name: 'Add Text', exact: true })));
  const ordered = await edit(() => palette(page, 'moveEarlier', 'Move Earlier 2'));
  const report = rootPayload(ordered.project.documents!.report) as unknown as ReportDocument;
  const inputs = ordered.project.steps!.report.inputs!;
  expect(report.blocks[0]).toMatchObject({ markdown: '**Velocity** by condition.' });
  expect(report.blocks).toHaveLength(2);
  const role = Object.keys(inputs)[0];
  expect(report.blocks[1]).toMatchObject({ figure: role });
  expect(inputs[role]).toMatchObject({ step: 'analyze', element: 'velocity-summary', product: 'figure' });

  const settled = await plan(ordered.revision);
  expect(settled.problems.filter(({ step }) => step === 'analyze' || step === 'report')).toEqual([]);
  expect(requests(settled)).toEqual(requests(original));
  const reportRequest = settled.requests?.find(({ key }) => key.variant === scope && key.step === 'report');
  expect(reportRequest?.request_id).toMatch(/^report:/);
  expect(settled.edges).toContainEqual(expect.objectContaining({
    from_key: expect.objectContaining({ step: 'analyze' }), to_key: expect.objectContaining({ step: 'report' }),
  }));
  expect(await attempts()).toEqual(admittedJobs);

  function light(projection: StudioProjectWindowMap) {
    const elements = projection.analysis_elements!.filter(({ key }) => key.variant === scope);
    const element = (name: string) => {
      const found = elements.find((item) => item.name === name)!;
      return { ...found, ...found.requests[0] };
    };
    return {
      aligned: element('aligned-velocity'),
      summary: element('trial-summary'),
      figure: element('velocity-summary'),
      report: projection.rows.find(({ key }) => key.variant === scope && key.step === 'report')!,
    };
  }
  function result(projection: StudioProjectWindowMap) {
    const { aligned, summary, figure, report } = light(projection);
    return [aligned.result!, summary.result!, figure.result!, report.light!.result!];
  }
  function identity(value?: StudioSavedLightResult | null) {
    return value && {
      job: value.job_id, request: value.request_id, digest: value.receipt.manifest_sha256,
    };
  }
  function lightStatus(projection: StudioProjectWindowMap) {
    const { aligned, summary, figure, report } = light(projection);
    return [aligned, summary, figure, { ...report, ...report.light }].map((entry) => ({
      key: entry.key, status: entry.status, reason: entry.reason, changed: entry.changed,
      current: identity(entry.result), previous: identity(entry.previous),
    }));
  }
  async function matchesCli(projection: StudioProjectWindowMap) {
    await test.step('Compare saved result status with CLI', async () => {
      const cli = JSON.parse(await runProjectFixture(fixture, 'status', home, project)) as StudioSavedTrainRuns;
      expect(lightStatus(cli.window_map!)).toEqual(lightStatus(projection));
    });
  }
  async function waitForLight(revision: string) {
    // The report runs last, so its Done status in the open Document view marks the whole chain.
    await expect(document.getByRole('status').filter({ hasText: /^Done/ })).toBeVisible({ timeout: jobTimeout });
    let projection: StudioProjectWindowMap | undefined;
    await expect.poll(async () => lightStatus(projection = await plan(revision)).map(({ status }) => status))
      .toEqual(['succeeded', 'succeeded', 'succeeded', 'succeeded']);
    return projection!;
  }
  /** Fetching a saved figure, loading Plotly and drawing it is the work these waits see through; under load it takes seconds. */
  const drawing = 20_000;
  async function showReport(url: string) {
    await expect(document.getByRole('textbox', { name: 'Text', exact: true })).toHaveValue('**Velocity** by condition.');
    const result = document.getByRole('button', { name: 'Result', exact: true });
    await activate(page, result);
    await expect(result).toHaveAttribute('aria-expanded', 'true');
    const frame = document.locator('iframe[aria-label="Report"]');
    await expect(frame).toHaveAttribute('src', url);
    const rendered = frame.contentFrame();
    await expect(rendered.getByText('Velocity', { exact: true })).toBeVisible();
    const reportFigure = rendered.locator('[data-report-source] svg.main-svg').first();
    await expect(reportFigure).toBeVisible({ timeout: drawing });
    if (paired) await expectSeparateVelocityLines(reportFigure);
  }
  async function showFigure() {
    await activate(page, mapStep(page, 'analyze'));
    await openView(page, 'figures');
    await activate(page, figures.getByRole('button', { name: 'Velocity Summary', exact: true }));
  }
  async function drawnFigure(name: 'Current Result' | 'Previous Result') {
    const region = figures.getByRole('region', { name, exact: true });
    await expect(region.locator('[role="img"]:not([aria-busy])')).toBeVisible({ timeout: drawing });
    const plot = region.locator('svg.main-svg').first();
    await expect(plot).toBeVisible();
    if (paired) await expectSeparateVelocityLines(plot);
  }

  async function showCurrentResults(reportUrl: string) {
    await showReport(reportUrl);
    await showFigure();
    await drawnFigure('Current Result');
  }

  const createdJobs = await runSaved();
  expect(createdJobs).toHaveLength(4);
  const completed = await test.step('Wait for Analyze, figure and Report', () => waitForLight(ordered.revision));
  const [aligned, summary, figure, renderedReport] = result(completed);
  expect(result(completed).map(({ job_id }) => job_id).sort()).toEqual(createdJobs.sort());
  for (const { receipt } of result(completed)) expect(receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(figure.request_id).toBe(requests(completed)['velocity-summary']);
  expect(renderedReport.request_id).toBe(reportRequest?.request_id);
  const firstFigure = await request.get(figure.products.figure);
  expect(firstFigure.ok(), await firstFigure.text()).toBe(true);
  const figureBody = await firstFigure.body();
  await Promise.all([matchesCli(completed), showCurrentResults(renderedReport.products.report)]);

  await page.reload();
  await activate(page, mapStep(page, 'report'));
  await openView(page, 'document');
  await showCurrentResults(renderedReport.products.report);
  expect((await snapshot()).revision).toBe(ordered.revision);
  expectAnalyzeSources(await plan(ordered.revision));
  const plot: { data: PlotTrace[] } = await firstFigure.json();
  const traces = plot.data.filter(({ showlegend }) => showlegend !== false);
  expect(traces.map(({ name }) => name)).toEqual(sourceLabels);
  if (!first) return;

  await test.step('Compare the paired figure and Report with the saved dataset', async () => {
    const memberKeys = sources.map(({ key }) => initialPlan.rows.find(({ key: candidate }) =>
      candidate.variant === key.variant && candidate.step === key.step && candidate.row === key.row)?.member_key);
    const retained = JSON.parse(await runProjectFixture(
      'web.e2e.studio.fixtures.managed_light_walkthrough', home, project, aligned.job_id, figure.job_id,
    ));
    expect(retained.members).toEqual(sources.map(({ result, request_id }, index) => ({
      key: memberKeys[index], job_id: result!.job_id, request_id,
      evaluation_manifest_sha256: result!.receipt.manifest_sha256,
      store_success_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
    })));
    expect(new Set(retained.members.map(({ store_success_sha256 }: { store_success_sha256: string }) => store_success_sha256)).size).toBe(2);
    expect(retained.producers.sort()).toEqual(sources.map(({ result }) => result!.job_id).sort());
    const condition = { seed: 0, trial_policy: 'paired', trials: firstCondition.trials };
    expect(retained.condition_inputs).toEqual([
      { ...condition, variation: [] },
      { ...condition, variation: [{ op: 'replace', path: `graph.nodes.${first.plantId}.params.mass`, value: 2 }] },
    ]);
    const samples = traces.map(({ name, x, y }) => ({ name, x: traceSamples(x), y: traceSamples(y) }));
    expect(samples.map(({ x, y }) => ({ x, y }))).toEqual(retained.figure_samples);
    const [nominal, heavy] = samples;
    expect(nominal.x).toHaveLength(first.task.fixed_steps! + 1);
    expect(nominal.x).toEqual(heavy.x);
    // Softmax pushes (1/2, 1/2) at unit mass, so the speed along the one reach grows by its share of that per second.
    const [[startX, startY], [targetX, targetY]] = [first.reach.variables!.start, first.reach.variables!.target] as number[][];
    const along = (targetX - startX + targetY - startY) / (2 * Math.hypot(targetX - startX, targetY - startY));
    for (const [index, time] of nominal.x.entries()) {
      expect(nominal.y[index]).toBeCloseTo(time * along, 6);
      expect(heavy.y[index]).toBeCloseTo(time * along / 2, 6);
    }
    expect(heavy.y.at(-1)!).toBeGreaterThan(0.04);
    expect(nominal.y.at(-1)! - heavy.y.at(-1)!).toBeGreaterThan(0.04);
    await info.attach('Nominal and Heavy velocity', {
      body: JSON.stringify({ Nominal: nominal, Heavy: heavy }), contentType: 'application/json',
    });
    const html = await request.get(renderedReport.products.report);
    expect(html.ok(), await html.text()).toBe(true);
    const member = /figures\/[a-f0-9]{64}\.json/.exec(await html.text())?.[0];
    expect(member).toBeDefined();
    const embedded = await request.get(renderedReport.products.report.replace('report.html', member!));
    expect(embedded.ok(), await embedded.text()).toBe(true);
    expect(await embedded.body()).toEqual(figureBody);
  });

  const beforeEdit = [...admittedJobs, ...createdJobs].sort();
  await activate(page, mapStep(page, 'train'));
  await activate(page, mapStep(page, 'train'), 'Meta+Enter');
  expect(await runSaved(beforeEdit.filter((job) => job !== first?.stopped))).toEqual([]);
  expect(await attempts()).toEqual(beforeEdit);

  await activate(page, mapStep(page, 'analyze'));
  await openView(page, 'analysis');
  await selectElement('Trial summary');
  if (!await inspector.isVisible()) await palette(page, 'showInspector', 'Show Inspector');
  const changed = await edit(() => number(page, boundsScale, '2'));
  const stale = await plan(changed.revision);
  const [stillCurrent, staleSummary, staleFigure, staleReport] = lightStatus(stale);
  expect(stillCurrent.current).toEqual(identity(aligned));
  expect(staleSummary.changed).toMatchObject({
    path: expect.stringContaining('bounds_scale'), place: { part: 'analysis', element: 'trial-summary' },
  });
  expect([staleSummary, staleFigure, staleReport].map(({ status, previous }) => [status, previous]))
    .toEqual([summary, figure, renderedReport].map((entry) => ['stale', identity(entry)]));
  await Promise.all([
    matchesCli(stale),
    (async () => {
      await openView(page, 'figures');
      await activate(page, figures.getByRole('button', { name: 'Velocity Summary', exact: true }));
      await drawnFigure('Previous Result');
    })(),
  ]);
  expect(await attempts()).toEqual(beforeEdit);

  await openView(page, 'analysis');
  const undo = await edit(() => palette(page, 'undo', 'Undo'));
  expect(lightStatus(await plan(undo.revision))).toEqual(lightStatus(completed));
  expect(await attempts()).toEqual(beforeEdit);

  await selectElement('Trial summary');
  const reapplied = await edit(() => number(page, boundsScale, '2'));
  await activate(page, mapStep(page, 'report'));
  await openView(page, 'document');
  const rerun = await runSaved();
  expect(rerun).toHaveLength(3);
  const replaced = await test.step('Wait for the changed light results', () => waitForLight(reapplied.revision));
  const [alignedAgain, newSummary, newFigure, newReport] = result(replaced);
  expect(identity(alignedAgain)).toEqual(identity(aligned));
  expect([newSummary, newFigure, newReport].map(({ job_id }) => job_id).sort()).toEqual(rerun.sort());
  expect(await attempts()).toEqual([...beforeEdit, ...rerun].sort());
  const previous = light(replaced).figure.previous!;
  expect(identity(previous)).toEqual(identity(figure));
  expect(previous.products.figure).not.toBe(figure.products.figure);
  const previousProduct = await request.get(previous.products.figure);
  expect(previousProduct.ok(), await previousProduct.text()).toBe(true);
  expect(await previousProduct.body()).toEqual(figureBody);
  await Promise.all([matchesCli(replaced), showCurrentResults(newReport.products.report)]);
  await activate(page, figures.getByRole('button', { name: 'Previous Result', exact: true }));
  await drawnFigure('Previous Result');
  await test.step('Share the full chain and isolate a compare Model edit', () => verifyWalkthroughVariant(walkthrough, first.plantId));
  await test.step('Widen the Evaluate subset and continue the exact Train rows', () => verifyWalkthroughSuccessors(walkthrough, evaluateStep));
}

test('runs and reopens saved figures and Report from a selected Evaluate row', { tag: '@round' }, ({ page, request, studio }, info) =>
  savedLightGate({ page, request, home: studio.home }, info));

isolatedTest('walks one saved project through paired results, variants, widened Evaluate and chained Train', { tag: '@round' }, ({ page, request, studio }, info) =>
  savedLightGate({ page, request, home: studio.home }, info, studio));

test('authors a paired Analyze and shared successors from the primary variant', { tag: '@round' }, async ({ page, request, studio }, info) => {
  const home = studio.home;
  const project = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.managed_light', 'seed-variants', home, 'managed-light-variants',
  ));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const variants = page.getByRole('combobox', { name: 'Variant', exact: true });
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const train = mapCard(page, 'train');
  async function snapshot() {
    const response = await request.get(path);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function plan() {
    const saved = await snapshot();
    const response = await request.get(runPath, { params: { revision: saved.revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  async function openRuns(step: string) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
  }
  async function runStep(step: string, count: number) {
    await openRuns(step);
    const admitted = page.waitForResponse((response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admitted;
    expect(response.status(), await response.text()).toBe(202);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.map(({ disposition }) => disposition)).toEqual(Array(count).fill('admitted'));
    await expect(table.getByText('Done', { exact: true })).toHaveCount(count, { timeout: jobTimeout });
  }
  async function selectPair() {
    await openRuns('conditions');
    const rows = (await plan()).window_map!.rows;
    const label = (row: string) => shownRunLabel(rows.find(({ key }) => key.step === 'conditions' && key.row === row)!);
    const nominal = table.getByRole('button', { name: runName(label('mass=nominal')) });
    const heavy = table.getByRole('button', { name: runName(label('mass=heavy')) });
    await nominal.click();
    await heavy.click({ modifiers: ['Shift'] });
    await expect(nominal).toHaveAttribute('aria-pressed', 'true');
    await expect(heavy).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Meta+K');
    await palette.getByRole('combobox').fill('New Step');
  }
  async function successor(kind: 'Train' | 'Evaluate', id: string) {
    await openRuns('train');
    const creation = page.waitForResponse((response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === path + '/files'
      && response.request().postDataJSON().file.id === id);
    await train.getByRole('button', { name: 'New Step Train', exact: true }).click();
    await chooseStepKind(page, kind);
    const created = await creation;
    expect(created.status(), await created.text()).toBe(201);
    await expect(mapStep(page, id)).toHaveAttribute('aria-pressed', 'true');
    return (await created.json() as StudioProjectDocument).snapshot;
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded', timeout: jobTimeout });
  await expect(variants).toHaveText('base');
  const initial = await snapshot();
  await runStep('train', 1);
  await runStep('conditions', 2);
  const settled = (await plan()).window_map!.evaluations!.filter(({ key }) => key.step === 'conditions');
  const base = settled.filter(({ key }) => key.variant === 'base');
  const compare = settled.filter(({ key }) => key.variant === 'compare');
  expect(base).toHaveLength(2);
  expect(compare.map(({ result }) => result?.receipt)).toEqual(base.map(({ result }) => result?.receipt));
  const [first, second] = base;
  for (const { result } of base) expect(result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(first.result!.receipt).not.toEqual(second.result!.receipt);
  expect(first.result!.checkpoint_source).toBeTruthy();
  expect(second.result!.checkpoint_source).toEqual(first.result!.checkpoint_source);
  expect(first.trials.length).toBeGreaterThan(0);
  expect(second.trials).toEqual(first.trials);

  await pick(variants, 'compare');
  await selectPair();
  await expect(palette.getByRole('option', { name: 'New Step', exact: true })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await openRuns('train');
  await expect(train.getByRole('button', { name: 'New Step Train', exact: true })).toHaveCount(0);

  await pick(variants, 'base');
  await selectPair();
  const creating = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === path + '/files'
    && response.request().postDataJSON().file.kind === 'step');
  await palette.getByRole('option', { name: 'New Step', exact: true }).press('Enter');
  await chooseStepKind(page, 'Analyze', true);
  const created = await creating;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: paired } = await created.json() as StudioProjectDocument;
  await expect(page.getByRole('tabpanel', { name: 'Analysis', exact: true })).toBeVisible();
  expect(paired.project.steps?.analyze).toMatchObject({
    kind: 'analyze', inputs: { states: { step: 'conditions' } },
  });
  expect(paired.project.sweep).toEqual(initial.project.sweep);
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('Aligned velocity');
  await drawer.getByRole('option', { name: 'Aligned velocity', exact: true }).dblclick();
  await expect.poll(async () => rootPayload((await snapshot()).project.documents?.analysis))
    .toMatchObject({ operations: { 'aligned-velocity': { operation: 'feedbax.aligned_velocity' } } });

  const evaluated = await successor('Evaluate', 'evaluate');
  const stepInspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(stepInspector.getByRole('combobox', { name: 'Subject', exact: true })).toBeFocused();
  expect(evaluated.project.steps?.evaluate).toMatchObject({
    kind: 'evaluate', inputs: { subject: { step: 'train', checkpoint: 'last' } },
  });
  const childId = 'train-2';
  const continuation = await successor('Train', childId);
  const batches = stepInspector.getByRole('textbox', { name: 'Batches', exact: true });
  await expect(batches).toBeEditable();
  const prepared = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === path + '/authoring/train/' + childId + '/fields'
    && response.request().postDataJSON().edit?.text === '4');
  const saving = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === path
    && response.request().postDataJSON().address?.id === childId);
  void saving.catch(() => {});
  await enter(batches, '4');
  const recipe = await prepared;
  expect(recipe.ok(), await recipe.text()).toBe(true);
  expect(recipe.request().postDataJSON()).toMatchObject({ revision: continuation.revision });
  // The compare variant reads the continuation too, so the edit asks where it applies.
  await applyTo(page).getByRole('menuitem', { name: 'Whole Step', exact: true }).click();
  const saved = await saving;
  expect(saved.ok(), await saved.text()).toBe(true);
  const edited = await saved.json() as StudioProjectSnapshot;
  const child = edited.project.steps?.[childId];
  expect(child).toMatchObject({
    base: { step: 'train' }, inputs: { checkpoint: { step: 'train', checkpoint: 'last' } },
  });
  expect(child?.deltas?.at(-1)?.patches).toEqual([
    { op: 'replace', path: 'training_config.n_batches', value: 4 },
  ]);
  expect(edited.project.steps?.train).toEqual(initial.project.steps?.train);
  expect(edited.project.sweep).toEqual(initial.project.sweep);
  const finished = await plan();
  expect(finished.revision).toBe(edited.revision);
  const successorEdges = finished.window_map!.edges.filter(({ to_key }) =>
    to_key.step === 'evaluate' || to_key.step === childId);
  expect(successorEdges).toHaveLength(4);
  for (const variant of ['base', 'compare']) {
    expect(successorEdges).toEqual(expect.arrayContaining([
      {
        from_key: { variant, step: 'train', row: 'default' },
        to_key: { variant, step: 'evaluate', row: 'subject=default' },
      },
      {
        from_key: { variant, step: 'train', row: 'default' },
        to_key: { variant, step: childId, row: 'default' },
      },
    ]));
  }
  expect(finished.window_map?.evaluations?.filter(({ key }) => key.step === 'conditions')).toEqual(settled);
});
