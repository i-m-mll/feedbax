import { expect, test, type Page } from '@playwright/test';
import type {
  ReportDocument, StudioJobSnapshot, StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot,
  StudioProjectWindowMap, StudioSavedLightResult, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

test('runs and reopens saved figures and Report from a selected Evaluate row', async ({ page, request }, info) => {
  test.setTimeout(130_000);
  const home = String(info.config.metadata.home);
  const fixture = 'web.e2e.studio.fixtures.managed_light';
  const project = JSON.parse(await runProjectFixture(fixture, 'seed', home, 'managed-light'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const reviewPath = path + '/authoring/analyze';
  const mounted = '/?project=' + encodeURIComponent(project);
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const document = page.getByRole('tabpanel', { name: 'Document', exact: true });
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const boundsScale = inspector.getByRole('spinbutton', { name: 'Edit Bounds Scale', exact: true });
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
  async function chooseCommand(name: string) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await palette.getByRole('combobox').fill(name);
    await palette.getByRole('option', { name, exact: true }).click();
  }
  async function runSaved(reused?: readonly string[]) {
    const admitted = mutation('POST', runPath);
    await chooseCommand('Run');
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
  async function runFixtureStep(step: string, doneCount = 1) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admitted;
    expect(response.status(), await response.text()).toBe(202);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.map(({ disposition, reason }) => [disposition, reason ?? null])).toEqual(
      Array.from({ length: doneCount }, () => ['admitted', null]),
    );
    const watched = new Set(items.map(({ job_id }) => job_id));
    await expect.poll(async () => {
      const jobs = await request.get('/api/jobs', { params: { project } });
      const history = await jobs.json() as StudioJobSnapshot;
      return history.jobs.filter(({ job_id }) => watched.has(job_id)).map(({ status }) => status);
    }, { timeout: 22_000 }).toEqual(Array(doneCount).fill('succeeded'));
    await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Done', { exact: true })).toHaveCount(doneCount);
  }

  await page.goto(mounted);
  await runFixtureStep('train');
  await runFixtureStep('conditions', 2);
  const initial = await snapshot();
  const settledEvaluate = (await plan(initial.revision)).evaluations!.filter(({ key }) => key.step === 'conditions');
  expect(settledEvaluate.map(({ key }) => key.row).sort()).toEqual(['mass=heavy', 'mass=nominal']);
  expect(new Set(settledEvaluate.map(({ request_id }) => request_id)).size).toBe(2);
  expect(new Set(settledEvaluate.map(({ result }) => result?.job_id)).size).toBe(2);
  expect(settledEvaluate.map(({ status }) => status)).toEqual(['succeeded', 'succeeded']);
  const [firstCondition, secondCondition] = settledEvaluate;
  expect(firstCondition.trials.length).toBeGreaterThan(0);
  expect(firstCondition.trials).toEqual(secondCondition.trials);
  const producer = (await request.get(runPath, { params: { revision: initial.revision } }));
  expect(producer.ok(), await producer.text()).toBe(true);
  const trainId = (await producer.json() as StudioSavedTrainRuns).rows.find(({ key }) => key.step === 'train')!.result!.job_id;
  for (const condition of settledEvaluate) {
    expect(condition.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(condition.result?.checkpoint_source?.job_id).toBe(trainId);
  }
  const upstream = settledEvaluate.find(({ key }) => key.row === 'mass=heavy')!;
  const admittedJobs = await attempts();
  expect(admittedJobs).toHaveLength(3);
  await page.getByRole('table', { name: 'Runs', exact: true })
    .getByRole('button', { name: upstream.key.row, exact: true }).click();
  await chooseCommand('Analyze');
  await expect(analysis).toBeVisible();
  const created = await snapshot();
  expect(created.project.steps?.analyze).toMatchObject({
    kind: 'analyze', base: { document: 'analyze' }, inputs: { states: { step: upstream.key.step, row: upstream.key.row } },
  });
  expect((await plan(created.revision)).edges).toContainEqual(expect.objectContaining({
    from_key: upstream.key, to_key: expect.objectContaining({ step: 'analyze' }),
  }));
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();

  await edit(() => drawer.getByRole('button', { name: 'Add Aligned velocity', exact: true }).click());
  await analysis.getByRole('group', { name: 'Select aligned_velocity', exact: true }).click();
  for (const [field, label] of [
    ['Position', /effector~1position$/], ['Velocity', /effector~1velocity$/], ['Target', /target_pos$/],
  ] as const) {
    const select = inspector.getByRole('combobox', { name: 'Edit ' + field, exact: true });
    const option = select.getByRole('option', { name: label });
    await expect(option).toHaveCount(1);
    await edit(async () => select.selectOption({ label: await option.innerText() }));
  }
  const dialog = page.getByRole('dialog', { name: 'Connect Ports', exact: true });
  async function chooseSource(label: string) {
    await drawer.getByRole('button', { name: 'Connect Ports', exact: true }).click();
    await dialog.getByRole('combobox', { name: 'Source Port' }).selectOption({ label });
  }
  const connect = () => dialog.getByRole('button', { name: 'Connect Ports', exact: true }).click();
  await chooseSource('states');
  await edit(connect);

  await edit(() => drawer.getByRole('button', { name: 'Add Trial summary', exact: true }).click());
  await analysis.getByRole('group', { name: 'Select summarize_velocity', exact: true }).click();
  await edit(() => enter(boundsScale, '1.5'));
  const beforeRefusal = await edit(() => enter(inspector.getByRole('spinbutton', { name: 'Edit Ddof', exact: true }), '0'));
  await chooseSource('states');
  const refusal = page.waitForResponse((response) => new URL(response.url()).pathname === reviewPath &&
    Boolean(response.request().postDataJSON()?.wire));
  await connect();
  expect((await refusal).status()).toBe(422);
  await expect(dialog.getByRole('alert')).toContainText('Requires dataset schemas');
  expect((await snapshot()).revision).toBe(beforeRefusal.revision);
  await dialog.getByRole('combobox', { name: 'Source Port' }).selectOption({ label: 'aligned_velocity.velocity' });
  await edit(connect);

  await edit(() => drawer.getByRole('button', { name: 'Add Figure', exact: true }).click());
  await analysis.getByRole('group', { name: 'Select velocity_summary', exact: true }).click();
  await chooseSource('summarize_velocity.summary');
  const wired = await edit(connect);
  await expect(analysis.locator('.react-flow__edge')).toHaveCount(3);
  const original = await plan(wired.revision);
  const requests = (projection: typeof original) => Object.fromEntries(
    projection.analysis_elements!.filter(({ key }) => key.step === 'analyze').map(({ name, request_id }) => [name, request_id]),
  );
  expect(Object.keys(requests(original)).sort()).toEqual(['aligned_velocity', 'summarize_velocity', 'velocity_summary']);
  expect(original.problems.filter(({ step }) => step === 'analyze')).toEqual([]);
  expect(original.edges).toContainEqual(expect.objectContaining({ from_key: upstream.key, to_key: expect.objectContaining({ step: 'analyze' }) }));
  await tab(page, 'Figures').click();
  await figures.getByRole('button', { name: 'velocity_summary', exact: true }).click();
  await chooseCommand('Report');
  await expect(document.getByRole('textbox', { name: 'Report Title', exact: true })).toBeVisible();
  await chooseCommand('Add Text');
  const addText = page.getByRole('dialog', { name: 'Add Text', exact: true });
  await addText.getByRole('textbox', { name: 'Text', exact: true }).fill('**Velocity** by condition.');
  await edit(() => addText.getByRole('button', { name: 'Add Text', exact: true }).click());
  const ordered = await edit(() => document.getByRole('listitem').nth(1)
    .getByRole('button', { name: 'Move Earlier 2', exact: true }).click());
  const report = ordered.project.documents!.report.payload as unknown as ReportDocument;
  const inputs = ordered.project.steps!.report.inputs!;
  expect(report.blocks[0]).toMatchObject({ markdown: '**Velocity** by condition.' });
  expect(report.blocks).toHaveLength(2);
  const role = Object.keys(inputs)[0];
  expect(report.blocks[1]).toMatchObject({ figure: role });
  expect(inputs[role]).toMatchObject({ step: 'analyze', row: 'default', element: 'velocity_summary', product: 'figure' });

  const settled = await plan(ordered.revision);
  expect(settled.problems.filter(({ step }) => step === 'analyze' || step === 'report')).toEqual([]);
  expect(requests(settled)).toEqual(requests(original));
  const reportRequest = settled.requests.find(({ key }) => key.step === 'report');
  expect(reportRequest?.request_id).toMatch(/^report:/);
  expect(settled.edges).toContainEqual(expect.objectContaining({
    from_key: expect.objectContaining({ step: 'analyze' }), to_key: expect.objectContaining({ step: 'report' }),
  }));
  expect(await attempts()).toEqual(admittedJobs);

  function light(projection: StudioProjectWindowMap) {
    const elements = projection.analysis_elements!;
    return {
      aligned: elements.find(({ name }) => name === 'aligned_velocity')!,
      summary: elements.find(({ name }) => name === 'summarize_velocity')!,
      figure: elements.find(({ name }) => name === 'velocity_summary')!,
      report: projection.rows.find(({ key }) => key.step === 'report')!,
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
      key: entry.key, status: entry.status, reason: entry.reason,
      current: identity(entry.result), previous: identity(entry.previous),
    }));
  }
  async function matchesCli(projection: StudioProjectWindowMap) {
    const cli = JSON.parse(await runProjectFixture(fixture, 'status', home, project)) as StudioSavedTrainRuns;
    expect(lightStatus(cli.window_map!)).toEqual(lightStatus(projection));
  }
  async function waitForLight(revision: string) {
    await expect.poll(async () => lightStatus(await plan(revision)).map(({ status }) => status), { timeout: 22_000 })
      .toEqual(['succeeded', 'succeeded', 'succeeded', 'succeeded']);
    return await plan(revision);
  }
  async function showReport(url: string) {
    await expect(document.getByRole('textbox', { name: 'Text', exact: true })).toHaveValue('**Velocity** by condition.');
    await document.getByText('Result', { exact: true }).click();
    const frame = document.locator('iframe[title="Report"]');
    await expect(frame).toHaveAttribute('src', url);
    const rendered = frame.contentFrame();
    await expect(rendered.getByText('Velocity', { exact: true })).toBeVisible();
    await expect(rendered.locator('[data-report-source] svg.main-svg').first()).toBeVisible();
  }
  async function showFigure() {
    await mapStep(page, 'analyze').click();
    await tab(page, 'Figures').click();
    await figures.getByRole('button', { name: 'velocity_summary', exact: true }).click();
  }
  async function drawnFigure(name: 'Current Result' | 'Previous Result') {
    await expect(figures.getByRole('region', { name, exact: true }).locator('svg.main-svg').first()).toBeVisible();
  }

  const createdJobs = await runSaved();
  expect(createdJobs).toHaveLength(4);
  const completed = await waitForLight(ordered.revision);
  const [aligned, summary, figure, renderedReport] = result(completed);
  expect(result(completed).map(({ job_id }) => job_id).sort()).toEqual(createdJobs.sort());
  for (const { receipt } of result(completed)) expect(receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(figure.request_id).toBe(requests(completed).velocity_summary);
  expect(renderedReport.request_id).toBe(reportRequest?.request_id);
  const firstFigure = await request.get(figure.products.figure);
  expect(firstFigure.ok(), await firstFigure.text()).toBe(true);
  const figureBody = await firstFigure.body();
  await showReport(renderedReport.products.report);
  await showFigure();
  await drawnFigure('Current Result');
  await matchesCli(completed);

  await page.reload();
  await mapStep(page, 'report').click();
  await tab(page, 'Document').click();
  await showReport(renderedReport.products.report);
  await showFigure();
  await drawnFigure('Current Result');
  expect((await snapshot()).revision).toBe(ordered.revision);

  const beforeEdit = [...admittedJobs, ...createdJobs].sort();
  await mapStep(page, 'train').click();
  await mapStep(page, 'train').click({ modifiers: ['Meta'] });
  expect(await runSaved(beforeEdit)).toEqual([]);
  expect(await attempts()).toEqual(beforeEdit);

  await mapStep(page, 'analyze').click();
  await tab(page, 'Analysis').click();
  await analysis.getByRole('group', { name: 'Select summarize_velocity', exact: true }).click();
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();
  const changed = await edit(() => enter(boundsScale, '2'));
  const stale = await plan(changed.revision);
  const [stillCurrent, staleSummary, staleFigure, staleReport] = lightStatus(stale);
  expect(stillCurrent.current).toEqual(identity(aligned));
  expect(staleSummary.reason).toContain('bounds_scale');
  expect([staleSummary, staleFigure, staleReport].map(({ status, previous }) => [status, previous]))
    .toEqual([summary, figure, renderedReport].map((entry) => ['stale', identity(entry)]));
  await matchesCli(stale);
  await tab(page, 'Figures').click();
  await figures.getByRole('button', { name: 'velocity_summary', exact: true }).click();
  await drawnFigure('Previous Result');
  expect(await attempts()).toEqual(beforeEdit);

  await tab(page, 'Analysis').click();
  const undo = await edit(() => page.keyboard.press('Meta+Z'));
  expect(lightStatus(await plan(undo.revision))).toEqual(lightStatus(completed));
  expect(await attempts()).toEqual(beforeEdit);

  await analysis.getByRole('group', { name: 'Select summarize_velocity', exact: true }).click();
  const reapplied = await edit(() => enter(boundsScale, '2'));
  await mapStep(page, 'report').click();
  await tab(page, 'Document').click();
  const rerun = await runSaved();
  expect(rerun).toHaveLength(3);
  const replaced = await waitForLight(reapplied.revision);
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
  await matchesCli(replaced);
  await showReport(newReport.products.report);
  await showFigure();
  await drawnFigure('Current Result');
  await figures.getByRole('button', { name: 'Previous Result', exact: true }).click();
  await drawnFigure('Previous Result');
});

test('authors a paired Analyze and shared successors from the primary variant', async ({ page, request }, info) => {
  test.setTimeout(65_000);
  const home = String(info.config.metadata.home);
  const project = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.managed_light', 'seed-variants', home, 'managed-light-variants',
  ));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const variants = page.getByRole('combobox', { name: 'Variant', exact: true });
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const train = page.getByRole('article', { name: 'Train train', exact: true });
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
    await expect(table.getByText('Done', { exact: true })).toHaveCount(count, { timeout: 22_000 });
  }
  async function selectPair() {
    await openRuns('conditions');
    const nominal = table.getByRole('button', { name: 'mass=nominal', exact: true });
    const heavy = table.getByRole('button', { name: 'mass=heavy', exact: true });
    await nominal.click();
    await heavy.click({ modifiers: ['Shift'] });
    await expect(nominal).toHaveAttribute('aria-pressed', 'true');
    await expect(heavy).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('Meta+K');
    await palette.getByRole('combobox').fill('Analyze');
  }
  async function successor(kind: 'Train' | 'Evaluate', id: string) {
    await openRuns('train');
    const creation = page.waitForResponse((response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === path + '/files'
      && response.request().postDataJSON().file.id === id);
    await train.getByRole('button', { name: 'New Step train', exact: true }).click();
    await page.getByRole('dialog', { name: 'New Step', exact: true })
      .getByRole('button', { name: kind, exact: true }).click();
    const created = await creation;
    expect(created.status(), await created.text()).toBe(201);
    await expect(mapStep(page, id)).toHaveAttribute('aria-pressed', 'true');
    return (await created.json() as StudioProjectDocument).snapshot;
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded', timeout: 10_000 });
  await expect(variants).toHaveValue('base');
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

  await variants.selectOption('compare');
  await selectPair();
  await expect(palette.getByRole('option', { name: 'Analyze', exact: true })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await openRuns('train');
  await expect(train.getByRole('button', { name: 'New Step train', exact: true })).toHaveCount(0);

  await variants.selectOption('base');
  await selectPair();
  const creating = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === path + '/files'
    && response.request().postDataJSON().file.kind === 'step');
  await palette.getByRole('option', { name: 'Analyze', exact: true }).press('Enter');
  const created = await creating;
  expect(created.status(), await created.text()).toBe(201);
  const { snapshot: paired } = await created.json() as StudioProjectDocument;
  await expect(page.getByRole('tabpanel', { name: 'Analysis', exact: true })).toBeVisible();
  expect(paired.project.steps?.analyze).toMatchObject({
    kind: 'analyze', inputs: { states: { step: 'conditions', where: { axes: {
      mass: base.map(({ key }) => key.row.split('=')[1]),
    } } } },
  });
  expect(paired.project.sweep).toEqual(initial.project.sweep);
  await expect(drawer.getByRole('button', { name: 'Add Aligned velocity', exact: true })).toBeEnabled();
  await drawer.getByRole('button', { name: 'Add Aligned velocity', exact: true }).click();
  await expect.poll(async () => (await snapshot()).project.documents?.analyze?.payload)
    .toMatchObject({ operations: { aligned_velocity: { operation: 'feedbax.aligned_velocity' } } });

  const evaluated = await successor('Evaluate', 'evaluate');
  await expect(drawer.getByRole('combobox', { name: 'Trial Set', exact: true })).toBeFocused();
  expect(evaluated.project.steps?.evaluate).toMatchObject({
    kind: 'evaluate', inputs: { subject: { step: 'train', checkpoint: 'last' } },
  });
  const childId = 'train-continuation';
  const continuation = await successor('Train', childId);
  const batches = drawer.getByRole('textbox', { name: 'N Batches', exact: true });
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
  const saved = await saving;
  expect(saved.ok(), await saved.text()).toBe(true);
  const edited = await saved.json() as StudioProjectSnapshot;
  const child = edited.project.steps?.[childId];
  expect(child).toMatchObject({
    base: { step: 'train' }, inputs: { checkpoint: { step: 'train', checkpoint: 'last' } },
  });
  expect(child?.deltas?.at(-1)?.patches).toEqual([
    { op: 'replace', path: 'training_config.n_batches', value: 4 },
    { op: 'replace', path: 'method_payload.payload.target_total', value: 4 },
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
