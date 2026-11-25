import { expect, test } from './fixtures/studio';
import { type Page, type Request } from '@playwright/test';
import type {
  AnalysisDocument, StudioJobSnapshot, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { appendManagedFiles, enter, mapStep, runProjectFixture, mapCard } from './fixtures/canonicalTrain';
import { pick, popupListbox } from './fixtures/controls';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

// Catches a placed figure that reads its key ("speed-by-trial") on its card while the Library names it "Speed by trial".
test('places the installed Speed by trial figure from the Analyze Library under the Library name', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'speed-library',
  ));
  const path = '/api/projects/' + project;
  const opened = await request.get(path + '/open');
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  await appendManagedFiles(request, snapshot, [
    { kind: 'document', id: 'analysis', document: {
      schema: 'feedbax.analysis_document.v1', payload: { inputs: { states: { kind: 'dataset' } } },
    } },
    { kind: 'step', id: 'analyze', step: {
      kind: 'analyze', base: { document: 'analysis' }, inputs: { states: { step: 'baseline' } },
    } },
  ]);

  await page.goto('/?project=' + encodeURIComponent(project));
  await mapStep(page, 'analyze').click();
  await tab(page, 'Analysis').click();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library' }).fill('Speed by trial');
  const stored = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === path);
  await drawer.getByRole('option', { name: 'Speed by trial', exact: true }).dblclick();
  const saved = await stored;
  expect(saved.ok(), await saved.text()).toBe(true);
  const after = await saved.json() as StudioProjectSnapshot;
  expect(rootPayload(after.project.documents?.analysis)).toMatchObject({
    views: { 'speed-by-trial': { label: 'Speed by trial', figure: { template: 'feedbax.SpeedByTrial' } } },
  });
  await expect(page.getByRole('tabpanel', { name: 'Analysis', exact: true })
    .getByRole('group', { name: 'Select Speed by trial', exact: true })).toBeVisible();
});

test('authors a managed Analyze graph from one saved Evaluate run', { tag: '@round' }, async ({ page, request, studio }, info) => {
  test.setTimeout(90_000);
  const home = studio.home;
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', home, 'managed-analyze'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const reviewPath = path + '/authoring/analyze';
  const mounted = '/?project=' + encodeURIComponent(project);
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const graph = (saved: StudioProjectSnapshot) => rootPayload(saved.project.documents?.analysis) as unknown as AnalysisDocument;
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
  async function runFixtureStep(step: string) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Done', { exact: true })).toBeVisible({
      timeout: 18_000,
    });
  }

  await page.goto(mounted);
  await runFixtureStep('train');
  await runFixtureStep('baseline');
  const initial = await snapshot();
  const upstream = (await plan(initial.revision)).evaluations!.find(({ key }) => key.step === 'baseline')!;
  expect(upstream.result).toBeTruthy();
  const admittedJobs = await attempts();
  await mapCard(page, 'baseline').locator('.studio-map-run-mark').click();
  await page.keyboard.press('Meta+K');
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  await palette.getByRole('combobox').fill('New Step');
  await palette.getByRole('option', { name: 'New Step', exact: true }).press('Enter');
  await chooseStepKind(page, 'Analyze', true);
  await expect(analysis).toBeVisible();
  const created = await snapshot();
  expect(created.project.steps?.analyze).toMatchObject({
    kind: 'analyze', base: { document: 'analysis' },
  });
  expect(created.project.steps?.analyze.inputs?.states).toEqual({ step: upstream.key.step });
  expect((await plan(created.revision)).edges).toContainEqual(expect.objectContaining({
    from_key: upstream.key, to_key: expect.objectContaining({ step: 'analyze' }),
  }));

  const search = drawer.getByRole('textbox', { name: 'Search Library', exact: true });
  const item = (name: string) => drawer.getByRole('option', { name, exact: true });
  await expect(drawer.getByRole('button', { name: /^Add/ })).toHaveCount(0);
  await search.fill('velocity');
  await item('Aligned velocity').click();
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();
  await expect(inspector.getByText('Aligned velocity', { exact: true })).toBeVisible();
  // An edit's own review answers with the catalog after it, so an edit asks the service once.
  const reviews: unknown[] = [];
  const countReview = (sent: Request) => {
    if (new URL(sent.url()).pathname === reviewPath) reviews.push(sent.postDataJSON()?.change?.kind ?? 'read');
  };
  page.on('request', countReview);
  await edit(() => page.keyboard.press('Enter'));
  await expect(search).toHaveValue('velocity');
  await expect(item('Aligned velocity')).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(item('Peak velocity summary')).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(item('Trial summary')).toBeFocused();
  await edit(() => page.keyboard.press('Enter'));
  await expect(item('Trial summary')).toBeFocused();
  await analysis.getByRole('group', { name: 'Select Aligned velocity', exact: true }).click();
  // A new operation's unset choice reads empty, not as a mixed selection.
  await expect(inspector.getByRole('combobox', { name: 'Position', exact: true })).toHaveText('');
  page.off('request', countReview);
  expect(reviews).toEqual(['edit', 'edit']);
  for (const [field, label] of [
    ['Position', / · effector position$/], ['Velocity', / · effector velocity$/], ['Target', /^target.pos$/i],
  ] as const) {
    const select = inspector.getByRole('combobox', { name: field, exact: true });
    await edit(() => pick(select, label));
  }
  const role = (name: string) => inspector.getByRole('combobox', { name, exact: true });
  await edit(() => pick(role('States'), 'states'));

  await analysis.getByRole('group', { name: 'Select Trial summary', exact: true }).click();
  await edit(() => enter(inspector.getByRole('textbox', { name: 'Bounds Scale', exact: true }), '1.5'));
  await edit(() => enter(inspector.getByRole('textbox', { name: 'Ddof', exact: true }), '0'));
  // A role offers only the sources that fit it: the saved states are no velocity a summary reads.
  await role('Velocity').click();
  await expect(popupListbox(page).getByRole('option', { name: 'Aligned velocity · velocity', exact: true })).toBeVisible();
  await expect(popupListbox(page).getByRole('option', { name: 'states', exact: true })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await edit(() => pick(role('Velocity'), 'Aligned velocity · velocity'));

  const pane = analysis.locator('.react-flow__pane');
  await analysis.getByRole('button', { name: 'Zoom Out', exact: true }).click();
  const [paneBox, summaryBox] = await Promise.all([
    pane.boundingBox(), analysis.getByRole('group', { name: 'Select Trial summary', exact: true }).boundingBox(),
  ]);
  await edit(() => item('Velocity Summary').dragTo(pane, { targetPosition: {
    x: summaryBox!.x - paneBox!.x + summaryBox!.width + 100, y: summaryBox!.y - paneBox!.y,
  } }));
  await analysis.getByRole('group', { name: 'Select Velocity Summary', exact: true }).click();
  const wired = await edit(() => pick(role('Summary'), 'Trial summary · summary'));
  await expect(analysis.locator('.react-flow__edge')).toHaveCount(3);
  const original = await plan(wired.revision);
  const requests = (projection: typeof original) => Object.fromEntries(
    projection.analysis_elements!.filter(({ key }) => key.step === 'analyze')
      .map(({ name, requests: [{ request_id }] }) => [name, request_id]),
  );
  expect(Object.keys(requests(original)).sort()).toEqual(['aligned-velocity', 'trial-summary', 'velocity-summary']);
  expect(original.problems.filter(({ step }) => step === 'analyze')).toEqual([]);
  expect(original.edges).toContainEqual(expect.objectContaining({ from_key: upstream.key, to_key: expect.objectContaining({ step: 'analyze' }) }));

  await analysis.getByRole('group', { name: 'Select Trial summary', exact: true }).click();
  const revised = await edit(() => enter(inspector.getByRole('textbox', { name: 'Bounds Scale', exact: true }), '2'));
  const changed = requests(await plan(revised.revision));
  expect(changed['aligned-velocity']).toBe(requests(original)['aligned-velocity']);
  expect(changed['trial-summary']).not.toBe(requests(original)['trial-summary']);
  expect(changed['velocity-summary']).not.toBe(requests(original)['velocity-summary']);
  await analysis.focus();
  const undone = await edit(() => page.keyboard.press('Meta+Z'));
  expect(requests(await plan(undone.revision))).toEqual(requests(original));
  const redone = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(requests(await plan(redone.revision))).toEqual(changed);

  await tab(page, 'Figures').click();
  await figures.getByRole('button', { name: 'Velocity Summary', exact: true }).click();
  await expect(figures.getByText('Waiting', { exact: true })).toBeVisible();
  await expect(inspector.getByRole('status')).toHaveText('Waiting');
  const caption = inspector.getByRole('textbox', { name: 'Caption', exact: true });
  const captioned = await edit(async () => { await caption.fill('Velocity by condition'); await caption.press('Enter'); });
  expect(graph(captioned).views?.['velocity-summary'].figure).toMatchObject({ caption: 'Velocity by condition' });

  // Catches marks that are added in the inspector but not saved, drawn, or redrawn after a reopen.
  await tab(page, 'Analysis').click();
  await search.fill('peak');
  await item('Peak velocity summary').click();
  await edit(() => page.keyboard.press('Enter'));
  await analysis.getByRole('group', { name: 'Select Peak velocity summary', exact: true }).click();
  await edit(() => enter(inspector.getByRole('textbox', { name: 'Ddof', exact: true }), '0'));
  await edit(() => pick(role('Velocity'), 'Aligned velocity · velocity'));
  await tab(page, 'Figures').click();
  await figures.getByRole('button', { name: 'Velocity Summary', exact: true }).click();
  const addMark = async (name: string) => {
    await inspector.getByRole('button', { name: 'Add Mark', exact: true }).click();
    await edit(() => page.getByRole('menuitem', { name, exact: true }).click());
  };
  await addMark('Data');
  await edit(() => pick(role('Data 2'), 'Peak velocity summary · summary'));
  await addMark('Points');
  await edit(() => pick(role('Data'), 'data-2'));
  await edit(() => pick(role('X'), 'peak_time'));
  // A new points layer reads what the figure's band reads; the peak summary names its fields the same.
  for (const [field, value] of [['Y', 'mean'], ['Lower', 'lower'], ['Upper', 'upper']] as const) {
    await expect(role(field)).toHaveText(value);
  }
  await addMark('Rule');
  await edit(() => enter(inspector.getByRole('textbox', { name: 'At', exact: true }), '0.05'));
  await addMark('Span');
  await edit(() => enter(inspector.getByRole('textbox', { name: 'Start', exact: true }), '0.1'));
  const marked = await edit(() => enter(inspector.getByRole('textbox', { name: 'End', exact: true }), '0.3'));
  expect(graph(marked).views?.['velocity-summary'].figure).toMatchObject({
    data: ['summary', 'data-2'],
    layers: [
      { mark: 'band' },
      { name: 'points', mark: 'points', role: 'data-2', encode: { x: 'peak_time', y: 'mean', error_y: { lower: 'lower', upper: 'upper' } } },
      { name: 'rule', mark: 'rule', encode: { axis: 'y', at: 0.05 } },
      { name: 'span', mark: 'span', encode: { axis: 'x', start: 0.1, end: 0.3 } },
    ],
  });
  expect((await plan(marked.revision)).problems.filter(({ step }) => step === 'analyze')).toEqual([]);

  await page.reload();
  await mapStep(page, 'analyze').click();
  await tab(page, 'Figures').click();
  await expect(figures.getByText('Velocity by condition', { exact: true })).toBeVisible();
  const restored = await snapshot();
  expect(graph(restored)).toEqual(graph(marked));
  expect(restored.revision).toBe(marked.revision);
  await figures.getByRole('button', { name: 'Velocity Summary', exact: true }).click();
  await expect(inspector.getByRole('tree', { name: 'Layers', exact: true }).getByRole('treeitem')).toHaveCount(4);
  expect(await attempts()).toEqual(admittedJobs);
});
