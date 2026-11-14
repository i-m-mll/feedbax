import { expect, test, type Page } from '@playwright/test';
import type {
  AnalysisDocument, StudioJobSnapshot, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

test('authors a managed Analyze graph from one saved Evaluate run', async ({ page, request }, info) => {
  test.setTimeout(70_000);
  const home = String(info.config.metadata.home);
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', home, 'managed-analyze'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const reviewPath = path + '/authoring/analyze';
  const mounted = '/?project=' + encodeURIComponent(project);
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const graph = (saved: StudioProjectSnapshot) => saved.project.documents?.analyze.payload as unknown as AnalysisDocument;
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
  await page.getByRole('article', { name: 'Evaluate baseline', exact: true }).locator('.studio-map-run-mark').click();
  await page.keyboard.press('Meta+K');
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  await palette.getByRole('combobox').fill('Analyze');
  await palette.getByRole('option', { name: 'Analyze', exact: true }).press('Enter');
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
  await edit(() => enter(inspector.getByRole('spinbutton', { name: 'Edit Bounds Scale', exact: true }), '1.5'));
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

  await analysis.getByRole('group', { name: 'Select summarize_velocity', exact: true }).click();
  const revised = await edit(() => enter(inspector.getByRole('spinbutton', { name: 'Edit Bounds Scale', exact: true }), '2'));
  const changed = requests(await plan(revised.revision));
  expect(changed.aligned_velocity).toBe(requests(original).aligned_velocity);
  expect(changed.summarize_velocity).not.toBe(requests(original).summarize_velocity);
  expect(changed.velocity_summary).not.toBe(requests(original).velocity_summary);
  await analysis.focus();
  const undone = await edit(() => page.keyboard.press('Meta+Z'));
  expect(requests(await plan(undone.revision))).toEqual(requests(original));
  const redone = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(requests(await plan(redone.revision))).toEqual(changed);

  await tab(page, 'Figures').click();
  await figures.getByRole('button', { name: 'velocity_summary', exact: true }).click();
  await expect(figures.getByText('Not run', { exact: true })).toBeVisible();
  const caption = inspector.getByRole('textbox', { name: 'Caption', exact: true });
  const captioned = await edit(async () => { await caption.fill('Velocity by condition'); await caption.press('Enter'); });
  expect(graph(captioned).views?.velocity_summary.figure).toMatchObject({ caption: 'Velocity by condition' });
  await page.reload();
  await mapStep(page, 'analyze').click();
  await tab(page, 'Figures').click();
  await expect(figures.getByText('Velocity by condition', { exact: true })).toBeVisible();
  const restored = await snapshot();
  expect(graph(restored)).toEqual(graph(captioned));
  expect(restored.revision).toBe(captioned.revision);
  expect(await attempts()).toEqual(admittedJobs);
});
