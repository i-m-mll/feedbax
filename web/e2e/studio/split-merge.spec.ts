import { expect, test, jobTimeout } from './fixtures/studio';
import { type Locator, type Page } from '@playwright/test';
import type { StudioProjectDocument, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

/** Opens a target's context menu and chooses a command from it. */
async function fromMenu(target: Locator, command: string) {
  await target.click({ button: 'right' });
  await target.page().getByRole('menuitem', { name: command, exact: true }).click();
}

test('splits an Analyze card into three and merges it back, recomputing nothing', { tag: '@round' }, async ({ page, request, studio }) => {
  const home = studio.home;
  const module = 'web.e2e.studio.fixtures.split_merge';
  const project = JSON.parse(await runProjectFixture(module, 'seed', home, 'split-merge'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const document = page.getByRole('tabpanel', { name: 'Document', exact: true });
  const element = (name: string) => analysis.getByRole('group', { name: 'Select ' + name, exact: true });

  async function snapshot() {
    const response = await request.get(path + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  }
  async function windowMap() {
    const response = await request.get(runPath, { params: { revision: (await snapshot()).revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  /** Every request id, and the status of every run and element of the shown variant, by key. */
  async function settled() {
    const { revision } = await snapshot();
    const map = await windowMap();
    const scope = map.variants[0];
    return {
      requests: JSON.parse(await runProjectFixture(module, 'requests', home, project, revision)),
      statuses: [...new Set([
        ...map.rows.filter(({ key }) => key.variant === scope).map(({ status }) => status),
        ...(map.analysis_elements ?? []).filter(({ key }) => key.variant === scope).map(({ status }) => status),
      ])],
      problems: map.problems,
      map,
    };
  }
  async function saved() {
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
  }
  async function run(step: string) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Done', { exact: true })).toBeVisible({
      timeout: jobTimeout,
    });
  }
  /** The Report's rendered result draws its figure. */
  async function reportShowsFigure() {
    await mapStep(page, 'report').click();
    await tab(page, 'Document').click();
    const result = document.getByRole('button', { name: 'Result', exact: true });
    if (await result.getAttribute('aria-expanded') !== 'true') await result.click();
    const frame = document.locator('iframe[aria-label="Report"]').contentFrame();
    await expect(frame.locator('[data-report-source] svg.main-svg').first()).toBeVisible();
  }
  async function split(card: string, chosen: readonly string[]) {
    const before = Object.keys((await snapshot()).project.steps ?? {});
    await mapStep(page, card).click();
    await tab(page, 'Analysis').click();
    for (const [index, name] of chosen.entries()) {
      await element(name).click(index ? { modifiers: ['Shift'] } : undefined);
    }
    await fromMenu(element(chosen.at(-1)!), 'Split');
    // The Split waits for anything the workspace is still settling, then saves the new card.
    const added = async () => Object.keys((await snapshot()).project.steps ?? {}).filter((id) => !before.includes(id));
    await expect.poll(added).toHaveLength(1);
    await saved();
    const [created] = await added();
    await expect(page.locator(`[data-map-step="${created}"]`)).toBeVisible();
    return created;
  }

  await page.goto('/?project=' + encodeURIComponent(project));
  await run('train');
  await run('baseline');
  await run('report');
  const original = await snapshot();
  const initial = await settled();
  expect(initial.problems).toEqual([]);
  expect(initial.statuses).toEqual(['succeeded']);
  await reportShowsFigure();

  // Split the downstream operation and figure off, then the figure off the new card.
  const middle = await split('analyze', ['Summarize', 'Profile']);
  const last = await split(middle, ['Profile']);
  const three = await settled();
  expect(three.requests).toEqual(initial.requests);
  expect(three.statuses).toEqual(['succeeded']);
  expect(three.problems).toEqual([]);
  for (const [from, to] of [['analyze', middle], [middle, last], [last, 'report']]) {
    expect(three.map.edges).toContainEqual(expect.objectContaining({
      from_key: expect.objectContaining({ step: from }), to_key: expect.objectContaining({ step: to }),
    }));
  }
  const cut = (await snapshot()).project.steps![last].inputs!;
  expect(cut).toEqual({ summary: { step: middle, element: 'summarize', product: 'summary' } });
  // The input bound across cards reads as the source card's operation output.
  await mapStep(page, last).click();
  await tab(page, 'Analysis').click();
  // The source card reads by its step's name, as the Service names a split card (`analyze-2` reads `Analyze 2`).
  const bound = 'Analyze 2 · Summarize · summary';
  await element(bound).click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();
  await expect(inspector.getByRole('combobox', { name: 'Summary', exact: true })).toHaveText(bound);
  await reportShowsFigure();

  // Merge the three cards back into the first.
  await mapStep(page, 'analyze').click();
  await mapStep(page, middle).click({ modifiers: ['Shift'] });
  await mapStep(page, last).click({ modifiers: ['Shift'] });
  await fromMenu(mapStep(page, last), 'Merge');
  await expect(page.locator(`[data-map-step="${middle}"]`)).toHaveCount(0);
  await saved();
  expect((await snapshot()).project).toEqual(original.project);
  const merged = await settled();
  expect(merged.requests).toEqual(initial.requests);
  expect(merged.statuses).toEqual(['succeeded']);
  await reportShowsFigure();

  // A card whose document a second card reads cannot be split.
  const unsplit = Object.keys((await snapshot()).project.steps ?? {}).length;
  await fromMenu(mapStep(page, 'analyze'), 'Duplicate Step');
  await expect.poll(async () => Object.keys((await snapshot()).project.steps ?? {}).length).toBe(unsplit + 1);
  await saved();
  await expect.poll(async () => (await windowMap()).regroup_refused).toContain('analyze');
  await mapStep(page, 'analyze').click();
  await tab(page, 'Analysis').click();
  await element('Profile').click({ button: 'right' });
  await expect(page.getByRole('menuitem', { name: 'Split', exact: true })).toBeDisabled();
});
