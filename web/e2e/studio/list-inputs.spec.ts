import { expect, test, jobTimeout } from './fixtures/studio';
import { type Page } from '@playwright/test';
import type { StudioJobSnapshot, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';
import { activate, openView } from './fixtures/studioKeyboard';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

/**
 * Drags a wire from a card's output handle and drops it on another card. The map moves its cards when it draws a newer
 * plan, and every card that accepts the wire shows it from the start of the drag, so the drag starts once the map draws
 * the saved revision's plan and the pointer goes to where the card's head is then.
 */
async function wire(page: Page, source: string, target: string) {
  const card = page.locator(`[data-map-step="${target}"]`);
  await expect(page.getByRole('region', { name: 'Pipeline Map', exact: true }).last()).not.toHaveAttribute('aria-busy', 'true');
  await page.locator(`[data-map-step="${source}"]`).hover();
  await page.locator(`[data-map-handle="${source}"]`).hover();
  await page.mouse.down();
  await card.hover({ position: { x: 40, y: 12 } });
  await expect(card).toHaveAttribute('data-pick', 'allowed');
  await page.mouse.up();
}

test('binds lists of selections to an Evaluate subject and an Analyze dataset role', { tag: '@round' }, async ({ page, request, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.list_inputs', 'seed', studio.home, 'lists'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  const mutation = (method: string, route: string) => page.waitForResponse((response) =>
    response.request().method() === method && new URL(response.url()).pathname === route);

  async function snapshot() {
    const response = await request.get(path);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function plan() {
    const response = await request.get(runPath, { params: { revision: (await snapshot()).revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  async function requests(step: string) {
    return Object.fromEntries((await plan()).evaluations!.filter(({ key }) => key.step === step)
      .map(({ key, request_id }) => [key.row, request_id]));
  }
  async function edit(action: () => Promise<unknown>) {
    const saving = mutation('PATCH', path);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return (await response.json() as StudioProjectSnapshot).project;
  }
  async function run(step: string, count: number) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(runs.getByText('Done', { exact: true }).or(runs.getByText('Failed', { exact: true })))
      .toHaveCount(count, { timeout: 60_000 });
    const failed = (await (await request.get('/api/jobs', { params: { project } })).json() as StudioJobSnapshot).jobs
      .filter(({ status }) => status === 'failed').map(({ kind, reason, log_tail }) => ({ kind, reason, log_tail }));
    expect(failed).toEqual([]);
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded' });
  await run('train', 1);
  await run('train_b', 2);

  // An Evaluate takes a second Train step's runs from the picker, and lists the runs of both.
  await mapStep(page, 'conditions').click();
  const subject = inspector.getByRole('list', { name: 'Subject', exact: true });
  expect(Object.keys(await requests('conditions'))).toHaveLength(2);
  await inspector.getByRole('button', { name: 'Add Subject', exact: true }).click();
  const listed = await edit(() => pick(subject.getByRole('combobox', { name: 'Subject', exact: true }).last(), 'Train b'));
  expect(listed.steps!.conditions.inputs!.subject).toEqual([
    { step: 'train', checkpoint: 'last' }, { step: 'train_b', checkpoint: 'last' },
  ]);
  await expect(subject.getByRole('listitem')).toHaveCount(2);
  const gathered = await requests('conditions');
  expect(Object.keys(gathered)).toHaveLength(6);
  await expect(runs.getByRole('row')).toHaveCount(7);
  await expect(runs.getByRole('columnheader', { name: 'Model', exact: true })).toBeVisible();
  await expect(runs.getByRole('cell', { name: 'Train', exact: true })).toHaveCount(2);
  await expect(runs.getByRole('cell', { name: 'Train b', exact: true })).toHaveCount(4);

  // Removing the second selection leaves the first one's requests as they were.
  const removed = await edit(() => subject.getByRole('button', { name: 'Remove Train b', exact: true }).click());
  expect(removed.steps!.conditions.inputs!.subject).toEqual({ step: 'train', checkpoint: 'last' });
  const kept = Object.values(await requests('conditions'));
  expect(kept).toHaveLength(2);
  expect(Object.values(gathered)).toEqual(expect.arrayContaining(kept));

  // A wire dragged from the Train card adds the same selection.
  const wired = await edit(() => wire(page, 'train_b', 'conditions'));
  expect(wired.steps!.conditions.inputs!.subject).toEqual(listed.steps!.conditions.inputs!.subject);
  expect(await requests('conditions')).toEqual(gathered);
  await edit(() => wire(page, 'train_b', 'conditions_b'));
  expect(Object.keys(await requests('conditions_b'))).toHaveLength(6);

  // An Analyze dataset role lists two Evaluate steps, and its figure splits by source.
  await mapStep(page, 'analyze').click();
  await openView(page, 'analysis');
  await page.getByRole('tabpanel', { name: 'Analysis', exact: true }).getByRole('group', { name: 'Select states', exact: true }).click();
  const states = inspector.getByRole('list', { name: 'States', exact: true });
  await inspector.getByRole('button', { name: 'Add States', exact: true }).click();
  const analyzed = await edit(() => pick(states.getByRole('combobox', { name: 'States', exact: true }).last(), 'Conditions b'));
  expect(analyzed.steps!.analyze.inputs!.states).toEqual([{ step: 'conditions' }, { step: 'conditions_b' }]);
  await expect(states.getByRole('listitem')).toHaveCount(2);
  expect((await plan()).problems.filter(({ step }) => step === 'analyze')).toEqual([]);

  await run('conditions', 6);
  await run('conditions_b', 6);
  await run('analyze', 1);
  await activate(page, mapStep(page, 'analyze'));
  await openView(page, 'figures');
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  await activate(page, figures.getByRole('button', { name: 'Profile', exact: true }));
  const legend = figures.locator('svg.main-svg text.legendtext');
  await expect(legend.filter({ hasText: /^Conditions (?!b )/ }).first()).toBeVisible({ timeout: jobTimeout });
  await expect(legend.filter({ hasText: /^Conditions b / }).first()).toBeVisible();
  // Each trace is a distinct run and condition, so no two legend entries share a name, and a run with no value
  // on a grouped label names none.
  const names = await legend.allTextContents();
  expect(names.length - new Set(names).size, names.join(' | ')).toBe(0);
  expect(names.filter((name) => /\bNone\b|\s{2}|\s$/.test(name))).toEqual([]);
});
