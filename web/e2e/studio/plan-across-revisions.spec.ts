import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { reveal } from './fixtures/controls';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

type ControlWindow = Window & { controlChanges?: string[]; stopControlChanges?: () => void };

/** Records every control added to or removed from the window until `stopControlChanges`. */
async function watchControls(page: Page) {
  await page.evaluate(() => {
    const state = window as ControlWindow;
    const controls = 'button, input, select, textarea, a[href], [role="button"], [role="combobox"], [role="tab"], ' +
      '[role="option"], [role="menuitem"], [role="checkbox"], [role="switch"], [contenteditable="true"]';
    const name = (element: Element) =>
      `${element.tagName.toLowerCase()} ${element.getAttribute('aria-label') ?? element.textContent?.trim().slice(0, 40) ?? ''}`;
    state.controlChanges = [];
    const record = (nodes: NodeList, change: string) => {
      for (const node of nodes) {
        if (!(node instanceof Element)) continue;
        for (const control of [...node.matches(controls) ? [node] : [], ...node.querySelectorAll(controls)]) {
          state.controlChanges!.push(`${change} ${name(control)}`);
        }
      }
    };
    const observer = new MutationObserver((mutations) => {
      for (const { addedNodes, removedNodes } of mutations) {
        record(addedNodes, 'added');
        record(removedNodes, 'removed');
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    state.stopControlChanges = () => observer.disconnect();
  });
}

/** The controls added or removed since `watchControls`, between the phases marked with `markControls`. */
async function controlChanges(page: Page) {
  return page.evaluate(() => {
    const state = window as ControlWindow;
    state.stopControlChanges!();
    return state.controlChanges!;
  });
}

const markControls = (page: Page, phase: string) =>
  page.evaluate((name) => (window as ControlWindow).controlChanges!.push(`--${name}`), phase);

const frames = (page: Page) => page.evaluate(() =>
  new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));

/**
 * From an edit until its revision's plan is read, the window draws what the last plan read said: no control of the
 * Model view, the Evaluate inspector or the rail goes away or comes back. A gesture made meanwhile runs once, after.
 */
test('a new revision\'s plan, while read, changes no control and holds a gesture until it is read', { tag: '@slow' }, async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'scoped-series-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const patches: string[] = [];
  page.on('request', (sent) => {
    if (sent.method() === 'PATCH' && new URL(sent.url()).pathname === projectPath) patches.push(sent.url());
  });
  const planRead = (revision: string) => page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === `${projectPath}/train-runs` && url.searchParams.get('revision') === revision &&
      !url.searchParams.has('stopped_checkpoint_job');
  });
  const saved = () => page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  /** Holds the plan read of the next revision after `revision`. */
  const holdPlan = (revision: string) => hold(page, `${projectPath}/train-runs`, (_, sent) => {
    const url = new URL(sent.url());
    return sent.method() === 'GET' && !url.searchParams.has('stopped_checkpoint_job') &&
      url.searchParams.get('revision') !== revision;
  });

  /**
   * Makes `edit` with the next plan read held, and `gesture` while it is held; releases it and returns the controls
   * added or removed from the edit until the gesture's revision is drawn.
   */
  async function heldEdit(revision: string, edit: () => Promise<void>, gesture: () => Promise<void>) {
    const held = await holdPlan(revision);
    const before = patches.length;
    await watchControls(page);
    const edited = saved();
    await markControls(page, 'edit');
    await edit();
    const response = await edited;
    expect(response.ok(), await response.text()).toBe(true);
    const next = (await response.json()).revision as string;
    await held.sent;
    await markControls(page, 'held');
    await gesture();
    await frames(page);
    expect(patches.length, 'the gesture waits for the plan').toBe(before + 1);
    const gestured = saved();
    const read = planRead(next);
    await markControls(page, 'release');
    held.release();
    await read;
    const last = (await (await gestured).json()).revision as string;
    await planRead(last);
    await frames(page);
    expect(patches.length, 'the gesture runs once').toBe(before + 2);
    const changes = await controlChanges(page);
    return { changes: changes.some((change) => !change.startsWith('--')) ? changes : [], revision: last };
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'evaluate').click();
  await page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Runs', exact: true }).click();
  await reveal(page, 'Inspector');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const every = inspector.getByRole('textbox', { name: 'Every', exact: true });
  await expect(every).toHaveValue('2');
  // The window has drawn what its first reads give: the Evaluate's Seed and the clock of its trials.
  await expect(inspector.getByRole('button', { name: 'Seed kind: Value', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible();
  await every.fill('3');
  const evaluate = await heldEdit(snapshot.revision, () => every.press('Enter'), () => page.keyboard.press('Meta+Z'));
  expect(evaluate.changes).toEqual([]);
  await expect(every).toHaveValue('2');

  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.locator('.react-flow__node[data-id="@inputs"]')).toHaveClass(/\bdraggable\b/);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await expect(drawer.getByRole('button', { name: 'Equinox', exact: true })).toBeVisible();
  await expect(model.getByRole('button', { name: 'Port target_on', exact: true })).toBeVisible();
  await expect(inspector.getByRole('button', { name: 'Step kind: Value', exact: true })).toBeVisible();
  await expect(page.locator('footer.studio-window-bottom').getByRole('group', { name: 'Clock', exact: true })
    .getByRole('button', { name: /^Trial / })).toBeVisible();
  const train = await heldEdit(evaluate.revision, () => page.keyboard.press('Meta+Shift+Z'), () => page.keyboard.press('Meta+Z'));
  expect(train.changes).toEqual([]);
  const { project } = await (await request.get(projectPath)).json();
  expect(project.steps.evaluate.inputs.subject.checkpoint).toEqual({ every: 2 });
});
