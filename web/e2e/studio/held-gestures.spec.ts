import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import type { Page } from '@playwright/test';

test.use({ viewport: { width: 1440, height: 900 } });

/** Gestures made while an edit is saving run once it saves, in the order made, each on the state it finds. */
test('gestures made while an edit saves run in order once it saves', async ({ page, request, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'held-gestures',
  ));
  const projectPath = `/api/projects/${projectId}`;
  const steps = async () => {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return Object.keys((await response.json() as StudioProjectSnapshot).project.steps ?? {}).sort();
  };
  const modelTab = page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Model', exact: true });

  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await mapStep(page, 'train').click();
  await expect(modelTab).toHaveAttribute('aria-selected', 'false');

  // Each edit and run the page sends, in order; the first, the duplicate's save, is held until released.
  const writes: string[] = [];
  let release!: () => void;
  const released = new Promise<void>((resolve) => { release = resolve; });
  let saving!: () => void;
  const saveSent = new Promise<void>((resolve) => { saving = resolve; });
  await page.route((url) => url.pathname.startsWith(projectPath), async (route) => {
    const sent = route.request();
    const path = new URL(sent.url()).pathname.slice(projectPath.length);
    // Reads are posted too; only these edit the project or run it.
    if (sent.method() === 'GET') return route.fallback();
    if (path === '/train-runs') {
      writes.push('run ' + (sent.postDataJSON() as { run_keys: { step: string }[] }).run_keys.map(({ step }) => step).join(','));
    } else if (path === '/members/command' || path === '/history/replay') {
      writes.push(path);
    } else return route.fallback();
    if (writes.length === 1) {
      saving();
      await released;
    }
    await route.fallback();
  });

  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Duplicate Step train\b/ }).click();
  await saveSent;
  await page.keyboard.press('Meta+Z');
  await page.keyboard.press('Meta+Shift+Z');
  await page.keyboard.press('Meta+2');
  await page.keyboard.press('Meta+R');
  // While the save is held, nothing made after it has run.
  await expect(modelTab).toHaveAttribute('aria-selected', 'false');
  expect(writes).toEqual(['/members/command']);
  release();

  // The duplicate saves, Undo removes the copy, Redo restores it, the Model view opens, and Run runs the copy it selects.
  await expect.poll(() => writes).toEqual(['/members/command', '/history/replay', '/history/replay', 'run train-2']);
  await expect(modelTab).toHaveAttribute('aria-selected', 'true');
  expect(await steps()).toEqual(['baseline', 'train', 'train-2']);
});

/** A gesture made while the window is still connecting runs once it connects. */
test('a gesture made while the window connects runs once it connects', async ({ page, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'held-connecting',
  ));
  const runsPath = `/api/projects/${projectId}/train-runs`;
  const runs: string[][] = [];
  page.on('request', (sent) => {
    if (sent.method() === 'POST' && new URL(sent.url()).pathname === runsPath) {
      runs.push((sent.postDataJSON() as { run_keys: { step: string }[] }).run_keys.map(({ step }) => step));
    }
  });
  // The event stream does not open until released, so the window stays connecting.
  let release!: () => void;
  const released = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => url.pathname === '/api/events', async (route) => {
    await released;
    await route.fallback();
  });

  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await mapStep(page, 'train').click();
  // The plan has arrived; only the connection is still settling.
  await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Not run', { exact: true }).first()).toBeVisible();
  await page.keyboard.press('Meta+R');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  expect(runs).toEqual([]);
  release();

  await expect.poll(() => runs).toEqual([['train']]);
});

/** Holds the requests `matches` picks until `release` is called. */
async function holdRequests(page: Page, matches: (url: URL, method: string) => boolean) {
  let release!: () => void;
  const released = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => matches(url, ''), async (route) => {
    if (!matches(new URL(route.request().url()), route.request().method())) return route.fallback();
    await released;
    await route.fallback();
  });
  return release;
}

/** Chooses New Step in the palette with Enter, as a keyboard user does, and checks that its menu opens once released. */
async function newStepFromPalette(page: Page, release: () => void) {
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
  await palette.getByRole('combobox').fill('New Step');
  await expect(palette.getByRole('option', { name: 'New Step', exact: true })).toHaveAttribute('aria-selected', 'true');
  await palette.getByRole('combobox').press('Enter');
  await expect(palette).toBeHidden();
  const kinds = page.getByRole('menu').locator('[data-step-insert]');
  await expect(kinds).toHaveCount(0);
  release();
  // It offers the kinds of the selection it finds; made from the keyboard, it starts on the first one offered.
  await expect(page.getByRole('menu').locator('[data-step-insert]:not([data-disabled])').first()).toBeFocused();
}

test('New Step chosen in the palette while the window connects opens once it connects', async ({ page, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'held-palette-connecting',
  ));
  const release = await holdRequests(page, (url) => url.pathname === '/api/events');
  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await mapStep(page, 'train').click();
  await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Not run', { exact: true }).first()).toBeVisible();
  await newStepFromPalette(page, release);
});

test('New Step chosen in the palette while the plan of an edit loads opens once it loads', async ({ page, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'held-palette-plan',
  ));
  const projectPath = `/api/projects/${projectId}`;
  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await mapStep(page, 'train').click();
  await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Not run', { exact: true }).first()).toBeVisible();
  // The plan of the duplicate's revision does not arrive until released.
  const release = await holdRequests(page, (url, method) => url.pathname === `${projectPath}/train-runs` && method !== 'POST');
  const saved = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/members/command`);
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Duplicate Step train\b/ }).click();
  expect((await saved).ok()).toBe(true);
  await newStepFromPalette(page, release);
});

test('entering a Model composite while an edit saves happens at once', async ({ page, request, studio }) => {
  const { projectPath, configuration } = await seedManagedProject(request, []);
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, configuration.project, 'sweep');
  const opened = await request.get(`${projectPath}/open`);
  expect(opened.ok(), await opened.text()).toBe(true);
  await page.goto(`/?project=${encodeURIComponent((await opened.json()).snapshot.project_id)}`);
  await mapStep(page, 'baseline').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.locator('[data-id="wrapper"] header')).toBeVisible();
  // The duplicate's save is held, so the workspace stays unsettled.
  const release = await holdRequests(page, (url) => url.pathname === `${projectPath}/members/command`);
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Duplicate Step baseline\b/ }).click();
  await model.locator('[data-id="wrapper"] header').dblclick();
  await expect(model.locator('[data-id="controller"] header')).toBeVisible();
  release();
});
