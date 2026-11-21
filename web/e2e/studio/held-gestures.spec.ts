import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

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
