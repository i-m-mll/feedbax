import { expect, test } from './fixtures/studio';
import type { Page, Route } from '@playwright/test';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type Box = { x: number; y: number; width: number; height: number };
type FrameWindow = Window & { cardFrames?: Box[]; stopCardFrames?: () => void };

/** Holds the first request to `path` whose body `matches`, until `release` is called. */
async function hold(page: Page, path: string, matches: (body: unknown) => boolean) {
  let release!: () => void;
  const released = new Promise<void>((resolve) => { release = resolve; });
  let arrived!: () => void;
  const sent = new Promise<void>((resolve) => { arrived = resolve; });
  let held = false;
  await page.route((url) => url.pathname === path, async (route: Route) => {
    if (!held && matches(route.request().postDataJSON())) {
      held = true;
      arrived();
      await released;
    }
    await route.fallback();
  });
  return { sent, release };
}

async function openModel(page: Page, projectId: string) {
  await page.goto(`/?project=${projectId}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  return page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
}

/** A card dropped while its move saves stays where, and as it was, dropped: no frame shows it moved, resized or hidden. */
test('a dragged Model card keeps its place and shape from release until its move has saved', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const move = await hold(page, `${projectPath}/model/edit`, (body) =>
    (body as { command?: { operation?: { op?: string } } }).command?.operation?.op === 'move_cards');
  const model = await openModel(page, snapshot.project_id);
  const card = model.locator('.react-flow__node[data-id="@inputs"]');
  await expect(card.getByRole('button', { name: 'Add Port', exact: true })).toBeVisible();

  const header = (await card.locator('header').boundingBox())!;
  await page.mouse.move(header.x + 20, header.y + header.height / 2);
  await page.mouse.down();
  await page.mouse.move(header.x + 80, header.y + header.height / 2 + 50, { steps: 8 });
  await page.mouse.up();
  const dropped = (await card.boundingBox())!;
  await card.evaluate((element) => {
    const state = window as FrameWindow;
    state.cardFrames = [];
    let frame = 0;
    const sample = () => {
      const { x, y, width, height } = element.getBoundingClientRect();
      const shown = element.isConnected && getComputedStyle(element).visibility === 'visible';
      state.cardFrames!.push(shown ? { x, y, width, height } : { x: NaN, y: NaN, width: 0, height: 0 });
      frame = requestAnimationFrame(sample);
    };
    sample();
    state.stopCardFrames = () => cancelAnimationFrame(frame);
  });
  await move.sent;
  const saved = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/model/edit`);
  move.release();
  const response = await saved;
  expect(response.ok(), await response.text()).toBe(true);
  expect((await response.json()).result.status).toBe('applied');
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const frames = await page.evaluate(() => {
    const state = window as FrameWindow;
    state.stopCardFrames!();
    return state.cardFrames!;
  });
  const near = (box: Box) => ({ x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) });
  expect(frames.length).toBeGreaterThan(1);
  for (const box of frames) expect(near(box)).toEqual(near(dropped));
});

/** A card whose data changes while its name is edited stays shown, and the edit keeps the focus. */
test('a Model card renamed while its ports load keeps the name field focused', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const projection = await hold(page, `${projectPath}/model/projection`, () => true);
  const model = await openModel(page, snapshot.project_id);
  const card = model.locator('.react-flow__node[data-id="cell"]');
  await expect(card).toBeVisible();
  await projection.sent;
  await card.locator('header').click();
  await page.keyboard.press('F2');
  const name = card.getByRole('textbox', { name: 'Name', exact: true });
  await expect(name).toBeFocused();
  await page.keyboard.type('Renamed');
  const before = await card.evaluate((element) => element.innerHTML);

  projection.release();
  await expect.poll(() => card.evaluate((element) => element.innerHTML)).not.toBe(before);
  await expect(name).toBeFocused();
  await expect(name).toHaveValue(/Renamed$/);
});
