import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StoredModelDocument as ModelDocument } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { palette, readSaved, startProject } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

type Box = { x: number; y: number; width: number; height: number };
type FrameWindow = Window & { cardFrames?: Box[]; stopCardFrames?: () => void };

async function openModel(page: Page, projectId: string) {
  await page.goto(`/?project=${projectId}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  return page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
}

/** Waits until the Model projection has drawn @inputs' Task-signal rows, which lengthen that card when they land. */
async function projected(model: Locator) {
  await expect(model.locator('.react-flow__node[data-id="@inputs"] [data-handleid^="output:@task:"]').first()).toBeVisible();
}

/**
 * A card dropped while its move saves stays where, and as it was, dropped: no frame shows it moved, resized or hidden.
 * Undo then names the move.
 */
test('a dragged Model card keeps its place and shape from release until its move has saved', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const move = await hold(page, `${projectPath}/model/edit`, (body) =>
    (body as { command?: { operation?: { op?: string } } }).command?.operation?.op === 'move_cards');
  const model = await openModel(page, snapshot.project_id);
  const card = model.locator('.react-flow__node[data-id="@inputs"]');
  await projected(model);

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
  await page.keyboard.press('Meta+K');
  const palette = page.getByRole('dialog', { name: 'Command Palette' });
  await page.keyboard.type('Undo');
  await expect(palette.locator('[data-command-id="undo"]')).toHaveAccessibleName(/^Undo Move Card\b/);
});

/** Waits for the page to draw `count` more frames. */
function animationFrames(page: Page, count: number) {
  return page.evaluate((remaining) => new Promise<void>((done) => {
    const next = () => (--remaining > 0 ? requestAnimationFrame(next) : done());
    requestAnimationFrame(next);
  }), count);
}

type CardFrames = Window & { addFrames?: Record<string, Box | null>[]; droppedAt?: number; stopAddFrames?: () => void };

/**
 * A card dropped from the Library is drawn at its drop point from the frame after the drop, as one element until its add
 * has saved, and no other card moves, resizes or hides on the way.
 */
test('a Model card added from the Library moves and redraws no other card', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const add = await hold(page, `${projectPath}/model/edit`, (body) =>
    (body as { command?: { operation?: { op?: string } } }).command?.operation?.op === 'add_component');
  const model = await openModel(page, snapshot.project_id);
  await expect(model.locator('.react-flow__node[data-id="cell"]')).toHaveClass(/\bdraggable\b/);
  await projected(model);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('GRU');
  const before = await model.locator('.react-flow__node').evaluateAll((nodes) => nodes.map((node) => (node as HTMLElement).dataset.id!));
  await page.evaluate(() => {
    const state = window as CardFrames;
    state.addFrames = [];
    let frame = 0;
    const sample = () => {
      const cards: Record<string, Box | null> = {};
      for (const node of document.querySelectorAll<HTMLElement>('[data-slot="primary"] .react-flow__node')) {
        const card = node.querySelector('.studio-model-card');
        const { x, y, width, height } = card?.getBoundingClientRect() ?? {};
        cards[node.dataset.id!] = card && getComputedStyle(node).visibility === 'visible' ? { x: x!, y: y!, width: width!, height: height! } : null;
      }
      state.addFrames!.push(cards);
      frame = requestAnimationFrame(sample);
    };
    sample();
    window.addEventListener('pointerup', () => { state.droppedAt = state.addFrames!.length; }, { capture: true, once: true });
    state.stopAddFrames = () => cancelAnimationFrame(frame);
  });
  const pane = model.locator('.react-flow__pane');
  const area = (await pane.boundingBox())!;
  const drop = { x: Math.round(area.width / 2), y: Math.round(area.height - 160) };
  await drawer.getByRole('option', { name: 'GRU', exact: true }).dragTo(pane, { targetPosition: drop });
  await add.sent;
  const markPending = () => page.evaluate((shown) => {
    const nodes = [...document.querySelectorAll<HTMLElement>('[data-slot="primary"] .react-flow__node')];
    const node = nodes.find((item) => !shown.includes(item.dataset.id!));
    if (node) (node as HTMLElement & { droppedCard?: boolean }).droppedCard = true;
    return node?.dataset.id;
  }, before);
  await expect.poll(markPending).toBeTruthy();
  await animationFrames(page, 10);
  const pending = await markPending();
  const saved = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/model/edit`);
  add.release();
  const { result } = await (await saved).json();
  expect(result.status).toBe('applied');
  const id = result.changed[0].element.id as string;
  expect(pending).toBe(id);
  const added = model.locator(`.react-flow__node[data-id="${id}"]`);
  await expect(added).toHaveClass(/\bdraggable\b/);
  expect(await added.evaluate((node) => (node as HTMLElement & { droppedCard?: boolean }).droppedCard)).toBe(true);
  await animationFrames(page, 20);
  const { frames, droppedAt } = await page.evaluate(() => {
    const state = window as CardFrames;
    state.stopAddFrames!();
    return { frames: state.addFrames!, droppedAt: state.droppedAt! };
  });
  const near = (box: Box | null) => box && { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) };
  const dropped = { x: Math.round(area.x + drop.x), y: Math.round(area.y + drop.y) };
  expect(droppedAt).toBeDefined();
  frames.forEach((cards, index) => {
    for (const other of before) expect(near(cards[other]), other).toEqual(near(frames[0][other]));
    const box = cards[id];
    if (index >= droppedAt) expect(box, `frame ${index - droppedAt} after the drop`).toBeTruthy();
    if (box) expect({ x: Math.round(box.x), y: Math.round(box.y) }).toEqual(dropped);
  });
});

/**
 * Each card added from the Library lands clear of every card already drawn, the boundary cards included, and every card
 * is where it was after a reload.
 */
test('Model cards added from the Library overlap no other card and stay put across a reload', async ({ page }) => {
  await startProject(page);
  await palette(page, 'newStep', 'New Step');
  await expect(page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model).toBeFocused();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const cards = model.locator('.react-flow__node.draggable');
  await expect(cards).toHaveCount(2);
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('Linear');
  for (const count of [3, 4]) {
    await drawer.getByRole('listbox', { name: 'Runtime', exact: true }).getByRole('option', { name: 'Linear', exact: true }).dblclick();
    await expect(cards).toHaveCount(count);
  }
  const boxes = await cards.evaluateAll((nodes) => nodes.map((node) => {
    const { x, y, width, height } = node.getBoundingClientRect();
    return { id: (node as HTMLElement).dataset.id!, x, y, width, height };
  }));
  for (const [index, a] of boxes.entries()) {
    for (const b of boxes.slice(index + 1)) {
      const overlaps = a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
      expect(overlaps, `${a.id} and ${b.id}`).toBe(false);
    }
  }
  const flow = () => cards.evaluateAll((nodes) => Object.fromEntries(nodes.map((node) =>
    [(node as HTMLElement).dataset.id!, (node as HTMLElement).style.transform])));
  const shown = await flow();
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(cards).toHaveCount(4);
  expect(await flow()).toEqual(shown);
});

/**
 * A Model card is drawn only once the Service has answered where it is stored, so it appears at its stored place and
 * not first at a default one; a move stores the card under its compact address, and the card is there after a reload.
 */
test('a moved Model card is stored under its compact address and is drawn there after a reload', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const projection = await hold(page, `${projectPath}/model/projection`, () => true);
  const model = await openModel(page, snapshot.project_id);
  const card = model.locator('.react-flow__node[data-id="cell"]');
  await projection.sent;
  await animationFrames(page, 10);
  await expect(card).toHaveCount(0);
  projection.release();
  await projected(model);

  const header = (await card.locator('header').boundingBox())!;
  const saved = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/model/edit` &&
    response.request().postDataJSON()?.command?.operation?.op === 'move_cards');
  await page.mouse.move(header.x + 20, header.y + header.height / 2);
  await page.mouse.down();
  await page.mouse.move(header.x + 100, header.y + header.height / 2 + 70, { steps: 8 });
  await page.mouse.up();
  const { result } = await (await saved).json() as { result: { status: string; snapshot: typeof snapshot } };
  expect(result.status).toBe('applied');
  const positions = result.snapshot.presentation?.documents?.['point-mass']?.positions ?? {};
  expect(Object.keys(positions).length).toBeGreaterThan(0);
  for (const key of Object.keys(positions)) expect(key).not.toMatch(/[{"]/);
  const moved = await card.evaluate((node) => (node as HTMLElement).style.transform);

  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await projected(model);
  expect(await card.evaluate((node) => (node as HTMLElement).style.transform)).toBe(moved);
});

/** A field stays editable while another edit saves, and an edit made in it then lands once that save completes. */
test('a Model field edited while a card move saves stays editable and saves after it', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const move = await hold(page, `${projectPath}/model/edit`, (body) =>
    (body as { command?: { operation?: { op?: string } } }).command?.operation?.op === 'move_cards');
  const model = await openModel(page, snapshot.project_id);
  const card = model.locator('.react-flow__node[data-id="cell"]');
  await card.locator('header').click();
  const hidden = page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('textbox', { name: 'Hidden size', exact: true });
  await expect(hidden).toBeEditable();
  const original = await hidden.inputValue();
  const next = original === '17' ? '19' : '17';

  const header = (await card.locator('header').boundingBox())!;
  await page.mouse.move(header.x + 20, header.y + header.height / 2);
  await page.mouse.down();
  await page.mouse.move(header.x + 60, header.y + header.height / 2 + 40, { steps: 6 });
  await page.mouse.up();
  await move.sent;
  await expect(hidden).toBeEditable();
  await enter(hidden, next);
  const cell = async () => {
    const saved = await readSaved(request, projectPath);
    const document = rootPayload(saved.project.documents?.['point-mass']) as unknown as ModelDocument;
    const component = document.interior.components?.cell;
    return component?.kind === 'leaf' ? component.fields?.hidden_size : undefined;
  };
  expect(String(await cell())).toBe(original);

  move.release();
  await expect.poll(cell).toBe(Number(next));
  await expect(hidden).toHaveValue(next);
  await expect(hidden).toBeEditable();
});
