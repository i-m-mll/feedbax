import { expect, test } from './fixtures/studio';
import { type Locator, type Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const views = ['Runs', 'Model', 'Task', 'Objective', 'Training', 'Scene', 'Results', 'Analysis', 'Figures', 'Document'];
const rail = (page: Page) => page.getByRole('navigation', { name: 'Details', exact: true });
const drawer = (page: Page) => page.getByRole('complementary', { name: 'Map Drawer', exact: true });
const menu = (page: Page) => page.getByRole('menu');

/** The colour an element is drawn on: its own background, or the nearest ancestor's that has one. */
function surface(locator: Locator) {
  return locator.evaluate((element) => {
    for (let node: Element | null = element; node; node = node.parentElement) {
      const color = getComputedStyle(node).backgroundColor;
      if (color !== 'rgba(0, 0, 0, 0)') return color;
    }
    return null;
  });
}

async function separator(page: Page) {
  return rail(page).evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    const value = getComputedStyle(element).getPropertyValue('--shell-rail-separator').trim();
    const divider = document.querySelector('.studio-window-panel__divider')!.getBoundingClientRect();
    return {
      line: bounds.top + (value === '100%' ? bounds.height : parseFloat(value)),
      bottom: bounds.bottom,
      divider: divider.height ? divider.top : null,
    };
  });
}

/** Each view glyph's offset below the separator; a view the step's kind lacks is a disabled tab. */
async function glyphOffsets(page: Page) {
  const { line } = await separator(page);
  const offsets = new Map<string, number>();
  for (const name of views) {
    const tab = rail(page).getByRole('tab', { name, exact: true });
    if (await tab.count()) offsets.set(name, (await tab.boundingBox())!.y - line);
  }
  return offsets;
}

/** Every drawn glyph sits in its view's fixed place: one pitch per view, in the views' order, whichever are absent. */
function expectFixedPlaces(offsets: Map<string, number>, pitch: number) {
  const first = offsets.get(views[0])!;
  for (const [name, offset] of offsets) expect(offset).toBeCloseTo(first + views.indexOf(name) * pitch, 0);
}

async function expectMenuBeside(page: Page, opener: { x: number; y: number; width: number; height: number }) {
  const box = (await menu(page).boundingBox())!;
  expect(Math.abs(box.y - (opener.y + opener.height))).toBeLessThan(12);
  expect(Math.abs(box.x - opener.x)).toBeLessThan(12);
  await expect(menu(page).getByRole('menuitem')).toHaveText(['Train', 'Evaluate', 'Analyze', 'Report']);
}

test('an empty project draws only the bar, the map with its drawer, the rail and the bottom bar', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  const add = drawer(page).getByRole('button', { name: 'New Step', exact: true });
  await expect(add).toBeVisible();
  const regions = await page.locator('[data-studio-focus-region]').evaluateAll((elements) => elements
    .filter((element) => !element.closest('[hidden]') && element.getClientRects().length > 0)
    .map((element) => (element as HTMLElement).dataset.studioFocusRegion));
  expect(regions).toEqual(['top', 'map', 'steps', 'rail', 'bottom']);
  const map = page.getByRole('region', { name: 'Pipeline Map', exact: true });
  const canvas = await surface(map.getByRole('region', { name: 'Pipeline Map', exact: true }));
  expect(canvas).toBe(await surface(page.locator('.studio-window-panel')));
  expect(canvas).not.toBe(await surface(add));
  for (const name of ['Show Both', 'Show Details', 'Show Drawer', 'Show Inspector']) {
    await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-disabled', 'true');
  }
  await expect(rail(page).getByRole('tab')).toHaveCount(0);
  const { line, bottom } = await separator(page);
  expect(line).toBeCloseTo(bottom, 0);
  for (const name of ['Fit View', 'Zoom In', 'Zoom Out']) {
    await expect(rail(page).getByRole('button', { name, exact: true })).toHaveAttribute('aria-disabled', 'true');
  }

  await expect(add).toBeEnabled();
  await add.press('Enter');
  await expectMenuBeside(page, (await add.boundingBox())!);
  await expect(menu(page).getByRole('menuitem', { name: 'Train' })).not.toHaveAttribute('aria-disabled', 'true');
  for (const name of ['Evaluate', 'Analyze', 'Report']) {
    await expect(menu(page).getByRole('menuitem', { name })).toHaveAttribute('aria-disabled', 'true');
  }
  await menu(page).getByRole('menuitem', { name: 'Train' }).click();
  await expect(page.getByRole('tabpanel', { name: 'Model', exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('a library tip moves to the next item the pointer rests on', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await drawer(page).getByRole('button', { name: 'New Step', exact: true }).press('Enter');
  await menu(page).getByRole('menuitem', { name: 'Train' }).click();
  const details = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await details.getByRole('button', { name: 'Equinox', exact: true }).click();
  const rows = details.getByRole('listbox', { name: 'Equinox', exact: true }).getByRole('option');
  const tip = page.locator('[role=presentation][data-side]');
  for (const index of [0, 1]) {
    const row = rows.nth(index);
    await row.hover();
    await expect(tip).toContainText(await row.innerText());
    const top = (await row.boundingBox())!.y;
    await expect.poll(async () => (await tip.boundingBox())!.y).toBeCloseTo(top, 0);
  }
});

test('the rail separator follows the divider and every view keeps its place', async ({ page, request, studio }, info) => {
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.managed_report', 'seed', studio.home, 'furniture-kinds',
  ));
  const response = await request.get(`/api/projects/${project}`);
  const steps = Object.entries((await response.json() as StudioProjectSnapshot).project.steps ?? {});
  const stepOf = (kind: string) => steps.find(([, step]) => step.kind === kind)![0];
  await page.goto(`/?project=${encodeURIComponent(project)}`);
  await mapStep(page, stepOf('train')).click();
  await rail(page).getByRole('tab', { name: 'Runs', exact: true }).click();

  const handle = page.getByRole('separator', { name: 'Panel Height', exact: true });
  for (const target of [260, 380, 500]) {
    const box = (await handle.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2, target, { steps: 4 });
    const during = await separator(page);
    expect(during.line).toBeCloseTo(during.divider!, 0);
    await page.mouse.up();
    const after = await separator(page);
    expect(after.line).toBeCloseTo(after.divider!, 0);
    for (const offset of (await glyphOffsets(page)).values()) expect(offset).toBeGreaterThanOrEqual(0);
  }

  const zoomOut = (await rail(page).getByRole('button', { name: 'Zoom Out', exact: true }).boundingBox())!;
  const { line } = await separator(page);
  expect(line - (zoomOut.y + zoomOut.height)).toBeGreaterThanOrEqual(0);
  expect(line - (zoomOut.y + zoomOut.height)).toBeLessThan(12);
  const transform = () => page.locator('[data-map-canvas] .react-flow__viewport').evaluate((element) => getComputedStyle(element).transform);
  const before = await transform();
  await rail(page).getByRole('button', { name: 'Zoom In', exact: true }).click();
  await expect.poll(transform).not.toBe(before);

  const trainOffsets = await glyphOffsets(page);
  const pitch = trainOffsets.get('Model')! - trainOffsets.get('Runs')!;
  expect(pitch).toBeGreaterThan(0);
  expectFixedPlaces(trainOffsets, pitch);
  for (const [kind, absent] of [['evaluate', ['Objective', 'Training', 'Scene', 'Results', 'Analysis', 'Figures', 'Document']],
    ['analyze', ['Model', 'Task', 'Objective', 'Training', 'Scene', 'Results', 'Document']]] as const) {
    // The zoomed canvas can carry a card under the window bar, so the step is chosen from the drawer.
    await page.getByRole('tree', { name: 'Steps', exact: true }).getByRole('treeitem', { name: new RegExp(`^${stepOf(kind)}\\b`) }).click();
    for (const name of views) {
      const tab = rail(page).getByRole('tab', { name, exact: true });
      if ((absent as readonly string[]).includes(name)) {
        await expect(tab).toBeVisible();
        await expect(tab).toHaveAttribute('aria-disabled', 'true');
        await expect(tab).toHaveAttribute('tabindex', '-1');
      } else await expect(tab).not.toHaveAttribute('aria-disabled', 'true');
    }
    if (kind === 'evaluate') {
      await rail(page).getByRole('tab', { name: 'Task', exact: true }).focus();
      await page.keyboard.press('ArrowDown');
      await expect(rail(page).getByRole('tab', { name: 'Runs', exact: true })).toBeFocused();
    }
    const offsets = await glyphOffsets(page);
    expect([...offsets.keys()]).toEqual(views);
    expectFixedPlaces(offsets, pitch);
    for (const [name, offset] of offsets) if (trainOffsets.has(name)) expect(offset).toBeCloseTo(trainOffsets.get(name)!, 0);
  }

  await rail(page).getByRole('button', { name: 'Fit View', exact: true }).click();
  await mapStep(page, stepOf('train')).click();
  const trainCard = page.locator(`[data-map-step="${stepOf('train')}"]`).getByRole('button', { name: /^New Step/ });
  await trainCard.click();
  await expectMenuBeside(page, (await trainCard.boundingBox())!);
  await expect(page.getByRole('dialog')).toHaveCount(0);
});
