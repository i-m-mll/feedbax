import { expect, test } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

test('navigates and resizes the saved pipeline canvas without changing zoom', async ({ page, request }) => {
  test.setTimeout(42_000);
  const [model, task, objective, train, successor] = await fixture('managed-files');
  const unrelated = { ...structuredClone(train), id: 'unrelated' };
  train.step.base.inline.method_payload.payload.seed = 0;
  train.step.sweep = { axes: { seed: { values: {
    first: { label: '1', patches: { 'method_payload.payload.seed': { value: 1 } } },
    second: { label: '2', patches: { 'method_payload.payload.seed': { value: 2 } } },
    third: { label: '3', patches: { 'method_payload.payload.seed': { value: 3 } } },
  } } } };
  successor.step.inputs.checkpoint.where = { axes: { seed: ['first', 'third'] } };
  const { snapshot, projectPath } = await seedManagedProject(request, [model, task, objective, train, successor, unrelated]);
  const region = page.locator('[data-studio-focus-region="map"]:visible');
  const canvas = page.locator('[data-map-canvas]');
  const edge = canvas.locator('[data-map-edge]');
  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(edge).toHaveCount(1);

  const planResponse = await request.get(`${projectPath}/train-runs`, {
    params: { revision: snapshot.revision, window_map: 'true' },
  });
  expect(planResponse.ok(), await planResponse.text()).toBe(true);
  const { window_map: plan } = await planResponse.json() as StudioSavedTrainRuns;
  const edges = plan?.edges.filter(({ from_key, to_key }) => from_key.step === 'train' && to_key.step === 'successor');
  expect(edges).toHaveLength(2);
  const runs = plan!.rows.filter(({ key }) => key.step === 'train');
  const included = runs.find(({ key }) => edges!.some(({ from_key }) => from_key.row === key.row))!;
  const excluded = runs.find(({ key }) => !edges!.some(({ from_key }) => from_key.row === key.row))!;

  const viewport = canvas.locator('.react-flow__viewport');
  const camera = () => viewport.evaluate((node) => {
    const matrix = new DOMMatrixReadOnly(getComputedStyle(node).transform);
    return { zoom: matrix.a, x: matrix.e, y: matrix.f };
  });
  const card = (name: string) => canvas.locator(`[data-map-step="${name}"]`);
  const source = card('train');
  const child = card('successor');
  const mark = (row: typeof included) => source.getByRole('button', {
    name: new RegExp(`^Select seed ${row.coordinates.seed.label ?? row.coordinates.seed.value_id} · `),
  });

  await page.keyboard.press('Control+Meta+1');
  await expect(region).toBeFocused();
  await page.keyboard.press('Shift+F6');
  await page.keyboard.press('F6');
  await expect(region).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(mapStep(page, 'train')).toBeFocused();
  await page.keyboard.press('ArrowRight');
  await expect(mapStep(page, 'successor')).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('Enter');
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await expect(card('unrelated')).toHaveAttribute('data-map-dimmed', 'true');
  await expect(child).toContainText('2 of 3');

  for (let index = 0; index < 16 && !await mark(included).evaluate((node) => node === document.activeElement); index += 1) {
    await page.keyboard.press('Tab');
  }
  await expect(mark(included)).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(mark(included)).toHaveAttribute('aria-pressed', 'true');
  await expect(child.locator('[data-map-focus="downstream"]')).toHaveCount(1);
  await expect(child).toHaveAttribute('data-map-related', '1');
  await expect(edge).toHaveAttribute('data-map-emphasis', 'strong');
  await mark(excluded).press('Enter');
  await expect(child.locator('[data-map-focus]')).toHaveCount(0);
  await expect(edge).toHaveAttribute('data-map-emphasis', 'light');
  await mark(excluded).press('Escape');
  await expect(mark(excluded)).toHaveAttribute('aria-pressed', 'true');

  const initial = await camera();
  await mapStep(page, 'train').press('Shift+F10');
  const menu = page.getByRole('menu', { name: 'Pipeline Map' });
  await menu.getByRole('menuitem', { name: 'Zoom In', exact: true }).press('Enter');
  await expect.poll(async () => (await camera()).zoom).toBeGreaterThan(initial.zoom);
  const zoomed = await camera();
  const pane = await canvas.locator('.react-flow__pane').boundingBox();
  expect(pane).not.toBeNull();
  await page.mouse.move(pane!.x + pane!.width - 20, pane!.y + pane!.height - 20);
  await page.mouse.down();
  await page.mouse.move(pane!.x + pane!.width - 80, pane!.y + pane!.height - 70, { steps: 4 });
  await page.mouse.up();
  await expect.poll(async () => (await camera()).x).not.toBe(zoomed.x);
  expect((await camera()).zoom).toBe(zoomed.zoom);

  await page.keyboard.press('Control+Meta+2');
  const separator = page.getByRole('separator', { name: 'Panel Height' });
  const originalHeight = await separator.getAttribute('aria-valuenow');
  await separator.press('ArrowDown');
  await expect(separator).not.toHaveAttribute('aria-valuenow', originalHeight!);
  await expect.poll(async () => (await camera()).zoom).toBe(zoomed.zoom);
  await page.keyboard.press('Control+Meta+3');
  await expect(canvas).toBeHidden();
  await expect(region.getByRole('button', { name: 'Select train', exact: true })).toBeVisible();
  await page.keyboard.press('Control+Meta+2');
  await page.setViewportSize({ width: 540, height: 850 });
  await expect(child).toHaveAttribute('data-map-density', 'compact');
  await expect(source).toHaveAttribute('data-map-density', 'full');
  await expect(mark(excluded)).toHaveAttribute('aria-pressed', 'true');
  expect((await camera()).zoom).toBe(zoomed.zoom);

  await page.keyboard.press('Meta+K');
  const palette = page.getByRole('dialog', { name: 'Command Palette' });
  await palette.getByRole('combobox').fill('Fit View');
  await expect(palette.getByRole('option', { name: 'Fit View', exact: true })).toHaveAttribute('data-command-id', 'fitView');
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await camera()).zoom).toBeLessThan(zoomed.zoom);

  const reopened = await request.get(projectPath);
  expect(reopened.ok(), await reopened.text()).toBe(true);
  const persisted = await reopened.json() as StudioProjectSnapshot;
  expect(persisted.revision).toBe(snapshot.revision);
  expect(persisted.presentation_revision).toBe(snapshot.presentation_revision);
});
