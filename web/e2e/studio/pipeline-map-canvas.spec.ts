import { expect, test } from '@playwright/test';
import type { StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { captureStudioWindow, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { scanStudio } from './fixtures/studioAccessibility';

test.use({ viewport: { width: 1440, height: 900 } });

test('navigates and resizes the saved pipeline canvas without changing zoom', async ({ page, request }, info) => {
  test.setTimeout(55_000);
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
  await expect(mark(included)).toBeInViewport();
  await expect(mark(included)).not.toHaveCSS('box-shadow', 'none');
  await expect(mark(included)).toHaveAccessibleName(/ · Not run$/);
  await expect(page.getByRole('tooltip')).toHaveText(/ · Not run$/);
  await page.keyboard.press('Enter');
  await expect(mark(included)).toHaveAttribute('aria-pressed', 'true');
  await expect(child.locator('[data-map-focus="downstream"]')).toHaveCount(1);
  await expect(child).toHaveAttribute('data-map-related', '1');
  await expect(edge).toHaveAttribute('data-map-emphasis', 'strong');
  await scanStudio(page, info, 'train-map-sweep');
  await mark(excluded).press('Enter');
  await expect(child.locator('[data-map-focus]')).toHaveCount(0);
  await expect(edge).toHaveAttribute('data-map-emphasis', 'light');
  await mark(excluded).press('Escape');
  await expect(mark(excluded)).toHaveAttribute('aria-pressed', 'true');

  const fitMap = async () => {
    await mapStep(page, 'train').press('Shift+F10');
    await page.getByRole('menuitem', { name: 'Fit View', exact: true }).press('Enter');
  };
  const paneControl = canvas.locator('.react-flow__pane');
  const motion = [];
  for (const reducedMotion of ['no-preference', 'reduce'] as const) {
    await page.emulateMedia({ reducedMotion });
    await fitMap();
    const before = await camera();
    const bounds = (await paneControl.boundingBox())!;
    const point = { x: bounds.width - 80, y: bounds.height - 40 };
    const samples = viewport.evaluate((node) => new Promise<number[]>((resolve) => {
      node.ownerDocument.addEventListener('dblclick', () => {
        const start = performance.now();
        const frames: number[] = [];
        const sample = () => {
          frames.push(new DOMMatrixReadOnly(getComputedStyle(node).transform).a);
          if (performance.now() - start < 350) requestAnimationFrame(sample);
          else resolve(frames);
        };
        sample();
      }, { capture: true, once: true });
    }));
    await paneControl.dblclick({ position: point });
    const frames = await samples;
    const after = await camera();
    expect(after.zoom).toBeCloseTo(before.zoom * 2);
    expect((point.x - after.x) / after.zoom).toBeCloseTo((point.x - before.x) / before.zoom);
    expect((point.y - after.y) / after.zoom).toBeCloseTo((point.y - before.y) / before.zoom);
    const intermediate = frames.filter((zoom) => zoom > before.zoom + 0.001 && zoom < after.zoom - 0.001);
    if (reducedMotion === 'reduce') expect(intermediate).toEqual([]);
    else expect(intermediate.length).toBeGreaterThan(0);
    motion.push({ reducedMotion, before, after, frames });
  }
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await fitMap();
  await info.attach('pipeline-zoom-motion', { body: JSON.stringify(motion), contentType: 'application/json' });

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
  await expect(region.getByRole('button', { name: 'Select train Train', exact: true })).toBeVisible();
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

  await page.setViewportSize({ width: 1440, height: 900 });
  const submitted = await request.post(`${projectPath}/train-runs`, {
    data: { revision: snapshot.revision, run_keys: [included.key], token: crypto.randomUUID() },
  });
  expect(submitted.status(), await submitted.text()).toBe(202);
  expect((await submitted.json() as StudioJobSubmission).items[0].disposition).toBe('admitted');
  await mapStep(page, 'train').press('Enter');
  await expect(mark(included)).toHaveAccessibleName(/ · Done$/, { timeout: 18_000 });
  await expect(mark(excluded)).toHaveAccessibleName(/ · Not run$/);
  const done = mark(included).locator('[data-run-status-mark="succeeded"] svg');
  const idle = mark(excluded).locator('[data-run-status-mark="not_run"] svg');
  await expect(done).toBeVisible();
  await expect(idle).toBeVisible();
  const donePath = done.locator('path');
  const idlePath = idle.locator('path');
  await expect(donePath).toHaveCSS('stroke', 'rgb(255, 255, 255)');
  await expect(idlePath).toHaveCSS('stroke', 'rgb(255, 255, 255)');
  expect(await donePath.getAttribute('d')).not.toBe(await idlePath.getAttribute('d'));
  await expect(child.getByText('2 Not run', { exact: true })).toBeVisible();
  await expect(page.getByRole('log', { name: 'Status' }).getByText(/^Train · train · seed=.+: Done$/)).toHaveCount(1);
  const vision = await page.context().newCDPSession(page);
  try {
    await vision.send('Emulation.setEmulatedVisionDeficiency', { type: 'achromatopsia' });
    await captureStudioWindow(page, info, 'mixed-authored-status-glyphs');
  } finally {
    await vision.send('Emulation.setEmulatedVisionDeficiency', { type: 'none' });
    await vision.detach();
  }
});
