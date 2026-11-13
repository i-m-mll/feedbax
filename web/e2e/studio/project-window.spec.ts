import { expect, test, type Locator } from '@playwright/test';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { captureStudioWindow, enter, fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('a real Model edit updates map, drawer, and Runs from the saved plan', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  const card = page.getByRole('article', { name: 'Train train' });
  const drawer = page.getByRole('navigation', { name: 'Steps' });
  const table = page.getByRole('table', { name: 'Runs' });
  const run = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === `${projectPath}/train-runs`);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  expect((await run).status()).toBe(202);
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 15_000 });

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('tab', { name: 'Model', exact: true }).click();
  await page.getByRole('tabpanel', { name: 'Model', exact: true })
    .getByRole('article', { name: 'Point mass', exact: true }).locator('header').click();
  const mass = page.getByRole('complementary', { name: 'Inspector' }).getByRole('textbox', { name: 'Mass', exact: true });
  await expect(mass).toHaveValue(/^1(?:\.0)?$/);
  await expect(mass).toBeEditable();
  const write = page.waitForResponse((response) => {
    if (response.request().method() !== 'POST' || new URL(response.url()).pathname !== projectPath + '/model/edit') return false;
    return response.request().postDataJSON()?.command?.operation?.op === 'set_field';
  });
  await mass.fill('2');
  await mass.press('Enter');
  const saved = await write;
  expect(saved.ok(), await saved.text()).toBe(true);
  expect((await saved.json()).result.status).toBe('applied');
  await expect(table.getByText('Out of date', { exact: true })).toBeVisible();
  await expect(card.getByText('1 Out of date')).toBeVisible();
  await expect(drawer.getByText('1 Out of date')).toBeVisible();
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: 'Undo', exact: true }).click();
  await expect(table.getByText('Done', { exact: true })).toBeVisible();
  await expect(card.getByText('1 Done')).toBeVisible();
  await expect(drawer.getByText('1 Done')).toBeVisible();
});

test('outside saved-folder edits join the mounted Undo history', async ({ page, request }, info) => {
  test.setTimeout(45_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const root = join(String(info.config.metadata.home), 'state', 'web', 'projects', snapshot.project_id);
  const modelPath = join(root, 'documents', 'point-mass.json');
  const stepPath = join(root, 'steps', 'train.json');
  const panel = (name: string) => page.getByRole('tabpanel', { name, exact: true });
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  async function saved(method: string, suffix: string, action: () => Promise<void>) {
    const transaction = page.waitForResponse((response) => response.request().method() === method &&
      new URL(response.url()).pathname === projectPath + suffix);
    const [response] = await Promise.all([transaction, action()]);
    expect(response.ok(), await response.text()).toBe(true);
    return response;
  }
  async function edit(field: Locator, value: string, model = false) {
    await expect(field).toBeEditable();
    const response = await saved(model ? 'POST' : 'PATCH', model ? '/model/edit' : '', () => enter(field, value));
    if (model) expect((await response.json()).result.status).toBe('applied');
    await expect(field).toHaveValue(value);
  }
  async function menu(label: string) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    const palette = page.getByRole('dialog', { name: 'Command Palette', exact: true });
    await expect(palette.getByRole('option', { name: label, exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(palette).toBeHidden();
  }
  const read = (path: string) => readFile(path, 'utf8');
  async function outside(path: string, contents: string) {
    const temporary = path + '.outside';
    await writeFile(temporary, contents);
    await rename(temporary, path);
  }
  const json = (data: unknown) => JSON.stringify(data, null, 4) + '\n';
  const model = panel('Model');
  const mass = page.getByRole('complementary', { name: 'Inspector' }).getByRole('textbox', { name: 'Mass', exact: true });
  const batches = panel('Training').getByRole('textbox', { name: 'N Batches', exact: true });
  const batchSize = panel('Training').getByRole('textbox', { name: 'Batch Size', exact: true });
  async function history(keys: string, view: string, suffix = '') {
    await panel(view).focus();
    const response = await saved(suffix ? 'POST' : 'PATCH', suffix, () => page.keyboard.press(keys));
    if (suffix === '/model/undo') expect((await response.json()).result.status).toBe('applied');
    return response.request().postDataJSON();
  }

  await page.goto('/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  await tab('Training').click();
  await edit(batchSize, '6');
  await edit(batches, '5');
  await tab('Model').click();
  await model.getByRole('article', { name: 'Point mass', exact: true }).locator('header').click();
  await edit(mass, '2', true);

  const beforeOutside = await read(modelPath);
  const changed = JSON.parse(beforeOutside);
  changed.payload.interior.components.plant.fields.mass = 3;
  await outside(modelPath, json(changed));
  await expect(mass).toHaveValue('3', { timeout: 2_000 });
  await menu('Undo Outside Change');
  await history('Meta+Z', 'Model', '/history/replay');
  await expect(mass).toHaveValue('2');
  expect(await read(modelPath)).toBe(beforeOutside);

  await history('Meta+Z', 'Model', '/model/undo');
  await expect(mass).toHaveValue(/^1(?:\.0)?$/);
  await tab('Training').click();
  await history('Meta+Z', 'Training');
  await expect(batches).toHaveValue('2');
  await history('Meta+Z', 'Training');
  await expect(batchSize).toHaveValue('4');

  await history('Meta+Shift+Z', 'Training');
  await expect(batchSize).toHaveValue('6');
  await history('Meta+Shift+Z', 'Training');
  await expect(batches).toHaveValue('5');
  await tab('Model').click();
  await history('Meta+Shift+Z', 'Model', '/model/undo');
  await expect(mass).toHaveValue('2');
  await menu('Redo Outside Change');
  await history('Meta+Shift+Z', 'Model', '/history/replay');
  await expect(mass).toHaveValue('3');
  expect(await read(modelPath)).toBe(json(changed));

  const step = await read(stepPath);
  const outsideStep = JSON.parse(step);
  outsideStep.base.inline.training_config.batch_size = 8;
  const outsideModel = structuredClone(changed);
  outsideModel.payload.interior.components.plant.fields.mass = 4;
  await outside(modelPath, json(outsideModel));
  await expect(mass).toHaveValue('4', { timeout: 2_000 });
  await outside(stepPath, json(outsideStep));
  await tab('Training').click();
  await expect(batchSize).toHaveValue('8', { timeout: 2_000 });
  const burst = await history('Meta+Z', 'Training', '/history/replay');
  expect(burst.files.map(({ address }: { address: { id: string } }) => address.id).sort()).toEqual(['point-mass', 'train']);
  await expect(batchSize).toHaveValue('6');
  expect(await read(stepPath)).toBe(step);
  expect(await read(modelPath)).toBe(json(changed));
});

test('the managed window shares selection and restores the folded slot without editing the folder', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('files'));
  const id = snapshot.project_id;
  const key = `feedbax:studio-window:v1:${id}`;
  const prefs = () => page.evaluate((key) => JSON.parse(localStorage.getItem(key)!), key);
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  const primary = page.locator('[data-slot="primary"]');
  const secondary = page.locator('[data-slot="secondary"]');
  await expect(primary.getByRole('table', { name: 'Runs' })).toBeVisible();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await expect(secondary.getByRole('heading', { name: 'Scene', exact: true })).toBeVisible();

  await mapStep(page, 'successor').click();
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');

  const separator = page.getByRole('separator', { name: 'Slot Width', exact: true });
  await separator.press('ArrowLeft');
  const width = await separator.getAttribute('aria-valuenow');
  const before = await prefs();
  await page.setViewportSize({ width: 620, height: 850 });
  await expect(separator).toBeVisible();
  await expect(page.getByRole('complementary', { name: 'Details Drawer' })).toBeHidden();
  expect(await prefs()).toEqual(before);
  await page.setViewportSize({ width: 520, height: 850 });
  await expect(separator).toBeHidden();
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  expect(await prefs()).toEqual(before);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(separator).toHaveAttribute('aria-valuenow', width!);

  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  await expect(mapStep(page, 'successor')).toBeHidden();
  await expect(page.locator('.studio-window-map__compact').getByRole('button', { name: 'Select successor' })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Show Details', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(secondary.getByRole('heading', { name: 'Scene', exact: true })).toBeVisible();
  await expect(separator).toHaveAttribute('aria-valuenow', width!);

  const latest = await request.get(projectPath);
  expect(latest.ok()).toBe(true);
  const reopened = await latest.json() as StudioProjectSnapshot;
  expect(reopened.revision).toBe(snapshot.revision);
  expect(reopened.presentation_revision).toBe(snapshot.presentation_revision);
});

test('keyboard reaches the last Runs column in a narrow split', async ({ page, request }, info) => {
  const files = (await fixture('files')).slice(0, 3);
  files[2].step.sweep = { axes: { seed: { values: {
    a: { label: 'DeterministicLongNameForFirstReplicate', patches: { 'method_payload.payload.seed': { op: 'add', value: 7 } } },
    b: { label: 'DeterministicLongNameForSecondReplicate', patches: { 'method_payload.payload.seed': { op: 'add', value: 9 } } },
  } } } };
  const { snapshot } = await seedManagedProject(request, files);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  await page.getByRole('button', { name: 'Hide Drawer', exact: true }).click();
  await page.getByRole('button', { name: 'Hide Inspector', exact: true }).click();
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.getByRole('tabpanel', { name: 'Scene', exact: true })).toBeVisible();
  const table = page.getByRole('table', { name: 'Runs', exact: true });
  const scroll = table.locator('..');
  const progress = table.getByRole('columnheader', { name: 'Progress', exact: true });
  const position = () => scroll.evaluate((element) => ({
    offset: element.scrollLeft,
    limit: element.scrollWidth - element.clientWidth,
  }));
  await expect.poll(async () => (await position()).limit).toBeGreaterThan(0);
  await expect(progress).not.toBeInViewport();
  const row = table.getByRole('button', { name: /FirstReplicate/ });
  await row.focus();
  let at = await position();
  for (let keypress = 0; keypress < 20 && at.limit - at.offset > 1; keypress++) {
    await page.keyboard.press('ArrowRight');
    await expect.poll(async () => (await position()).offset).toBeGreaterThan(at.offset);
    at = await position();
  }
  expect(at.limit - at.offset).toBeLessThanOrEqual(1);
  await expect(progress).toBeInViewport({ ratio: 0.9 });
  await expect(row).toBeFocused();
  await expect(page.locator('.studio-window-slot__strip')).toHaveCount(0);
  await captureStudioWindow(page, info, 'narrow-runs-overflow');
});

test('mounted Scene owns its drawer and inspector and Problems matches the saved Plan', async ({ page, request }, info) => {
  const home = String(info.config.metadata.home);
  const id = 'project-window-scene';
  const path = `/api/projects/${id}`;
  await runProjectFixture('tests.fixtures.scene_geometry', 'seed',
    join(home, 'state', 'web', 'projects', id));
  const initial = await request.get(path);
  expect(initial.ok()).toBe(true);
  let snapshot = await initial.json() as StudioProjectSnapshot;
  const [modelFile, task, template] = await fixture('files');
  for (const file of [
    task,
    { ...template, step: { ...template.step, base: { inline: { ...template.step.base.inline, graph: { document: 'chain' } } } } },
    { ...template, id: 'single', step: { ...template.step, base: { inline: { ...template.step.base.inline, graph: { document: 'single' } } } } },
    { kind: 'step', id: 'broken', step: { kind: 'train', base: { inline: {} } } },
    { kind: 'step', id: 'rowless', step: { kind: 'train', base: { document: 'absent-base' } } },
    { ...modelFile, id: 'empty-model', document: { ...modelFile.document, payload: { nodes: {} } } },
    { ...template, id: 'empty', step: { ...template.step, base: { inline: { ...template.step.base.inline, graph: { document: 'empty-model' } } } } },
  ]) {
    const saved = await request.post(`${path}/files`, { data: { revision: snapshot.revision, file } });
    expect(saved.status(), await saved.text()).toBe(201);
    ({ snapshot } = await saved.json() as StudioProjectDocument);
  }
  await page.goto(`/?project=${id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const scene = page.locator('[data-slot="secondary"]').getByRole('tabpanel', { name: 'Scene', exact: true });
  const upper = scene.getByRole('button', { name: 'Select upper', exact: true });
  await expect(upper).toBeVisible();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer' });
  await drawer.getByRole('region', { name: 'Bodies' }).getByRole('button', { name: 'upper', exact: true }).click();
  await expect(upper).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toContainText('upper.proximal');
  const primary = page.locator('[data-slot="primary"] > section');
  await primary.focus();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await scene.focus();
  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  await scene.focus();
  const model = primary.locator('.react-flow');
  const centeredModel = () => model.evaluate((canvas) => {
    const bounds = canvas.getBoundingClientRect();
    const nodes = [...canvas.querySelectorAll('.react-flow__node')];
    if (nodes.length !== 1) return false;
    const rect = nodes[0].getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && rect.left >= bounds.left && rect.right <= bounds.right
      && rect.top >= bounds.top && rect.bottom <= bounds.bottom
      && Math.abs((rect.left + rect.right) - (bounds.left + bounds.right)) < 6
      && Math.abs((rect.top + rect.bottom) - (bounds.top + bounds.bottom)) < 6;
  });
  await expect.poll(centeredModel).toBe(true);
  const drawing = scene.getByRole('group', { name: 'Scene', exact: true });
  await expect.poll(() => drawing.evaluate((svg: SVGSVGElement) =>
    Math.abs(svg.viewBox.baseVal.width - svg.clientWidth) + Math.abs(svg.viewBox.baseVal.height - svg.clientHeight),
  )).toBeLessThanOrEqual(2);
  await expect(primary.getByRole('heading', { name: 'Model', exact: true })).toBeVisible();
  await expect(primary.locator('.studio-window-slot__header')).toHaveCSS('height', '38px');
  await expect(scene.locator('.studio-window-slot__header')).toHaveCSS('height', '38px');
  await captureStudioWindow(page, info, '02-model-scene');
  await page.setViewportSize({ width: 1240, height: 820 });
  await expect.poll(centeredModel).toBe(true);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect.poll(centeredModel).toBe(true);

  const sceneScale = () => upper.evaluate((limb: SVGGraphicsElement) => limb.getBBox().width);
  const initialScale = await sceneScale();
  await scene.getByRole('button', { name: 'Zoom In', exact: true }).click();
  await expect.poll(sceneScale).toBeGreaterThan(initialScale);
  const deliberateScale = await sceneScale();
  await primary.focus();
  const zoom = () => model.locator('.react-flow__viewport').evaluate((viewport) =>
    new DOMMatrix(getComputedStyle(viewport).transform).a);
  const initialZoom = await zoom();
  await model.hover();
  await page.mouse.wheel(0, 180);
  await expect.poll(zoom).toBeLessThan(initialZoom);
  const deliberateZoom = await zoom();
  const modelWidth = await model.evaluate(({ clientWidth }) => clientWidth);
  await page.setViewportSize({ width: 1160, height: 820 });
  await expect.poll(() => model.evaluate(({ clientWidth }) => clientWidth)).toBeLessThan(modelWidth);
  expect(await zoom()).toBeCloseTo(deliberateZoom);
  expect(await sceneScale()).toBeCloseTo(deliberateScale);
  await scene.getByRole('button', { name: 'Fit View', exact: true }).click();
  await expect.poll(sceneScale).toBeLessThan(deliberateScale);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('button', { name: 'Show Both', exact: true }).click();
  await primary.focus();
  await page.keyboard.press('Control+3');
  await expect(primary.getByRole('heading', { name: 'Task', exact: true })).toBeVisible();
  await expect(primary.locator('.studio-window-slot__controls')).toContainText('task');
  await expect(page.getByRole('complementary', { name: 'Inspector' })).not.toContainText('upper.proximal');
  await scene.focus();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toContainText('upper.proximal');
  await mapStep(page, 'single').click();
  await expect(scene.locator('header')).toContainText('single');
  await expect(scene.getByRole('button', { name: 'Select forearm', exact: true })).toBeHidden();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).not.toContainText('upper.proximal');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(upper).toBeVisible();

  const read = await request.get(`${path}/train-runs?revision=${snapshot.revision}&window_map=true`);
  expect(read.ok()).toBe(true);
  const { window_map } = await read.json() as StudioSavedTrainRuns;
  const problems = window_map!.problems;
  const brokenProblems = problems.filter(({ step }) => step === 'broken');
  expect(brokenProblems.length).toBeGreaterThan(1);
  expect(window_map!.steps.find(({ step_id }) => step_id === 'broken')?.status_counts).toEqual({ cannot_run: 1 });
  expect(window_map!.rows.filter(({ key }) => key.step === 'rowless')).toHaveLength(0);
  await page.getByRole('button', { name: 'Show Both', exact: true }).click();
  const problemButton = page.getByRole('button', { name: 'Problems', exact: true });
  await expect(problemButton).toHaveText(`Problems ${problems.length}`);
  await expect(problemButton).toHaveAccessibleDescription(String(problems.length));
  await problemButton.click();
  const list = page.getByRole('dialog', { name: 'Problems' });
  await expect(list.getByRole('listitem')).toHaveCount(problems.length);
  const close = list.getByRole('button', { name: 'Close', exact: true });
  await expect(close).toBeFocused();
  async function anchored() {
    const entry = (await problemButton.boundingBox())!;
    const panel = (await list.boundingBox())!;
    expect(Math.abs(panel.x + panel.width - entry.x - entry.width)).toBeLessThan(2);
    expect(panel.y + panel.height).toBeLessThan(entry.y);
    expect(panel.x).toBeGreaterThanOrEqual(0);
  }
  await anchored();
  await close.press('Escape');
  await expect(problemButton).toBeFocused();

  const broken = page.getByRole('article', { name: 'Train broken' });
  const scoped = broken.getByRole('button', { name: "1 Can't run", exact: true });
  await scoped.press('Enter');
  await expect(list.getByRole('listitem')).toHaveCount(brokenProblems.length);
  await anchored();
  await close.press('Escape');
  await expect(scoped).toBeFocused();
  await scoped.press('Enter');
  await list.getByRole('button', { name: 'Training Method · Field required', exact: true }).press('Enter');
  await expect(list).toBeHidden();
  const training = page.getByRole('tabpanel', { name: 'Training', exact: true });
  await expect(training.getByText('Training Method · Field required', { exact: true })).toBeFocused();
  await expect(broken.locator('.studio-map-run-mark')).toHaveAttribute('aria-pressed', 'true');

  const rowless = page.getByRole('article', { name: 'Train rowless' });
  await rowless.getByRole('button', { name: "Can't run", exact: true }).click();
  await expect(list.getByRole('listitem')).toHaveCount(1);
  await list.getByRole('button', { name: /absent-base/ }).click();
  await expect(mapStep(page, 'rowless')).toHaveAttribute('aria-pressed', 'true');

  await problemButton.click();
  await list.getByRole('button', { name: /Components · Model has no components/ }).click();
  const emptyReason = page.getByText('Components · Model has no components.', { exact: true });
  const empty = page.getByRole('tabpanel', { name: 'Model', exact: true }).filter({ has: emptyReason });
  await expect(empty.locator('header')).toContainText('empty-model');
  await expect(emptyReason).toBeFocused();
  await expect(page.getByRole('article', { name: 'Train empty' }).locator('.studio-map-run-mark')).toHaveAttribute('aria-pressed', 'true');

  await page.setViewportSize({ width: 520, height: 860 });
  await problemButton.click();
  await expect(list.getByRole('listitem')).toHaveCount(problems.length);
  await anchored();
});
