import { expect, test, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { PlanarOperation } from '../../src/features/domains/planarAuthoring';
import type { mountAuthoringProbe } from './fixtures/sceneAuthoringProbe';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
const projectId = 'scene-authoring';
const qpos = [0.43, -0.67, 0.26];
type Probe = Window & { sceneAuthoring: Awaited<ReturnType<typeof mountAuthoringProbe>> };
test.use({ locale: 'en-US' });

async function mount(page: Page) {
  await page.evaluate(async (id) => {
    const entry = '/e2e/studio/fixtures/sceneAuthoringProbe.tsx';
    const { mountAuthoringProbe } = await import(entry) as { mountAuthoringProbe: typeof mountAuthoringProbe };
    const container = document.createElement('section');
    container.id = 'scene-authoring';
    document.body.replaceChildren(container);
    (window as Probe).sceneAuthoring = await mountAuthoringProbe(container, id);
  }, projectId);
}

const state = (page: Page) => page.evaluate(() => (window as Probe).sceneAuthoring.state());
const perform = (page: Page, operation: PlanarOperation) => page.evaluate((op) => (window as Probe).sceneAuthoring.operate(op), operation);
const flush = (page: Page) => page.evaluate(() => (window as Probe).sceneAuthoring.flush());

async function guideState(page: Page) {
  const result = await state(page);
  return result.scene?.anchors.find((anchor) => anchor.label === 'forearm.distal guide');
}

async function showInspection(page: Page) {
  await page.evaluate((pose) => (window as Probe).sceneAuthoring.show({ kind: 'inspection', qpos: pose }), qpos);
  await expect.poll(async () => (await state(page)).scene?.anchors.find((anchor) => anchor.label === 'upper.proximal')?.metadata.angle)
    .toBeCloseTo(0.22 + qpos[0], 12);
}

test('M2 Scene authoring shares fields, drag previews, history and the saved model', async ({ page }, info) => {
  test.setTimeout(45_000);
  const home = String(info.config.metadata.home);
  const folder = join(home, 'state', 'web', 'projects', projectId);
  const fixture = ['run', '--no-sync', 'python', '-B', '-m', 'tests.fixtures.scene_geometry'];
  await run('uv', [...fixture, 'seed', folder], { cwd: repo, timeout: 12_000 });
  await page.goto('/');
  await mount(page);
  const proof = page.locator('#scene-authoring');
  const bodies = proof.getByRole('region', { name: 'Bodies' });
  const muscles = proof.getByRole('region', { name: 'Muscles' });
  await expect(bodies.getByRole('button', { name: 'upper', exact: true })).toBeVisible();
  for (const op of [
    { kind: 'append-link', node: 'forearm', joint: 'elbow', length: 0.32 },
    { kind: 'append-link', node: 'hand', joint: 'wrist', length: 0.23 },
    { kind: 'bind-joint', node: 'elbow', socket: 'child', frame: 'forearm.distal' },
    { kind: 'set-joint', node: 'elbow', field: 'frame_angle', value: -0.36 },
    { kind: 'add-muscle', node: 'cross_chain', points: [
      { frame: 'base.frame', offset: [-0.11, 0.14] },
      { frame: 'forearm.distal', offset: [0.03, -0.09] },
      { frame: 'hand.distal', offset: [-0.07, 0.05] },
    ] },
  ] satisfies PlanarOperation[]) expect(await perform(page, op)).toMatchObject({ ok: true });
  await flush(page);
  await expect(bodies.getByRole('button')).toHaveCount(3);
  expect(await perform(page, { kind: 'remove-tail', node: 'hand' })).toMatchObject({ ok: false, reason: /cross_chain/ });
  await showInspection(page);
  await muscles.getByRole('button', { name: 'cross_chain' }).waitFor();

  await bodies.getByRole('button', { name: 'forearm' }).click();
  const length = proof.getByRole('spinbutton', { name: 'Length (m)' });
  const edit = { kind: 'set-link', node: 'forearm', field: 'length', value: 0.37 } as const;
  const expected = await page.evaluate((op) => (window as Probe).sceneAuthoring.expected(op), edit);
  await length.fill('0.99');
  await length.press('Escape');
  await expect(length).toHaveValue('0.32');
  await length.fill('0.37');
  await length.press('Enter');
  const afterField = await state(page);
  expect(afterField.draft).toEqual(expected);
  expect(afterField.assembly?.rows.find((item) => item.node_id === 'forearm')?.length).toBe(0.37);
  const segment = afterField.scene?.elements.find((item) => item.entity_id === afterField.scene?.entities.find((entity) => entity.label === 'forearm')?.id);
  expect(segment?.geometry.kind).toBe('polyline');
  if (segment?.geometry.kind !== 'polyline') throw new Error('Missing forearm preview');
  const [start, end] = segment.geometry.points;
  expect(Math.hypot(end[0] - start[0], end[1] - start[1])).toBeCloseTo(0.37, 12);
  expect(afterField.undoName).toBe('Set link length');
  await flush(page);
  await expect.poll(async () => (await state(page)).compiledIdentity).toMatch(/^[0-9a-f]{64}$/);

  await muscles.getByRole('button', { name: 'cross_chain' }).click();
  const handle = proof.getByRole('button', { name: 'forearm.distal guide' });
  await handle.scrollIntoViewIfNeeded();
  const original = (await guideState(page))!;
  const beforeGesture = await state(page);
  let prepareCount = 0;
  page.on('request', (request) => { if (request.url().endsWith('/scene/planar')) prepareCount += 1; });
  async function moveGuide() {
    const box = await handle.boundingBox();
    if (!box) throw new Error('Missing attachment handle');
    const x = box.x + box.width / 2;
    const y = box.y + box.height / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 35, y - 22, { steps: 6 });
  }
  await moveGuide();
  const preview = (await guideState(page))!;
  expect(preview.position).not.toEqual(original.position);
  expect(preview.metadata.authored_frame).toBe('forearm.distal');
  expect((await state(page)).draft).toEqual(beforeGesture.draft);
  expect(prepareCount).toBe(0);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  expect(await guideState(page)).toMatchObject({ id: original.id, position: original.position });
  expect((await state(page)).undoName).toBe(beforeGesture.undoName);

  await moveGuide();
  const finalOffset = (await guideState(page))!.metadata.offset;
  await page.mouse.up();
  expect((await state(page)).points).toMatchObject([{}, { frame: 'forearm.distal', offset: finalOffset }, {}]);
  expect((await state(page)).undoName).toBe('Edit attachment');
  await page.evaluate(() => (window as Probe).sceneAuthoring.undo());
  expect((await state(page)).points).toMatchObject([{}, { frame: 'forearm.distal', offset: [0.03, -0.09] }, {}]);
  expect((await state(page)).undoName).toBe('Set link length');
  await page.evaluate(() => (window as Probe).sceneAuthoring.redo());
  const revision = await flush(page);
  await expect.poll(async () => (await state(page)).compiledIdentity).toMatch(/^[0-9a-f]{64}$/);
  const saved = await state(page);
  await page.evaluate(() => (window as Probe).sceneAuthoring.close());
  await page.reload();
  await mount(page);
  await showInspection(page);
  await expect.poll(async () => (await guideState(page))?.id).toBe(original.id);
  const reopened = await state(page);
  expect(reopened.savedRevision).toBe(revision);
  expect(reopened.draft).toEqual(saved.draft);
  expect(reopened.points).toMatchObject([{}, { frame: 'forearm.distal', offset: finalOffset }, {}]);

  const output = join(home, 'authoring-native.json');
  await run('uv', [...fixture, 'read', folder, output, 'single'], { cwd: repo, timeout: 15_000 });
  const compiled = JSON.parse(await readFile(output, 'utf8')).single;
  expect(compiled.identity).toBe(reopened.compiledIdentity);
  const nativeGuide = compiled.samples[0].muscles.cross_chain[1] as [number, number];
  const observed = (await guideState(page))!.position!;
  const residual = Math.hypot(observed[0] - nativeGuide[0], observed[1] - nativeGuide[1]);
  expect(residual).toBeLessThanOrEqual(1e-9);
  console.log('Scene authoring guide', JSON.stringify({ frame: 'forearm.distal', offset: finalOffset, residual }));
  await page.evaluate(() => (window as Probe).sceneAuthoring.close());
});
