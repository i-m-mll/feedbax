import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { PlanarScenePose } from '../../src/components/scene/planarModel';
import type { StudioPlanarSceneRequest } from '../../src/generated/studioContracts';
import type { mountSceneProbe } from './fixtures/sceneGeometryProbe';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
const projectId = 'scene-geometry';
const projectUrl = `/api/projects/${projectId}`;
const params = '/payload/interior/components/mechanics/interior/components';
type Probe = Window & { sceneProof: ReturnType<typeof mountSceneProbe> };
type Point = [number, number];
type NativeSample = {
  qpos: number[];
  bodies: Record<string, { start: Point; end: Point; rotation: number[] }>;
  joints: Record<string, Point>;
  muscles: Record<string, Point[]>;
};
type NativeModel = { identity: string; revision: string; samples: NativeSample[] };

test.use({ locale: 'en-US' });

async function fixture(action: 'seed' | 'read', folder: string, ...documents: string[]) {
  const output = join(folder, '..', '..', 'scene-native.json');
  const args = action === 'seed' ? [folder] : [folder, output, ...documents];
  await run('uv', ['run', '--no-sync', 'python', '-B', '-m', 'tests.fixtures.scene_geometry', action, ...args],
    { cwd: repo, timeout: 15_000 });
  return action === 'read' ? JSON.parse(await readFile(output, 'utf8')) as Record<string, NativeModel> : {};
}

function source(document_id: string, revision: string): StudioPlanarSceneRequest {
  return { document_id, revision, node_path: ['mechanics'] };
}

async function mount(page: Page) {
  await page.evaluate(async (id) => {
    const entry = '/e2e/studio/fixtures/sceneGeometryProbe.tsx';
    const { mountSceneProbe } = await import(entry) as { mountSceneProbe: typeof mountSceneProbe };
    const container = document.createElement('section');
    container.id = 'scene-proof';
    document.body.replaceChildren(container);
    (window as Probe).sceneProof = mountSceneProbe(container, id);
  }, projectId);
}

async function show(page: Page, address: StudioPlanarSceneRequest, pose: PlanarScenePose) {
  const version = await page.evaluate(({ address, pose }) => (window as Probe).sceneProof.show(address, pose), { address, pose });
  await expect.poll(() => page.evaluate(() => {
    const current = (window as Probe).sceneProof.current();
    return current?.view.kind === 'geometry' ? current.version : null;
  })).toBe(version);
}

async function patch(request: APIRequestContext, revision: string, patches: unknown[]) {
  const response = await request.patch(projectUrl, { data: {
    revision, address: { kind: 'document', id: 'chain' }, patches,
  } });
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json() as { revision: string }).revision;
}

async function selectFixedGuide(page: Page) {
  const proof = page.locator('#scene-proof');
  const muscle = proof.getByRole('button', { name: 'Select cross_chain', exact: true });
  await muscle.locator(':scope > circle').nth(1).click();
  await expect(muscle).toHaveAttribute('aria-pressed', 'true');
  await expect(proof.getByRole('region', { name: 'Muscles' }).getByRole('button', { name: 'cross_chain' }))
    .toHaveAttribute('aria-pressed', 'true');
}

async function expectInvalidGeometry(page: Page, code: string, message: RegExp) {
  const proof = page.locator('#scene-proof');
  await expect(proof.getByRole('alert')).toContainText(message);
  await expect(proof.locator('svg')).toHaveCount(0);
  const codes = await page.evaluate(() => {
    const view = (window as Probe).sceneProof.current()?.view;
    return view?.kind === 'geometry' ? view.scene.validation.map((issue) => issue.type) : [];
  });
  expect(codes).toEqual([code]);
}

function distance(actual: Point, expected: Point) {
  return Math.hypot(actual[0] - expected[0], actual[1] - expected[1]);
}

async function comparePose(page: Page, address: StudioPlanarSceneRequest, sample: NativeSample, pose: PlanarScenePose = { kind: 'inspection', qpos: sample.qpos }) {
  await show(page, address, pose);
  const scene = await page.evaluate(() => {
    const view = (window as Probe).sceneProof.current()?.view;
    if (view?.kind !== 'geometry') throw new Error('Planar Scene was not resolved');
    return view.scene;
  });
  expect(scene.validation).toEqual([]);
  const polylines = await page.locator('#scene-proof svg g[aria-label] > polyline').evaluateAll((lines) => Object.fromEntries(
    lines.map((line) => [line.parentElement?.getAttribute('aria-label'), line.getAttribute('points')?.split(' ').map((point) => point.split(',').map(Number))]),
  )) as Record<string, Point[]>;
  expect(Object.keys(polylines)).toHaveLength(Object.keys(sample.bodies).length + Object.keys(sample.muscles).length);
  const basis = sample.bodies.upper;
  const [screenStart, screenEnd] = polylines['Select upper'];
  const scale = distance(screenEnd, screenStart) / distance(basis.end, basis.start);
  const fromScreen = ([x, y]: Point): Point => [basis.start[0] + (x - screenStart[0]) / scale, basis.start[1] - (y - screenStart[1]) / scale];
  let maxPosition = 0;
  let maxRotation = 0;
  for (const [name, body] of Object.entries(sample.bodies)) {
    const [a, b] = polylines[`Select ${name}`];
    const frame = scene.anchors.find((anchor) => anchor.label === `${name}.proximal`);
    const tip = scene.anchors.find((anchor) => anchor.label === `${name}.distal`);
    expect(frame?.position).toBeTruthy();
    expect(tip?.position).toBeTruthy();
    const angle = Number(frame?.metadata.angle);
    const rotation = [Math.cos(angle), -Math.sin(angle), Math.sin(angle), Math.cos(angle)];
    maxRotation = Math.max(maxRotation, ...rotation.map((value, index) => Math.abs(value - body.rotation[index])));
    maxPosition = Math.max(maxPosition, distance(frame!.position!, body.start), distance(tip!.position!, body.end),
      distance(fromScreen(a), body.start), distance(fromScreen(b), body.end));
  }
  for (const [joint, point] of Object.entries(sample.joints)) {
    const circle = page.getByRole('button', { name: `Select ${joint}`, exact: true }).locator('circle').first();
    const screen = await circle.evaluate((node) => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))] as Point);
    maxPosition = Math.max(maxPosition, distance(fromScreen(screen), point));
  }
  for (const [name, points] of Object.entries(sample.muscles)) {
    const rendered = polylines[`Select ${name}`];
    expect(rendered).toHaveLength(points.length);
    for (const [index, point] of points.entries()) maxPosition = Math.max(maxPosition, distance(fromScreen(rendered[index]), point));
  }
  expect(maxPosition).toBeLessThanOrEqual(1e-9);
  expect(maxRotation).toBeLessThanOrEqual(1e-10);
  return { maxPosition, maxRotation };
}

test('M2 Scene geometry agrees with compiled kinematics across a saved folder edit', async ({ page, request }, info) => {
  test.setTimeout(40_000);
  const folder = join(String(info.config.metadata.home), 'state', 'web', 'projects', projectId);
  await fixture('seed', folder);
  const initial = await fixture('read', folder, 'single', 'chain');
  await page.goto('/');
  await mount(page);
  const geometryRequests: string[] = [];
  page.on('request', (req) => {
    if (new URL(req.url()).pathname === `${projectUrl}/scene/planar`) geometryRequests.push(req.postDataJSON().document_id);
  });

  const residuals = [];
  for (const [document, model] of Object.entries(initial)) {
    for (const sample of model.samples) residuals.push(await comparePose(page, source(document, model.revision), sample));
  }
  await selectFixedGuide(page);
  const guide = page.getByRole('region', { name: 'Details', exact: true }).locator('div').filter({
    has: page.getByText('forearm.distal guide', { exact: true }),
  }).locator('dd');
  await expect(guide).toHaveText('0.03, -0.09 m');

  const editedRevision = await patch(request, initial.chain.revision, [
    { op: 'replace', path: `${params}/forearm/fields/length`, value: 0.37 },
    { op: 'replace', path: `${params}/cross_chain/fields/path_points/1/offset`, value: [0.05, -0.07] },
  ]);
  const { chain: edited } = await fixture('read', folder, 'chain');
  expect(edited.revision).toBe(editedRevision);
  expect(edited.identity).not.toBe(initial.chain.identity);
  await page.reload();
  await mount(page);
  const address = source('chain', editedRevision);
  for (const sample of edited.samples) residuals.push(await comparePose(page, address, sample));
  residuals.push(await comparePose(page, address, edited.samples[0], {
    kind: 'recorded', sample: { modelIdentity: edited.identity, qpos: edited.samples[0].qpos },
  }));
  await selectFixedGuide(page);
  await expect(guide).toHaveText('0.05, -0.07 m');

  await show(page, address, { kind: 'recorded', sample: { modelIdentity: initial.chain.identity, qpos: edited.samples[0].qpos } });
  await expectInvalidGeometry(page, 'scene.pose_model_mismatch', /different model/);
  await show(page, address, { kind: 'recorded', sample: null });
  await expectInvalidGeometry(page, 'scene.pose_unavailable', /no planar pose/);
  await show(page, address, { kind: 'inspection', qpos: [0.4] });
  await expectInvalidGeometry(page, 'scene.invalid_pose', /coordinate per joint/);
  const invalid = await patch(request, editedRevision, [
    { op: 'replace', path: `${params}/cross_chain/fields/path_points/1/frame`, value: 'missing.distal' },
  ]);
  await show(page, source('chain', invalid), { kind: 'reference' });
  await expectInvalidGeometry(page, 'mechanics.muscle_path_missing_frame', /cross_chain.*missing\.distal/);
  expect(geometryRequests).toEqual(['single', 'chain', 'chain', 'chain']);
  const measured = {
    samples: residuals.length,
    maxPosition: Math.max(...residuals.map((item) => item.maxPosition)),
    maxRotation: Math.max(...residuals.map((item) => item.maxRotation)),
  };
  console.log('Scene kinematics', JSON.stringify(measured));
  await page.evaluate(() => (window as Probe).sceneProof.close());
});
