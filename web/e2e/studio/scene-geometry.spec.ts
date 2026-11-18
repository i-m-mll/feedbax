import { expect, test } from './fixtures/studio';
import { runPython } from './fixtures/python';
import { type APIRequestContext, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { PlanarScenePose } from '../../src/components/scene/planarModel';
import type { ModelEditResponse, StudioPlanarSceneRequest, StudioProjectDocument } from '../../src/generated/studioContracts';
import type { mountSceneProbe } from './fixtures/sceneGeometryProbe';

test.use({ studioFrontend: 'dev' });

const alias = 'scene-geometry';
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
  await runPython({ module: 'tests.fixtures.scene_geometry' }, [action, ...args]);
  return action === 'read' ? JSON.parse(await readFile(output, 'utf8')) as Record<string, NativeModel> : {};
}

function source(document_id: string, revision: string): StudioPlanarSceneRequest {
  return { document_id, revision, node_path: ['mechanics'] };
}

async function mount(page: Page, projectId: string) {
  await page.evaluate(async (id) => {
    const entry = '/e2e/studio/fixtures/sceneGeometryProbe.tsx';
    const probe = await import(entry) as { mountSceneProbe: typeof mountSceneProbe };
    const container = document.createElement('section');
    container.id = 'scene-proof';
    document.body.replaceChildren(container);
    (window as unknown as Probe).sceneProof = probe.mountSceneProbe(container, id);
  }, projectId);
}

async function show(page: Page, address: StudioPlanarSceneRequest, pose: PlanarScenePose) {
  const version = await page.evaluate(({ address, pose }) => (window as unknown as Probe).sceneProof.show(address, pose), { address, pose });
  await expect.poll(() => page.evaluate(() => {
    const current = (window as unknown as Probe).sceneProof.current();
    return current?.view.kind === 'geometry' ? current.version : null;
  })).toBe(version);
}

type Revisions = { revision: string; presentation_revision: string };
type FieldEdit = { component: string; path: (string | number)[]; value: unknown };

/** Sets fields of the `mechanics` graph of the `chain` Model through addressed edits; returns the final revisions. */
async function setFields(request: APIRequestContext, projectUrl: string, revisions: Revisions, fields: FieldEdit[]) {
  for (const { component, path, value } of fields) {
    const response = await request.post(`${projectUrl}/model/edit`, { data: {
      ...revisions, document: 'chain', command: { operation: { op: 'set_field', value, address: {
        composites: ['mechanics'],
        element: { kind: 'field', parent: { kind: 'component', id: component }, path: ['fields', ...path] },
      } } },
    } });
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    if (result.status !== 'applied') throw new Error(`Model edit refused: ${JSON.stringify(result.problem)}`);
    revisions = { revision: result.snapshot.revision, presentation_revision: result.snapshot.presentation_revision };
  }
  return revisions;
}

async function selectFixedGuide(page: Page) {
  const proof = page.locator('#scene-proof');
  const muscle = proof.getByRole('button', { name: 'Select cross_chain', exact: true });
  await muscle.locator(':scope > circle').nth(1).click();
  await expect(muscle).toHaveAttribute('aria-pressed', 'true');
  await expect(proof.getByRole('region', { name: 'Muscles' }).getByRole('treeitem', { name: 'cross_chain' }))
    .toHaveAttribute('aria-selected', 'true');
}

async function expectInvalidGeometry(page: Page, code: string, message: RegExp) {
  const proof = page.locator('#scene-proof');
  await expect(proof.getByRole('alert')).toContainText(message);
  await expect(proof.locator('svg')).toHaveCount(0);
  const codes = await page.evaluate(() => {
    const view = (window as unknown as Probe).sceneProof.current()?.view;
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
    const view = (window as unknown as Probe).sceneProof.current()?.view;
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

test('M2 Scene geometry agrees with compiled kinematics across a saved folder edit', async ({ page, request, studio }, info) => {
  test.setTimeout(40_000);
  const folder = join(studio.home, 'state', 'web', 'projects', alias);
  await fixture('seed', folder);
  const initial = await fixture('read', folder, 'single', 'chain');
  const opened = await request.get(`/api/projects/${alias}/open`);
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  const projectId = snapshot.project_id;
  const projectUrl = `/api/projects/${encodeURIComponent(projectId)}`;
  await page.goto('/');
  await mount(page, projectId);
  const geometryRequests: string[] = [];
  page.on('request', (req) => {
    if (new URL(req.url()).pathname === `${projectUrl}/scene/planar`) geometryRequests.push(req.postDataJSON().document_id);
  });

  const residuals = [];
  for (const [document, model] of Object.entries(initial)) {
    for (const sample of model.samples) residuals.push(await comparePose(page, source(document, model.revision), sample));
  }
  await selectFixedGuide(page);
  const guide = page.locator('#scene-proof').getByRole('region', { name: 'cross_chain', exact: true })
    .getByRole('textbox', { name: 'forearm.distal guide', exact: true });
  await expect(guide).toHaveValue('0.03, -0.09');

  expect(snapshot.revision).toBe(initial.chain.revision);
  const editedRevisions = await setFields(request, projectUrl, {
    revision: snapshot.revision, presentation_revision: snapshot.presentation_revision,
  }, [
    { component: 'forearm', path: ['length'], value: 0.37 },
    { component: 'cross_chain', path: ['path_points', 1, 'offset'], value: [0.05, -0.07] },
  ]);
  const editedRevision = editedRevisions.revision;
  const { chain: edited } = await fixture('read', folder, 'chain');
  expect(edited.revision).toBe(editedRevision);
  expect(edited.identity).not.toBe(initial.chain.identity);
  await page.reload();
  await mount(page, projectId);
  const address = source('chain', editedRevision);
  for (const sample of edited.samples) residuals.push(await comparePose(page, address, sample));
  residuals.push(await comparePose(page, address, edited.samples[0], {
    kind: 'recorded', sample: { modelIdentity: edited.identity, qpos: edited.samples[0].qpos },
  }));
  await selectFixedGuide(page);
  await expect(guide).toHaveValue('0.05, -0.07');

  await show(page, address, { kind: 'recorded', sample: { modelIdentity: initial.chain.identity, qpos: edited.samples[0].qpos } });
  await expectInvalidGeometry(page, 'scene.pose_model_mismatch', /different model/);
  await show(page, address, { kind: 'recorded', sample: null });
  await expectInvalidGeometry(page, 'scene.pose_unavailable', /no planar pose/);
  await show(page, address, { kind: 'inspection', qpos: [0.4] });
  await expectInvalidGeometry(page, 'scene.invalid_pose', /coordinate per joint/);
  const { revision: invalid } = await setFields(request, projectUrl, editedRevisions, [
    { component: 'cross_chain', path: ['path_points', 1, 'frame'], value: 'missing.distal' },
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
  await page.evaluate(() => (window as unknown as Probe).sceneProof.close());
});
