import { expect, test } from './fixtures/studio';
import { type APIRequestContext, type Locator, type Page } from '@playwright/test';
import type { StoredModelDocument as ModelDocument, StoredPhysicalInterior, ModelEditResponse, StudioProjectDocument, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { palette } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

function card(model: Locator, id: string) {
  return model.locator(`.react-flow__node[data-id="${id}"]`);
}

function edge(model: Locator, id: string) {
  return model.locator(`.react-flow__edge[data-id="${id}"]`);
}

function modelPanel(page: Page) {
  return page.getByRole('tabpanel', { name: 'Model', exact: true });
}

async function revealModel(page: Page, url: string) {
  await page.goto(url);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
}

/** The physical connections stored in the fixture's Equations interior. */
function storedConnections(current: StudioProjectDocument['snapshot']) {
  const document = rootPayload(current.project.documents?.['reach-model']) as unknown as ModelDocument;
  const component = document.interior.components?.equations;
  const interior = component?.kind === 'composite' ? component.interior : undefined;
  if (!interior?.physical_domain) {
    throw new Error('Expected a stored Equations interior');
  }
  return (interior as StoredPhysicalInterior).connections ?? {};
}

async function openSaved(request: APIRequestContext, projectPath: string) {
  const response = await request.get(projectPath + '/open');
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json() as StudioProjectDocument).snapshot;
}

test('edits a conserving connection only inside its physical interior', async ({ page, request, studio }, info) => {
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, []);
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, configuration.project, 'equations');
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const model = modelPanel(page);
  const fit = () => model.getByRole('button', { name: 'Fit View', exact: true }).click();
  const rootPort = card(model, 'equations').locator('[data-handleid="physical:free"]');
  const boundaryPort = model.locator('[data-boundary="true"] [data-handleid="physical:free"]');
  const massPort = card(model, 'mass').locator('[data-handleid="physical:flange"]');
  const connectionLine = model.locator('.react-flow__connection-path');
  const root = { composites: ['equations'], element: { kind: 'graph' } };
  const stored = storedConnections;
  async function apply(kind: 'edit' | 'undo', action: () => Promise<void>) {
    const [response] = await Promise.all([page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === `${projectPath}/model/${kind}`, { timeout: 6_000 }), action()]);
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    if (result.status !== 'applied') throw new Error(JSON.stringify(result));
    return result;
  }
  async function enterEquations() {
    await card(model, 'equations').locator('header').dblclick();
    await expect(boundaryPort).toBeVisible();
    await fit();
  }
  /** A reloaded window reopens the Model where it was left, inside Equations. */
  async function reopenedInEquations() {
    await expect(boundaryPort).toBeVisible();
    await fit();
  }

  await revealModel(page, url);
  await fit();
  await expect(rootPort).toBeVisible();
  await expect(rootPort).toHaveCSS('border-radius', '50%');
  const initial = await openSaved(request, projectPath);
  const point = await rootPort.boundingBox();
  if (!point) throw new Error('The Graph physical handle is not measurable');
  await rootPort.hover();
  await page.mouse.down();
  await page.mouse.move(point.x - 45, point.y + 30, { steps: 6 });
  await expect(connectionLine).toHaveCount(0);
  await page.mouse.up();
  expect((await openSaved(request, projectPath)).revision).toBe(initial.revision);

  await enterEquations();
  await expect(massPort).toHaveCSS('border-radius', '50%');
  await expect(boundaryPort).toHaveCSS('border-radius', '50%');
  const connected = await apply('edit', async () => {
    await massPort.hover();
    await page.mouse.down();
    await boundaryPort.hover();
    await expect(connectionLine).toHaveAttribute('d', /^M/);
    await page.mouse.up();
  });
  const [address] = connected.changed;
  expect(address).toMatchObject({ ...root, element: { kind: 'physical_connection' } });
  if (address.element.kind !== 'physical_connection') throw new Error('Expected a physical connection address');
  const id = address.element.id;
  const connection = stored(connected.snapshot)[id];
  expect([connection.a, connection.b]).toEqual(expect.arrayContaining([
    { owner: 'component', component: 'mass', port: 'flange' },
    { owner: 'boundary', port: 'free' },
  ]));
  await revealModel(page, url);
  await reopenedInEquations();
  const savedEdge = edge(model, id);
  const savedPath = savedEdge.locator('.react-flow__edge-path');
  await expect(savedPath).toHaveAttribute('d', /^M/);
  const midpoint = await savedPath.evaluate((path: SVGPathElement) => {
    const transform = path.getScreenCTM();
    if (!transform) throw new Error('The saved connection has no screen transform');
    const { x, y } = path.getPointAtLength(path.getTotalLength() / 2).matrixTransform(transform);
    return { x, y };
  });
  await page.mouse.click(midpoint.x, midpoint.y);
  const removed = await apply('edit', () => model.press('Backspace'));
  expect(stored(removed.snapshot)).toEqual({});
  await expect(savedEdge).toHaveCount(0);
  const restored = await apply('undo', () => model.press('Meta+Z'));
  expect(stored(restored.snapshot)).toEqual({ [id]: connection });
  await revealModel(page, url);
  await reopenedInEquations();
  await expect(savedPath).toHaveAttribute('d', /^M/);
  expect(stored(await openSaved(request, projectPath))).toEqual({ [id]: connection });
});

/**
 * A Spring dropped from the Library onto the connection between a Mass and the boundary's physical port is stored as
 * two physical connections through it, which a reopened window draws (catches a connection drawn and not stored).
 */
test('a Library element dropped on a physical connection is joined into it', async ({ page, request, studio }) => {
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, []);
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, configuration.project, 'equations');
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const model = modelPanel(page);
  const boundaryPort = model.locator('[data-boundary="true"] [data-handleid="physical:free"]');
  const massPort = card(model, 'mass').locator('[data-handleid="physical:flange"]');
  const connections = model.locator('.react-flow__edge');
  const edited = async (action: () => Promise<void>) => {
    const [response] = await Promise.all([page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === `${projectPath}/model/edit`), action()]);
    const { result } = await response.json() as ModelEditResponse;
    if (result.status !== 'applied') throw new Error(JSON.stringify(result));
    return result;
  };

  await revealModel(page, url);
  await card(model, 'equations').locator('header').dblclick();
  await model.getByRole('button', { name: 'Fit View', exact: true }).click();
  await edited(async () => {
    await massPort.hover();
    await page.mouse.down();
    await boundaryPort.hover();
    await page.mouse.up();
  });
  await expect(connections).toHaveCount(1);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('LinearSpring');
  const spring = drawer.getByRole('option', { name: 'LinearSpring', exact: true });
  const path = connections.first().locator('path.react-flow__edge-path');
  // A point of the connection that no card covers.
  const middle = await path.evaluate((line: SVGPathElement) => {
    const length = line.getTotalLength();
    for (const fraction of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) {
      const { x, y } = line.getPointAtLength(length * fraction).matrixTransform(line.getScreenCTM()!);
      if (document.elementFromPoint(x, y)?.closest('.react-flow__edge')) return { x, y };
    }
    throw new Error('Every point of the connection is covered');
  });
  const inserted = await edited(async () => {
    await spring.hover();
    await page.mouse.down();
    await page.mouse.move(middle.x, middle.y, { steps: 6 });
    await expect(model.locator('.react-flow__edge.studio-carried')).toHaveCount(2);
    await page.mouse.up();
  });
  const saved = storedConnections(inserted.snapshot);
  const ends = Object.values(saved).flatMap(({ a, b }) => [a, b]);
  const springId = ends.find((end) => end.owner === 'component' && end.component !== 'mass');
  if (springId?.owner !== 'component') throw new Error('Expected a connection to the dropped Spring');
  expect(Object.values(saved)).toHaveLength(2);
  expect(ends).toEqual(expect.arrayContaining([
    { owner: 'component', component: 'mass', port: 'flange' },
    { owner: 'boundary', port: 'free' },
    { owner: 'component', component: springId.component, port: 'flange_a' },
    { owner: 'component', component: springId.component, port: 'flange_b' },
  ]));

  await revealModel(page, url);
  await expect(boundaryPort).toBeVisible();
  for (const id of Object.keys(saved)) {
    await expect(model.locator(`.react-flow__edge[data-id="${id}"] path.react-flow__edge-path`)).toHaveAttribute('d', /^M/);
  }
  expect(storedConnections(await openSaved(request, projectPath))).toEqual(saved);
});

for (const { scope, element, composites, target } of [
  { scope: 'mechanics-port', element: { kind: 'port', end: { owner: 'boundary', port: 'mount' } },
    composites: ['arm'], target: (model: Locator) => model.locator('[data-boundary="true"] [data-handleid="physical:mount"]') },
  { scope: 'within-step', element: { kind: 'wire', id: 'drive' }, composites: [],
    target: (model: Locator) => edge(model, 'drive') },
]) {
  test(`reveals the saved Plan problem at its ${element.kind} address (${scope})`, async ({ page, request, studio }, info) => {
    const { snapshot, projectPath, configuration } = await seedManagedProject(request, []);
    await runProjectFixture('web.e2e.studio.fixtures.stored_model', studio.home, configuration.project, scope);
    // The window's plan request names the revision it read; the test reads that plan itself,
    // since the browser may already have dropped the body when the page refetches.
    const planRead = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/train-runs` &&
      new URL(response.url()).searchParams.get('window_map') === 'true');
    await revealModel(page, `/?project=${encodeURIComponent(snapshot.project_id)}`);
    const response = await request.get((await planRead).url());
    expect(response.ok(), await response.text()).toBe(true);
    const { window_map: map } = await response.json() as StudioSavedTrainRuns;
    expect(map?.rows.find((row) => row.key.step === 'train')?.status).toBe('cannot_run');
    const address = { composites, element };
    const problem = map?.problems.find((candidate) => candidate.step === 'train');
    expect(problem?.place).toMatchObject({ part: 'model', member: address });
    expect(problem?.detail).toBeTruthy();
    if (!problem?.detail) throw new Error('Expected an addressed Plan problem');
    const reason = problem.detail;
    await palette(page, 'problems', 'Problems');
    await page.getByRole('dialog', { name: 'Problems' }).getByRole('button').filter({ hasText: reason }).click();
    await expect(target(modelPanel(page))).toHaveCount(1);
    await expect(modelPanel(page).getByText(reason, { exact: false }).first()).toBeFocused();
  });
}
