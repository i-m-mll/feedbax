import { expect, test, type Locator, type Page } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, StudioProjectDocument, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', viewport: { width: 1440, height: 900 } });

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

test('edits a conserving connection only inside its physical interior', async ({ page, request }, info) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', info.config.metadata.home, snapshot.project_id, 'equations');
  const url = `/?project=${encodeURIComponent(snapshot.project_id)}`;
  const model = modelPanel(page);
  const fit = () => model.getByRole('button', { name: 'Fit View', exact: true }).click();
  const rootPort = card(model, 'equations').locator('[data-handleid="physical:free"]');
  const boundaryPort = model.locator('[data-boundary="true"] [data-handleid="physical:free"]');
  const massPort = card(model, 'mass').locator('[data-handleid="physical:flange"]');
  const connectionLine = model.locator('.react-flow__connection-path');
  const root = { document: 'reach-model', composites: ['equations'], element: { kind: 'graph' } };
  const stored = (current: StudioProjectDocument['snapshot']) => {
    const document = current.project.documents?.['reach-model'].payload as unknown as ModelDocument;
    const component = document.interior.components?.equations;
    if (component?.kind !== 'composite' || !('connections' in component.interior)) {
      throw new Error('Expected a stored Equations interior');
    }
    return component.interior.connections ?? {};
  };
  const openSaved = async () => {
    const response = await request.get(projectPath + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  };
  async function apply(kind: 'edit' | 'undo', action: () => Promise<void>) {
    const [response] = await Promise.all([page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === `${projectPath}/model/${kind}`, { timeout: 6_000 }), action()]);
    expect(response.ok(), await response.text()).toBe(true);
    const { result } = await response.json() as ModelEditResponse;
    if (result.status !== 'applied') throw new Error(result.problem.detail);
    return result;
  }
  async function enterEquations() {
    await card(model, 'equations').locator('header').dblclick();
    await expect(boundaryPort).toBeVisible();
    await fit();
  }

  await revealModel(page, url);
  await fit();
  await expect(rootPort).toBeVisible();
  await expect(rootPort).toHaveCSS('border-radius', '50%');
  const initial = await openSaved();
  const point = await rootPort.boundingBox();
  if (!point) throw new Error('The Graph physical handle is not measurable');
  await rootPort.hover();
  await page.mouse.down();
  await page.mouse.move(point.x - 45, point.y + 30, { steps: 6 });
  await expect(connectionLine).toHaveCount(0);
  await page.mouse.up();
  expect((await openSaved()).revision).toBe(initial.revision);

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
  await enterEquations();
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
  await enterEquations();
  await expect(savedPath).toHaveAttribute('d', /^M/);
  expect(stored(await openSaved())).toEqual({ [id]: connection });
});

for (const { scope, element, composites, target } of [
  { scope: 'mechanics-port', element: { kind: 'port', end: { owner: 'boundary', port: 'mount' } },
    composites: ['arm'], target: (model: Locator) => model.locator('[data-boundary="true"] [data-handleid="physical:mount"]') },
  { scope: 'within-step', element: { kind: 'wire', id: 'drive' }, composites: [],
    target: (model: Locator) => edge(model, 'drive') },
]) {
  test(`reveals the saved Plan problem at its ${element.kind} address (${scope})`, async ({ page, request }, info) => {
    const { snapshot, projectPath } = await seedManagedProject(request, []);
    await runProjectFixture('web.e2e.studio.fixtures.stored_model', info.config.metadata.home, snapshot.project_id, scope);
    const plan = page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/train-runs` &&
      new URL(response.url()).searchParams.get('window_map') === 'true');
    await revealModel(page, `/?project=${encodeURIComponent(snapshot.project_id)}`);
    const response = await plan;
    expect(response.ok(), await response.text()).toBe(true);
    const { window_map: map } = await response.json() as StudioSavedTrainRuns;
    expect(map?.rows.find((row) => row.key.step === 'train')?.status).toBe('cannot_run');
    const address = { document: 'reach-model', composites, element };
    const problem = map?.problems.find((candidate) => candidate.step === 'train');
    expect(problem?.model_address).toMatchObject(address);
    expect(problem?.detail).toBeTruthy();
    if (!problem?.detail) throw new Error('Expected an addressed Plan problem');
    const reason = problem.detail.startsWith(problem.path + ': ') ? problem.detail.slice(problem.path.length + 2) : problem.detail;
    await page.getByRole('article', { name: 'Train train', exact: true })
      .getByRole('button', { name: /Can't run/ }).and(page.locator('[aria-haspopup="dialog"]')).click();
    await page.getByRole('dialog', { name: 'Problems' }).getByRole('button').filter({ hasText: reason }).click();
    await expect(target(modelPanel(page))).toHaveCount(1);
    await expect(modelPanel(page).getByText(reason, { exact: false }).first()).toBeFocused();
  });
}
