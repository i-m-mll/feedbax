import { expect, test } from './fixtures/studio';
import type { Locator } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, StudioTrialAuthoringRequest } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const opacity = (handle: Locator) => handle.evaluate((element) => Number(getComputedStyle(element).opacity));

// Catches a stored wire whose type mismatch is listed in Problems but shown nowhere on the canvas or the inspector, and
// a port's Problem mark whose text no hover shows.
test('a drag dims the input that states another type yet still stores a wire to it, and a port shows its type', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const modelFile = files.find((file: { id: string }) => file.id === 'point-mass');
  const definition = modelFile.document.payload as ModelDocument;
  // The Model's output takes seconds, so a wire to it from the metre input states a type that differs.
  definition.ports.outputs![0].type = { shape: [2], unit: 's' };
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const source = model.locator('[data-id="@inputs"] [data-handleid="output:target_pos"]');
  const command = model.locator('[data-id="@outputs"] [data-handleid="input:command"]');
  const post = (operation: 'connections' | 'edit') => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/${operation}`);

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(source).toHaveAttribute('aria-disabled', 'false');
  await expect(command.locator('..').locator('.studio-model-port-type')).toHaveText('[2] s');
  // A port whose type has no unit still shows its type.
  await expect(model.locator('[data-id="readout"] [data-handleid="output:output"]').locator('..').locator('.studio-model-port-type'))
    .toHaveText('[2]');

  await source.hover();
  const reviewed = post('connections');
  await page.mouse.down();
  await command.hover();
  expect((await reviewed).ok()).toBe(true);
  await expect(command.locator('..')).toHaveAttribute('data-verdict', 'marked');
  const offered = model.locator('.studio-model-port[data-verdict="allowed"] .react-flow__handle');
  await expect(offered.first()).toBeVisible();
  expect(await opacity(command)).toBeLessThan(0.5);
  expect(await opacity(offered.first())).toBe(1);

  const edited = post('edit');
  await page.mouse.up();
  const { result } = await (await edited).json() as ModelEditResponse;
  if (result.status !== 'applied') throw new Error(JSON.stringify(result));
  // The stored wire's Problem is on the input it does not fit, and on the wire's and that port's inspectors.
  const mark = command.locator('..').getByRole('img', { name: 'Problem', exact: true });
  await expect(mark).toHaveAccessibleDescription(/does not fit/);
  // The mark's hover tip shows its Problem, which is otherwise read only by a screen reader.
  await expect(mark).toHaveAttribute('data-app-tip', /does not fit/);
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await command.click();
  await expect(inspector.getByText(/does not fit/)).toBeVisible();
  await page.getByRole('treeitem', { name: /^Inputs\.target_pos → Outputs\.command/ }).click();
  await expect(inspector.getByText(/does not fit/)).toBeVisible();
});

test("a port's + asks for a term on that port, and a refusal lands on the port", async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const { snapshot } = await seedManagedProject(request, files);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const port = model.locator('[data-id="@outputs"] [data-handleid="input:command"]').locator('..');
  const relates: StudioTrialAuthoringRequest[] = [];
  await page.route('**/authoring/trials', async (route) => {
    const body = route.request().postDataJSON() as StudioTrialAuthoringRequest;
    if (body.change?.kind !== 'relate') return route.fallback();
    relates.push(body);
    return route.fulfill({ status: 422, json: { detail: 'The Objective reads no signal here.' } });
  });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await port.hover();
  await port.getByRole('button', { name: 'Add Term command', exact: true }).click();
  await expect(port.getByRole('img', { name: 'Problem', exact: true })).toHaveAccessibleDescription('The Objective reads no signal here.');
  await expect(model.getByRole('alert')).toHaveCount(0);
  expect(relates.map(({ change }) => change)).toEqual([{
    kind: 'relate', first: { kind: 'port', address: { composites: [], element: { kind: 'port', end: { owner: 'outputs', port: 'command' } } } },
  }]);
});

// Catches a + on a port whose value is a record of fields (the Point mass's effector) that asks for a term on the whole
// record, which no term can read, instead of on the field chosen.
test("a struct port's + offers its fields, and the one chosen is related", async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const { snapshot } = await seedManagedProject(request, files);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const port = model.locator('[data-id="plant"] [data-handleid="output:effector"]').locator('..');
  const related = page.waitForResponse((response) => response.request().method() === 'POST'
    && response.url().endsWith('/authoring/trials')
    && (response.request().postDataJSON() as StudioTrialAuthoringRequest).change?.kind === 'relate');

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(port.locator('.studio-model-port-type')).toHaveText('State');
  await port.hover();
  await port.getByRole('button', { name: 'Add Term effector', exact: true }).click();
  await page.getByRole('menu', { name: 'Add Term effector' }).getByRole('menuitem', { name: 'effector.pos', exact: true }).click();
  const response = await related;
  expect(response.ok(), await response.text()).toBe(true);
  const { change } = response.request().postDataJSON() as StudioTrialAuthoringRequest;
  expect(change).toMatchObject({ kind: 'relate', first: { kind: 'port', address: { element: { kind: 'field', path: ['pos'] } } } });
  await expect(port.getByRole('img', { name: 'Problem', exact: true })).toHaveCount(0);
});

// Catches the Inputs card reordering its task signals when one of them is wired and becomes a Model input.
test('wiring a task signal keeps the Inputs order', async ({ page, request }) => {
  const files = await fixture('delayed-reaches-files');
  const modelFile = files.find((file: { id: string }) => file.id === 'point-mass');
  const definition = modelFile.document.payload as ModelDocument;
  delete definition.interior.wires!.target;
  definition.ports.inputs = [];
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const inputs = model.locator('[data-id="@inputs"]');
  const names = () => inputs.locator('.studio-model-port > span:first-of-type').allTextContents();

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(inputs.locator('[data-handleid^="output:@task:"]').nth(2)).toBeVisible();
  const before = await names();
  const last = inputs.locator('[data-handleid^="output:@task:"]').last();
  const edited = page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/edit`);
  const from = (await last.boundingBox())!;
  const to = (await model.locator('[data-id="cell"] [data-handleid="input:input"]').boundingBox())!;
  await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
  await page.mouse.down();
  await page.mouse.move(to.x + to.width / 2, to.y + to.height / 2, { steps: 6 });
  await page.mouse.up();
  const { result } = await (await edited).json() as ModelEditResponse;
  if (result.status !== 'applied') throw new Error(JSON.stringify(result));
  await expect(inputs.locator('[data-handleid^="output:@task:"]')).toHaveCount(before.length - 2);
  expect(await names()).toEqual(before);
});
