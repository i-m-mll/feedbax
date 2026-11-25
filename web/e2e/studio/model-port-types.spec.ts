import { expect, test } from './fixtures/studio';
import type { Locator } from '@playwright/test';
import type { ModelDocument, ModelEditResponse, StudioTrialAuthoringRequest } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const opacity = (handle: Locator) => handle.evaluate((element) => Number(getComputedStyle(element).opacity));

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
  await expect(command.locator('..').locator('.studio-model-port-type')).toHaveText('s');

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
  await expect(port.getByRole('img', { name: 'Problems', exact: true })).toHaveAccessibleDescription('The Objective reads no signal here.');
  expect(relates.map(({ change }) => change)).toEqual([{
    kind: 'relate', first: { kind: 'port', address: { composites: [], element: { kind: 'port', end: { owner: 'outputs', port: 'command' } } } },
  }]);
});
