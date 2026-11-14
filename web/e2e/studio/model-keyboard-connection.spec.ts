import { expect, test } from '@playwright/test';
import type { ModelDocument, ModelEditResponse } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

test('connects the saved Model task signal from the keyboard', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const modelFile = files.find((file: { id: string }) => file.id === 'point-mass');
  const definition = modelFile.document.payload as ModelDocument;
  const firstWire = definition.interior.wires?.target;
  if (!firstWire) throw new Error('Expected the saved task wire in the managed fixture');
  const signal = firstWire.source.port;
  delete definition.interior.wires!.target;
  definition.ports.inputs = [];
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const source = model.locator(`[data-handleid="output:@task:${signal}"]`);
  const input = model.locator('[data-id="cell"] [data-handleid="input:input"]');
  const post = (operation: 'connections' | 'edit') => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/${operation}`);

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(model).toBeFocused();
  await expect(source).toBeVisible();
  await expect(source).toHaveAttribute('aria-disabled', 'false');

  await source.evaluate((handle) => {
    handle.addEventListener('focusout', () => handle.setAttribute('data-test-lost-focus', ''), { once: true });
  });
  for (let hop = 0; hop < 15; hop++) {
    if (await source.evaluate((handle) => handle === document.activeElement)) break;
    await page.keyboard.press('Tab');
    expect(await model.evaluate((panel) => panel.contains(document.activeElement))).toBe(true);
  }
  await expect(source).toBeFocused();
  await expect(source).toHaveAccessibleDescription(/^Output · /);
  const review = post('connections');
  await page.keyboard.press('Enter');
  const reviewed = await review;
  expect(reviewed.ok(), await reviewed.text()).toBe(true);
  expect(reviewed.request().postDataJSON()).toMatchObject({
    revision: snapshot.revision,
    source: { owner: 'task', signal, run: { step: 'train' } },
  });
  await expect(source).toHaveAttribute('aria-disabled', 'true');
  await expect(input).toHaveAttribute('aria-disabled', 'false');
  await source.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
  await expect(source).toBeFocused();
  await expect(source).not.toHaveAttribute('data-test-lost-focus', '');
  await page.keyboard.press('ArrowRight');
  await expect(input).toBeFocused();
  await expect(input).toHaveAccessibleDescription(/^Input · /);
  const edit = post('edit');
  await page.keyboard.press('Enter');
  const applied = await edit;
  const { result } = await applied.json() as ModelEditResponse;
  expect(applied.ok(), JSON.stringify(result)).toBe(true);
  if (result.status !== 'applied') throw new Error(result.problem.detail);
  const saved = result.snapshot.project.documents[modelFile.id].payload as unknown as ModelDocument;
  const taskPort = saved.ports.inputs?.find(({ name }) => name === signal);
  expect(taskPort).toBeDefined();
  expect(Object.values(saved.interior.wires ?? {})).toContainEqual(expect.objectContaining({
    source: { owner: 'inputs', port: taskPort?.id }, target: firstWire.target,
  }));
});
