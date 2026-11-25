import { expect, test } from './fixtures/studio';
import type { ModelDocument, ModelEditResponse } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { expectVisibleFocus } from './fixtures/studioAccessibility';
import { rootPayload } from './fixtures/rootPayload';
import { palette } from './fixtures/studioKeyboard';

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
  await expectVisibleFocus(source);
  await expect(source).toHaveRole('button');
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
  // Arrows visit every offered input, those whose type differs included, in the canvas order.
  for (let hop = 0; hop < 8 && !await input.evaluate((handle) => handle === document.activeElement); hop++) {
    await page.keyboard.press('ArrowRight');
  }
  await expectVisibleFocus(input);
  await expect(input).toHaveRole('button');
  await expect(input).toHaveAccessibleDescription('Input · real [2]');
  const edit = post('edit');
  await page.keyboard.press('Space');
  const applied = await edit;
  const { result } = await applied.json() as ModelEditResponse;
  expect(applied.ok(), JSON.stringify(result)).toBe(true);
  if (result.status !== 'applied') throw new Error(JSON.stringify(result));
  const saved = rootPayload(result.snapshot.project.documents![modelFile.id]) as unknown as ModelDocument;
  const taskPort = saved.ports.inputs?.find(({ name }) => name === signal);
  expect(taskPort).toBeDefined();
  expect(Object.values(saved.interior.wires ?? {})).toContainEqual(expect.objectContaining({
    source: { owner: 'inputs', port: taskPort?.id }, target: firstWire.target,
  }));
});

test('a wire drawn to an occupied input of another type replaces its wire, keeping its attachment and waypoint', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const modelFile = files.find((file: { id: string }) => file.id === 'point-mass');
  const definition = modelFile.document.payload as ModelDocument;
  // The Model's output takes seconds, so a wire to it from the metre input differs in type.
  definition.ports.outputs![0].type = { shape: [2], unit: 's' };
  const drive = definition.interior.wires!.command;
  const original = drive.source;
  drive.attachments = [{ id: 'gain', kind: 'intervention', operation: 'scale', operands: { factor: { kind: 'value', value: 2 } } }];
  const seeded = await seedManagedProject(request, files);
  const { projectPath } = seeded;
  const wire = { composites: [], element: { kind: 'wire', id: 'command' } } as const;
  const waypoint = { x: 600, y: 300 };
  const routed = await request.post(`${projectPath}/model/edit`, { data: {
    revision: seeded.snapshot.revision, presentation_revision: seeded.snapshot.presentation_revision, document: modelFile.id,
    command: { operation: { op: 'route_wire', address: wire, waypoints: [waypoint] } },
  } });
  expect(routed.ok(), await routed.text()).toBe(true);
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const source = model.locator('[data-id="@inputs"] [data-handleid="output:target_pos"]');
  const force = model.locator('[data-id="@outputs"] [data-handleid="input:command"]');
  const edge = model.locator('.react-flow__edge[data-id="command"]');
  const post = (operation: 'connections' | 'edit' | 'undo') => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/${operation}`);
  const storedDrive = (response: ModelEditResponse) => {
    if (response.result.status !== 'applied') throw new Error(JSON.stringify(response.result));
    const snapshot = response.result.snapshot;
    const saved = rootPayload(snapshot.project.documents![modelFile.id]) as unknown as ModelDocument;
    const key = Object.keys(snapshot.presentation?.documents?.[modelFile.id]?.waypoints ?? {})
      .find((address) => address.includes('"command"'));
    return { wire: saved.interior.wires?.command, waypoints: key && snapshot.presentation?.documents?.[modelFile.id]?.waypoints?.[key] };
  };

  await page.goto(`/?project=${seeded.snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(source).toHaveAttribute('aria-disabled', 'false');

  // The keyboard offers the occupied input of another type, marked, and previews the wire it keeps.
  await source.focus();
  const reviewed = post('connections');
  await page.keyboard.press('Enter');
  expect((await reviewed).ok()).toBe(true);
  await expect(force.locator('..')).toHaveAttribute('data-verdict', 'marked');
  await force.focus();
  await expect(edge).toHaveAttribute('data-preview', 'kept');
  await page.keyboard.press('Escape');
  await expect(edge).not.toHaveAttribute('data-preview');

  // The pointer's drop on the same input stores the replacement: the wire's identity, attachment and waypoint stay.
  await source.hover();
  const dragged = post('connections');
  await page.mouse.down();
  await force.hover();
  expect((await dragged).ok()).toBe(true);
  await expect(force.locator('..')).toHaveAttribute('data-verdict', 'marked');
  await expect(edge).toHaveAttribute('data-preview', 'kept');
  await expect(edge.locator('.react-flow__edge-path')).toHaveAttribute('d', new RegExp(`${waypoint.x},${waypoint.y}`));
  await expect(model.getByRole('button', { name: /^Attachment / })).toHaveAccessibleDescription(/Inputs\.target_pos →/);
  const edited = post('edit');
  await page.mouse.up();
  const replaced = storedDrive(await (await edited).json() as ModelEditResponse);
  expect(replaced.wire).toMatchObject({ source: { owner: 'inputs', port: 'target_pos' }, target: drive.target, attachments: drive.attachments });
  expect(replaced.waypoints).toEqual([waypoint]);

  await model.focus();
  const undone = post('undo');
  await page.keyboard.press('Meta+Z');
  const restored = storedDrive(await (await undone).json() as ModelEditResponse);
  expect(restored.wire).toMatchObject({ source: original, attachments: drive.attachments });
  expect(restored.waypoints).toEqual([waypoint]);
  const redone = post('undo');
  await page.keyboard.press('Meta+Shift+Z');
  expect(storedDrive(await (await redone).json() as ModelEditResponse).wire?.source).toEqual({ owner: 'inputs', port: 'target_pos' });

  // The mismatch is stored: after a reload the wire carries its Problem.
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(model.locator('.studio-model-attachments').getByRole('img', { name: 'Problem', exact: true }))
    .toHaveAccessibleDescription(/expected unit s, received m/);
  // Problems lists each Problem the canvas marks.
  const marked = await model.locator('[role="img"][aria-label="Problem"][aria-description]')
    .evaluateAll((marks) => marks.map((mark) => mark.getAttribute('aria-description')!));
  expect(marked.length).toBeGreaterThan(0);
  await palette(page, 'problems', 'Problems');
  const listed = page.getByRole('dialog', { name: 'Problems', exact: true }).getByRole('button');
  for (const reason of new Set(marked)) await expect(listed.filter({ hasText: reason }).first()).toBeVisible();
});
