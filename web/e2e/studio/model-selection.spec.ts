import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StoredModelDocument as ModelDocument, ModelEditResponse } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

async function openModel(page: Page, request: Parameters<typeof seedManagedProject>[0], change?: (model: ModelDocument) => void) {
  const files = await fixture('ordinary-files', '2');
  const saved = files.find((file: { id: string }) => file.id === 'point-mass').document.payload as ModelDocument;
  change?.(saved);
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.locator('.react-flow__node').first()).toBeVisible();
  // Cards and attachments take presses once the Model is editable, which Lock follows.
  await expect(model.getByRole('button', { name: 'Lock', exact: true })).toBeEnabled();
  const edit = () => page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/model/edit`);
  return { model, errors, edit, saved: Object.keys(saved.interior.components ?? {}).sort() };
}

test('a Shift-drag box selects the cards it covers', async ({ page, request }) => {
  const { model, errors } = await openModel(page, request);
  const cards = model.locator('.react-flow__node');
  const boxes = await cards.evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().toJSON() as DOMRect));
  const left = Math.min(...boxes.map((box) => box.left)) - 20;
  const top = Math.min(...boxes.map((box) => box.top)) - 20;
  const right = Math.max(...boxes.map((box) => box.right)) + 20;
  const bottom = Math.max(...boxes.map((box) => box.bottom)) + 20;
  await page.keyboard.down('Shift');
  await page.mouse.move(left, top);
  await page.mouse.down();
  await page.mouse.move(right, bottom, { steps: 12 });
  await page.mouse.up();
  await page.keyboard.up('Shift');
  await expect(model.locator('.react-flow__node.selected').first()).toBeVisible();
  await expect(model.getByText('Model failed to render')).toHaveCount(0);
  expect(errors).toEqual([]);
});

type Applied = Extract<ModelEditResponse['result'], { status: 'applied' }>;
const components = (result: ModelEditResponse['result']) =>
  Object.keys(((result as Applied).snapshot.project.documents!['point-mass'] as unknown as { payload: ModelDocument }).payload.interior.components ?? {}).sort();

test('Delete removes the wired cards Duplicate just made', async ({ page, request }) => {
  const { model, errors, edit, saved } = await openModel(page, request);
  await model.locator('.react-flow__node[data-id="cell"] header').click();
  await page.keyboard.press('Meta+A');
  const duplicated = edit();
  await page.keyboard.press('Meta+D');
  const duplicate = (await (await duplicated).json() as ModelEditResponse).result;
  expect(duplicate.status).toBe('applied');
  const removed = edit();
  await page.keyboard.press('Backspace');
  const remove = (await (await removed).json() as ModelEditResponse).result;
  expect(remove.status, remove.status === 'refused' ? remove.problem.detail : '').toBe('applied');
  expect(components(duplicate)).toHaveLength(2 * saved.length);
  expect(components(remove)).toEqual(saved);
  expect(errors).toEqual([]);
});

/** A Duplicate leaves its new cards selected, not the wires it copied between them (catches a copy that selects a wire). */
test('Duplicate selects the cards it made', async ({ page, request }) => {
  const { model, errors, edit, saved } = await openModel(page, request);
  await model.locator('.react-flow__node[data-id="cell"] header').click();
  await page.keyboard.press('Meta+A');
  const duplicated = edit();
  await page.keyboard.press('Meta+D');
  const duplicate = (await (await duplicated).json() as ModelEditResponse).result;
  expect(duplicate.status).toBe('applied');
  const made = components(duplicate).filter((id) => !saved.includes(id));
  await expect(model.locator('.react-flow__node.selected')).toHaveCount(made.length);
  for (const id of made) await expect(model.locator(`.react-flow__node[data-id="${id}"]`)).toHaveClass(/\bselected\b/);
  await expect(model.locator('.react-flow__edge.selected')).toHaveCount(0);
  expect(errors).toEqual([]);
});

/**
 * An attachment dragged onto another wire is drawn on that wire while it is dragged, and moved there on release (catches
 * an attachment that appears on its new wire only on drop). The `target` wire's middle lies under a card, and the
 * attachment is drawn where no card hides it (catches a wire label drawn under a card).
 */
test('an attachment dragged onto another wire is drawn there before it is dropped', async ({ page, request }) => {
  const { model, errors, edit } = await openModel(page, request, (document) => {
    document.interior.wires!.decode.attachments = { gain: { kind: 'intervention', operation: 'scale', operands: { factor: { kind: 'value', value: 2 } } } };
  });
  const attachment = model.getByRole('button', { name: 'Attachment Scale', exact: true });
  const path = model.locator('.react-flow__edge[data-id="target"] path.react-flow__edge-path');
  const target = (await path.boundingBox())!;
  // A point of the wire that no card or label covers.
  const on = await path.evaluate((element: SVGPathElement) => {
    const length = element.getTotalLength();
    for (let step = 1; step < 20; step += 1) {
      const point = element.getPointAtLength(length * step / 20).matrixTransform(element.getScreenCTM()!);
      if (document.elementFromPoint(point.x, point.y)?.closest('.react-flow__edge') === element.closest('.react-flow__edge')) {
        return { x: point.x, y: point.y };
      }
    }
    throw new Error('The wire is covered everywhere.');
  });
  await attachment.hover();
  await page.mouse.down();
  await page.mouse.move(on.x, on.y, { steps: 6 });
  await expect(attachment).toHaveCount(1);
  // The drawing follows the pointer on a later frame than the move.
  await expect.poll(async () => {
    const drawn = (await attachment.boundingBox())!;
    const middle = { x: drawn.x + drawn.width / 2, y: drawn.y + drawn.height / 2 };
    return middle.x > target.x - 20 && middle.x < target.x + target.width + 20
      && middle.y > target.y - 20 && middle.y < target.y + target.height + 20;
  }, { message: 'The attachment is drawn on the wire under the pointer' }).toBe(true);
  const uncovered = () => attachment.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2)?.closest('button') === element;
  });
  await expect.poll(uncovered, { message: 'No card covers the attachment while it is dragged' }).toBe(true);
  const moved = edit();
  await page.mouse.up();
  const request_ = (await moved).request().postDataJSON();
  expect(request_.command.operation).toMatchObject({
    op: 'move_wire_attachment', destination: { element: { kind: 'wire', id: 'target' } },
  });
  await expect.poll(uncovered, { message: 'No card covers the dropped attachment' }).toBe(true);
  expect(errors).toEqual([]);
});

/** Where the dashing of `wire` begins, on screen; undefined when the wire is solid. */
function dashStart(model: ReturnType<Page['locator']>, wire: string) {
  return model.locator(`.react-flow__edge[data-id="${wire}"] path.react-flow__edge-path`).evaluate((element: SVGPathElement) => {
    const along = Number.parseFloat(element.style.strokeDasharray);
    if (!Number.isFinite(along)) return undefined;
    const point = element.getPointAtLength(along).matrixTransform(element.getScreenCTM()!);
    return { x: point.x, y: point.y };
  });
}

/**
 * A delay is a square on its wire, with the wire solid up to it and dashed after it, and it drags to another wire of
 * its loop, which is then the one dashed (catches dashing that starts at the wire's middle while the square is drawn
 * elsewhere, a round delay mark, and dashing left on the wire the delay left).
 */
test('a delay is a square where its wire turns dashed, and drags to another wire of its loop', async ({ page, request }) => {
  const { model, errors, edit } = await openModel(page, request, (document) => {
    const { components, wires } = document.interior;
    for (const [id, field] of [['cell', 'hidden_size'], ['readout', 'input_size']] as const) {
      const leaf = components![id];
      if (leaf.kind === 'leaf') leaf.fields = { ...leaf.fields, [field]: 2 };
    }
    wires!.feedback = {
      source: { owner: 'component', component: 'readout', port: 'output' },
      target: { owner: 'component', component: 'cell', port: 'hidden' },
      attachments: { lag: { kind: 'delay', delay: { value: 5, unit: 'st' } } },
    };
  });
  const delay = model.getByRole('button', { name: 'Attachment Delay', exact: true });
  const centre = async () => {
    const box = (await delay.boundingBox())!;
    return { x: box.x + box.width / 2, y: box.y + box.height / 2, width: box.width, height: box.height };
  };
  const onSquare = async (wire: string) => {
    if (!await delay.isVisible()) return false;
    const [square, start] = [await centre(), await dashStart(model, wire)];
    return !!start && Math.hypot(square.x - start.x, square.y - start.y) < 6;
  };
  await expect(delay).toHaveText('5');
  const drawn = await centre();
  expect(Math.abs(drawn.width - drawn.height)).toBeLessThan(1);
  expect(await delay.evaluate((element) => getComputedStyle(element).borderTopLeftRadius)).toBe('0px');
  await expect.poll(() => onSquare('feedback'), { message: 'The feedback wire turns dashed at the square' }).toBe(true);
  expect(await dashStart(model, 'decode')).toBeUndefined();

  const path = model.locator('.react-flow__edge[data-id="decode"] path.react-flow__edge-path');
  const on = await path.evaluate((element: SVGPathElement) => {
    const length = element.getTotalLength();
    for (let step = 1; step < 20; step += 1) {
      const point = element.getPointAtLength(length * step / 20).matrixTransform(element.getScreenCTM()!);
      if (document.elementFromPoint(point.x, point.y)?.closest('.react-flow__edge') === element.closest('.react-flow__edge')) {
        return { x: point.x, y: point.y };
      }
    }
    throw new Error('The wire is covered everywhere.');
  });
  await delay.hover();
  await page.mouse.down();
  await page.mouse.move(on.x, on.y, { steps: 6 });
  const moved = edit();
  await page.mouse.up();
  const { result } = await (await moved).json() as ModelEditResponse;
  expect(result.status, result.status === 'refused' ? result.problem.detail : '').toBe('applied');
  await expect.poll(() => onSquare('decode'), { message: 'The decode wire turns dashed at the square' }).toBe(true);
  expect(await dashStart(model, 'feedback')).toBeUndefined();
  expect(errors).toEqual([]);
});

/** Add Hold on a wire stores a Hold there and draws it (catches a Hold that is not offered, or refused as sent). */
test('Add Hold puts a Hold on the selected wire', async ({ page, request }) => {
  const { model, errors, edit } = await openModel(page, request);
  await model.locator('.react-flow__edge[data-id="decode"]').click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  const added = edit();
  await page.getByRole('menuitem', { name: 'Add Hold', exact: true }).click();
  const response = await added;
  expect(response.request().postDataJSON().command.operation.difference.attachment).toMatchObject({ kind: 'hold' });
  const { result } = await response.json() as ModelEditResponse;
  expect(result.status, result.status === 'refused' ? result.problem.detail : '').toBe('applied');
  await expect(model.getByRole('button', { name: 'Attachment Hold', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});
