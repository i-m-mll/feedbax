import { expect, test } from './fixtures/studio';
import { addFromLibrary, edited, newModel } from './fixtures/newModel';

test.use({ viewport: { width: 1440, height: 900 } });

/**
 * Inside a composite, the Inputs card's empty handle lists the signals outside, each told apart from components of the
 * same name; Escape dismisses the list, and choosing a signal adds the port.
 */
test('a composite\'s empty input handle lists the outside signals apart and brings one in', async ({ page }) => {
  const model = await newModel(page);
  const linears = [];
  for (let count = 0; count < 4; count += 1) linears.push(await addFromLibrary(page, model, 'Linear'));
  await linears[0].locator('header').click();
  await linears[1].locator('header').click({ modifiers: ['Shift'] });
  const { changed } = await edited(page, () => page.keyboard.press('Meta+G'));
  await model.locator(`.react-flow__node[data-id="${changed.at(-1)!.element.id}"] header`).dblclick();
  const inputs = model.locator('.react-flow__node[data-id="@inputs"]');
  const handles = inputs.locator('.react-flow__handle[data-model-port]');
  const ports = await handles.count();
  const list = page.getByRole('menu', { name: 'Add Port', exact: true });

  await inputs.getByRole('button', { name: 'Add Port', exact: true }).click();
  const signals = list.getByRole('menuitem');
  const names = await signals.allTextContents();
  expect(names.filter((name) => name.startsWith('Linear.output'))).toHaveLength(2);
  expect(new Set(names).size).toBe(names.length);
  await page.keyboard.press('Escape');
  await expect(list).toBeHidden();

  await inputs.getByRole('button', { name: 'Add Port', exact: true }).click();
  await edited(page, () => list.getByRole('menuitem', { name: names.find((name) => name.startsWith('Linear.output')), exact: true }).click());
  await expect(list).toBeHidden();
  await expect(handles).toHaveCount(ports + 1);
});

/** Wiring from the root Inputs card's empty handle adds an input named after the port it feeds, wired to it. */
test('a wire from the root Inputs card\'s empty handle adds the input it feeds', async ({ page }) => {
  const model = await newModel(page);
  const linear = await addFromLibrary(page, model, 'Linear');
  const inputs = model.locator('.react-flow__node[data-id="@inputs"]');
  const ports = inputs.locator('.react-flow__handle[data-model-port]');
  const before = await ports.count();
  await edited(page, async () => {
    await inputs.locator('[data-handleid="output:__new_port__"]').hover();
    await page.mouse.down();
    await linear.locator('[data-handleid="input:input"]').hover();
    await page.mouse.up();
  });
  await expect(ports).toHaveCount(before + 1);
  await expect(inputs.getByRole('button', { name: 'Port input', exact: true })).toBeVisible();
});

/** Wiring to the Outputs card's empty handle adds an output named after the wire's source, wired from it. */
test('a wire to the Outputs card\'s empty handle adds the output it carries', async ({ page }) => {
  const model = await newModel(page);
  const linear = await addFromLibrary(page, model, 'Linear');
  const outputs = model.locator('.react-flow__node[data-id="@outputs"]');
  await edited(page, async () => {
    await linear.locator('[data-handleid="output:output"]').hover();
    await page.mouse.down();
    await outputs.locator('[data-handleid="input:__new_port__"]').hover();
    await page.mouse.up();
  });
  await expect(outputs.getByRole('button', { name: 'Port output', exact: true })).toBeVisible();
});

/** Ungroup is dimmed, with the reason as its tip, while the composite's port carries an attachment, and applies once it does not. */
test('Ungroup is dimmed with its reason while a composite port has an attachment', async ({ page }) => {
  const model = await newModel(page);
  const source = await addFromLibrary(page, model, 'Linear');
  const grouped = await addFromLibrary(page, model, 'Linear');
  await edited(page, async () => {
    await source.locator('[data-handleid="output:output"]').hover();
    await page.mouse.down();
    await grouped.locator('[data-handleid="input:input"]').hover();
    await page.mouse.up();
  });
  await grouped.locator('header').click();
  const { changed } = await edited(page, () => page.keyboard.press('Meta+G'));
  const group = model.locator(`.react-flow__node[data-id="${changed.at(-1)!.element.id}"]`);
  await group.locator('.react-flow__handle[data-model-port][data-handleid^="input:"]').first().click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  await edited(page, () => page.getByRole('menuitem', { name: 'Add Probe', exact: true }).click());

  const menu = page.getByRole('menu', { name: 'Model Commands', exact: true });
  const ungroup = menu.getByRole('menuitem', { name: 'Ungroup', exact: true });
  await group.locator('header').click({ button: 'right' });
  await expect(ungroup).toBeDisabled();
  await expect(ungroup).toHaveAccessibleDescription(/attachment/i);
  await page.keyboard.press('Escape');

  await model.focus();
  const undone = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/model/undo'));
  await page.keyboard.press('Meta+Z');
  expect((await undone).ok()).toBe(true);
  await group.locator('header').click({ button: 'right' });
  await expect(ungroup).toBeEnabled();
  await edited(page, () => ungroup.click());
  await expect(group).toHaveCount(0);
});

/** A template in the Library's Templates group places one composite card, which can be entered. */
test('a Library template places one composite', async ({ page }) => {
  const model = await newModel(page);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('CDEStandard');
  const cards = model.locator('.react-flow__node.draggable');
  const before = await cards.count();
  await edited(page, () => drawer.getByRole('listbox', { name: 'Templates', exact: true })
    .getByRole('option', { name: 'CDEStandard', exact: true }).dblclick());
  await expect(cards).toHaveCount(before + 1);
  const placed = cards.filter({ has: page.locator('.studio-model-card[data-stacked="true"]') });
  await expect(placed).toHaveCount(1);
  await placed.locator('header').dblclick();
  await expect(model.getByRole('navigation', { name: 'Model Path' })).toBeVisible();
});

/**
 * A Library template dropped on a wire is placed on it as one composite, wired in between the wire's two ends; the drag
 * draws that before the drop (catches a drag that shows nothing until release).
 */
test('a Library template dropped on a wire is inserted into it', async ({ page }) => {
  const model = await newModel(page);
  const linear = await addFromLibrary(page, model, 'Linear');
  const outputs = model.locator('.react-flow__node[data-id="@outputs"]');
  await edited(page, async () => {
    await linear.locator('[data-handleid="output:output"]').hover();
    await page.mouse.down();
    await outputs.locator('[data-handleid="input:__new_port__"]').hover();
    await page.mouse.up();
  });
  const wires = model.locator('.react-flow__edge');
  await expect(wires).toHaveCount(1);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('CDEStandard');
  const template = drawer.getByRole('listbox', { name: 'Templates', exact: true }).getByRole('option', { name: 'CDEStandard', exact: true });
  const saved = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/model/edit'));
  // While it is dragged over the wire, the card is drawn where it will land, with the wire drawn into and out of it.
  const box = (await wires.first().locator('path.react-flow__edge-path').boundingBox())!;
  await template.hover();
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 6 });
  await expect(model.locator('.studio-model-card[data-stacked="true"]')).toHaveCount(1);
  await expect(model.locator('.react-flow__edge.studio-carried')).toHaveCount(2);
  await page.mouse.up();
  const response = await saved;
  expect(response.request().postDataJSON().command.operation).toMatchObject({
    op: 'insert_component', component: { kind: 'definition', definition: 'feedbax.CDEStandard' },
  });
  const { result } = await response.json();
  expect(result.status, result.problem?.detail).toBe('applied');
  await expect(model.locator('.studio-model-card[data-stacked="true"]')).toHaveCount(1);
  await expect(wires).toHaveCount(2);
});

/** A template the Service refuses to place is dimmed with its reason, and placing it from the Library sends nothing. */
test('a refused Library template is dimmed with its reason and places nothing', async ({ page }) => {
  const reason = 'Its recurrent history cannot be stored.';
  await page.route('**/api/components', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const component of body.data.components) {
      if (component.name === 'feedbax.CDEDecay') component.placement_reason = reason;
    }
    await route.fulfill({ response, json: body });
  });
  const edits: string[] = [];
  page.on('request', (request) => { if (request.url().endsWith('/model/edit')) edits.push(request.url()); });
  const model = await newModel(page);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill('CDE');
  const templates = drawer.getByRole('listbox', { name: 'Templates', exact: true });
  const refused = templates.getByRole('option', { name: 'CDEDecay', exact: true });
  await expect(refused).toHaveAttribute('aria-disabled', 'true');
  await expect(refused).toHaveAccessibleDescription(reason);
  await refused.hover();
  await expect(page.getByRole('status').filter({ hasText: reason })).toBeVisible();
  await expect(templates.getByRole('option', { name: 'CDEStandard', exact: true })).not.toHaveAttribute('aria-disabled', 'true');
  const cards = await model.locator('.react-flow__node').count();
  await refused.dblclick({ force: true });
  await refused.press('Enter');
  await expect(model.locator('.react-flow__node')).toHaveCount(cards);
  expect(edits).toEqual([]);
});
