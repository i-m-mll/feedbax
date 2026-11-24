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
