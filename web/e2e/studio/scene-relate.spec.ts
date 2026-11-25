import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const scene = (page: Page) => page.getByRole('tabpanel', { name: 'Scene', exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

const centre = async (target: Locator) => {
  const box = (await target.boundingBox())!;
  return [box.x + box.width / 2, box.y + box.height / 2] as const;
};

// Catches a Relate that writes a term the Scene then cannot draw: a term whose relation does not end at the hand and
// the target, a write that adds no row or more than one, and a new term that is not selected with its Measure focused.
test('Relate dragged from the hand to the target adds one Objective term and its dashed line', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  const table = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Objective Terms', exact: true });
  const rows = await table.getByRole('row').count();
  await tab(page, 'Scene').click();
  const canvas = scene(page).getByRole('group', { name: 'Scene', exact: true });
  await expect(canvas).toBeVisible();
  const lines = canvas.locator('[data-scene-relation]');
  const drawn = await lines.count();

  await scene(page).getByRole('toolbar').getByRole('button', { name: 'Relate', exact: true }).click();
  const hand = canvas.getByRole('button', { name: 'Relate Point mass', exact: true });
  const target = canvas.getByRole('button', { name: 'Relate target', exact: true });
  const [fromX, fromY] = await centre(hand);
  const [toX, toY] = await centre(target);
  await page.mouse.move(fromX, fromY);
  await page.mouse.down();
  await page.mouse.move(toX, toY, { steps: 8 });
  await expect(canvas.locator('[data-scene-candidate]')).not.toHaveCount(0);
  await page.mouse.up();

  await expect(lines).toHaveCount(drawn + 1);
  const line = lines.last();
  const ends = await line.evaluate((element: SVGLineElement) =>
    [[element.x1.baseVal.value, element.y1.baseVal.value], [element.x2.baseVal.value, element.y2.baseVal.value]]);
  const at = async (end: Locator) => end.locator('circle').first().evaluate((circle: SVGCircleElement) =>
    [circle.cx.baseVal.value, circle.cy.baseVal.value]);
  expect(ends).toEqual(expect.arrayContaining([await at(hand), await at(target)]));
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('textbox', { name: 'Measure', exact: true })).toBeFocused();

  await tab(page, 'Objective').click();
  await expect(table.getByRole('row')).toHaveCount(rows + 1);
});
