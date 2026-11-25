import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { trialPicker } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const scene = (page: Page) => page.getByRole('tabpanel', { name: 'Scene', exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

const centre = async (target: Locator) => {
  const box = (await target.boundingBox())!;
  return [box.x + box.width / 2, box.y + box.height / 2] as const;
};

// Catches a Relate that writes a term the Scene then cannot draw: a term whose relation does not end at the hand and
// the target, a write that adds no row or more than one, a new term that is not selected with its Measure focused,
// and a second relation of the same two places drawn exactly over the first.
test('Relate dragged from the hand to the target adds one Objective term and its dashed line', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await tab(page, 'Objective').click();
  const table = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Objective Terms', exact: true });
  // A header, the fixture's one term, and the foot that adds a term.
  await expect(table.getByRole('row')).toHaveCount(3);
  await tab(page, 'Scene').click();
  const canvas = scene(page).getByRole('group', { name: 'Scene', exact: true });
  await expect(canvas).toBeVisible();
  // The nominal trial's target is at the start; a sample's is not.
  await trialPicker(page).click();
  await page.getByRole('radio', { name: /^sample 2 of \d+$/ }).click();
  await expect(trialPicker(page)).toHaveText(/^sample 2 of \d+$/);

  await scene(page).getByRole('toolbar').getByRole('button', { name: 'Relate', exact: true }).click();
  const hand = canvas.getByRole('button', { name: 'Relate Point mass', exact: true });
  const target = canvas.getByRole('button', { name: 'Relate target', exact: true });
  const start = canvas.getByRole('button', { name: 'Relate start', exact: true });
  const at = async (end: Locator) => end.locator('circle').first().evaluate((circle: SVGCircleElement) =>
    [circle.cx.baseVal.value, circle.cy.baseVal.value]);
  // The Model is posed by the picked trial once its geometry is read: at the start, apart from the target.
  await expect.poll(async () => {
    const [handAt, startAt, targetAt] = await Promise.all([hand, start, target].map(at));
    return JSON.stringify(handAt) === JSON.stringify(startAt) && JSON.stringify(targetAt) !== JSON.stringify(startAt);
  }).toBe(true);
  const lines = canvas.locator('[data-scene-relation]');
  const drawn = await lines.count();
  const [fromX, fromY] = await centre(hand.locator('circle').first());
  const [toX, toY] = await centre(target.locator('circle').first());
  await page.mouse.move(fromX, fromY);
  await page.mouse.down();
  await page.mouse.move(toX, toY, { steps: 8 });
  await expect(canvas.locator('[data-scene-candidate]')).not.toHaveCount(0);
  await page.mouse.up();

  await expect(lines).toHaveCount(drawn + 1);
  const line = lines.last();
  const ends = await line.evaluate((element: SVGLineElement) =>
    [[element.x1.baseVal.value, element.y1.baseVal.value], [element.x2.baseVal.value, element.y2.baseVal.value]]);
  expect(ends).toEqual(expect.arrayContaining([await at(hand), await at(target)]));
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('textbox', { name: 'Measure', exact: true })).toBeFocused();


  // A second relation of the same two places is drawn beside the first, not over it.
  await page.mouse.move(fromX, fromY);
  await page.mouse.down();
  await page.mouse.move(toX, toY, { steps: 8 });
  await page.mouse.up();
  await expect(lines).toHaveCount(drawn + 2);
  const segments = await lines.evaluateAll((elements) => (elements as SVGLineElement[]).slice(-2).map((element) =>
    [element.x1.baseVal.value, element.y1.baseVal.value, element.x2.baseVal.value, element.y2.baseVal.value]));
  const apart = Math.max(...segments[0].map((value, index) => Math.abs(value - segments[1][index])));
  expect(apart).toBeGreaterThan(3);

  await tab(page, 'Objective').click();
  await expect(table.getByRole('row')).toHaveCount(5);
});
