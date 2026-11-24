import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import { addFromLibrary as add, edited, newModel } from './fixtures/newModel';

test.use({ viewport: { width: 1440, height: 900 } });

type Point = { x: number; y: number };

async function wire(page: Page, from: Locator, to: Locator) {
  await edited(page, async () => {
    await from.hover();
    await page.mouse.down();
    await to.hover();
    await page.mouse.up();
  });
}

/** Points along the drawn path of the wire whose label contains `label`, in page coordinates. */
async function route(model: Locator, label: string) {
  const path = model.locator(`.react-flow__edge[aria-label*="${label}"] path.react-flow__edge-path`);
  await expect(path).toHaveCount(1);
  return path.evaluate((element: SVGPathElement) => {
    const length = element.getTotalLength();
    const matrix = element.getScreenCTM()!;
    return Array.from({ length: 81 }, (_, index) => {
      const point = element.getPointAtLength((length * index) / 80).matrixTransform(matrix);
      return { x: point.x, y: point.y };
    });
  });
}

/** Whether `point` lies inside `card`'s drawn box, ignoring a margin along its edges where the ports sit. */
async function inside(card: Locator) {
  const box = (await card.locator('.studio-model-card').boundingBox())!;
  return ({ x, y }: Point) => x > box.x + 4 && x < box.x + box.width - 4 && y > box.y + 4 && y < box.y + box.height - 4;
}

function crosses(a: readonly Point[], b: readonly Point[]) {
  const side = (p: Point, q: Point, r: Point) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
  for (let i = 1; i < a.length; i += 1) {
    for (let j = 1; j < b.length; j += 1) {
      const [p, q, r, s] = [a[i - 1], a[i], b[j - 1], b[j]];
      if (side(p, q, r) * side(p, q, s) < 0 && side(r, s, p) * side(r, s, q) < 0) return true;
    }
  }
  return false;
}

async function wiredPair(page: Page) {
  const model = await newModel(page);
  const first = await add(page, model, 'Linear');
  const second = await add(page, model, 'Linear');
  const handle = (card: Locator, id: string) => card.locator(`[data-handleid="${id}"]`);
  await wire(page, handle(first, 'output:output'), handle(second, 'input:input'));
  const passesBehind = async () => {
    const points = await route(model, '.output → ');
    const cards = [await inside(first), await inside(second)];
    return points.filter((point) => cards.some((within) => within(point)));
  };
  return { model, first, second, passesBehind };
}

/** A wire into a flipped card, whose input faces away from its source, runs around both cards, not through one. */
test('a wire into a flipped card passes behind no card', async ({ page }) => {
  const { second, passesBehind } = await wiredPair(page);
  await second.locator('header').click({ button: 'right' });
  await edited(page, () => page.getByRole('menuitem', { name: 'Flip', exact: true }).press('Enter'));
  await expect(second.locator('[data-handleid="input:input"]').locator('..')).toHaveAttribute('data-side', 'right');
  expect(await passesBehind()).toEqual([]);
});

/** A wire back to a card on its left runs under both cards, not behind them. */
test('a wire to a card on its source\'s left passes behind no card', async ({ page }) => {
  const { first, second, passesBehind } = await wiredPair(page);
  const [from, to] = [(await first.locator('header').boundingBox())!, (await second.locator('header').boundingBox())!];
  await edited(page, async () => {
    await page.mouse.move(to.x + 20, to.y + to.height / 2);
    await page.mouse.down();
    await page.mouse.move(from.x - 260, from.y + from.height / 2, { steps: 10 });
    await page.mouse.up();
  });
  expect(await passesBehind()).toEqual([]);
});

/** Two wires from a card back to itself nest: they do not cross, and the pair of lower ports takes the inner route. */
test('wires from a card back to itself nest with the lowest ports innermost', async ({ page }) => {
  const model = await newModel(page);
  const gru = await add(page, model, 'GRU');
  const handle = (id: string) => gru.locator(`[data-handleid="${id}"]`);
  await wire(page, handle('output:output'), handle('input:input'));
  await wire(page, handle('output:hidden'), handle('input:hidden'));
  const outer = await route(model, '.output → ');
  const inner = await route(model, '.hidden → ');
  const lowest = (points: readonly Point[]) => Math.max(...points.map(({ y }) => y));
  const [output, hidden] = [(await handle('output:output').boundingBox())!, (await handle('output:hidden').boundingBox())!];
  expect(hidden.y).toBeGreaterThan(output.y);
  expect(crosses(outer, inner)).toBe(false);
  expect(lowest(inner)).toBeLessThan(lowest(outer));
});
