import { expect, test, type Locator } from '@playwright/test';

test.use({ trace: 'off' });

/** The least widths the gallery's Runs table declares, in rem. */
const least = [4, 5, 7, 6];

async function box(table: Locator) {
  return table.evaluate((element) => {
    const scroller = element.parentElement!;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    return {
      overflows: scroller.scrollWidth > scroller.clientWidth + 1,
      columns: [...(element as HTMLTableElement).rows[0].cells].filter((cell) => !cell.matches('.studio-table-margin'))
        .map((cell) => cell.getBoundingClientRect().width / rem),
    };
  });
}

/**
 * Catches a table squashed below its columns' least widths, the horizontal scroll applied to the view instead of the
 * table that needs it, and a row name that scrolls away with the other columns.
 */
test('a table narrower than its least width scrolls in its own box with its name column held, and one that fits does not', async ({ page }) => {
  await page.goto('/gallery.html');
  const narrow = page.getByRole('table', { name: 'Runs, narrow' });
  const wide = page.getByRole('table', { name: 'Runs, wide' });
  await narrow.scrollIntoViewIfNeeded();

  const squeezed = await box(narrow);
  expect(squeezed.overflows).toBe(true);
  squeezed.columns.forEach((width, index) => expect(width).toBeGreaterThanOrEqual(least[index] - 0.01));
  expect(await page.evaluate(() => document.scrollingElement!.scrollWidth <= window.innerWidth)).toBe(true);
  expect((await box(wide)).overflows).toBe(false);

  const name = narrow.getByRole('rowheader').first();
  const loss = narrow.getByRole('cell', { name: '0.412' });
  const [nameAt, lossAt] = [(await name.boundingBox())!.x, (await loss.boundingBox())!.x];
  await narrow.evaluate((element) => { element.parentElement!.scrollLeft = 1000; });
  await expect.poll(async () => (await loss.boundingBox())!.x).toBeLessThan(lossAt);
  expect((await name.boundingBox())!.x).toBe(nameAt);
});

/** Catches a field in a cell that sits higher or lower than the text in the cells beside it. */
test('a cell holding a field and a cell holding text centre their content on the same line', async ({ page }) => {
  await page.goto('/gallery.html');
  const row = page.getByRole('table', { name: 'Runs, wide' }).getByRole('row').nth(1);
  const centre = (target: Locator) => target.evaluate((element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const { top, height } = element instanceof HTMLInputElement ? element.getBoundingClientRect() : range.getBoundingClientRect();
    return top + height / 2;
  });
  const text = await centre(row.getByRole('cell', { name: '0.412' }));
  expect(Math.abs(await centre(row.getByRole('textbox', { name: /Learning rate/ })) - text)).toBeLessThanOrEqual(1);
  expect(Math.abs(await centre(row.getByRole('textbox', { name: /Rename/ })) - text)).toBeLessThanOrEqual(1);
});

/** Catches a column stretched to fill its box instead of fitting its content. */
test('a table of short names is as wide as its content, its name column narrower than its greatest width', async ({ page }) => {
  await page.goto('/gallery.html');
  const wide = page.getByRole('table', { name: 'Runs, wide' });
  await wide.scrollIntoViewIfNeeded();
  const measured = await wide.evaluate((element) => {
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    return {
      table: element.getBoundingClientRect().width,
      box: element.parentElement!.clientWidth,
      name: element.querySelector('thead th:not(.studio-table-margin)')!.getBoundingClientRect().width / rem,
    };
  });
  expect(measured.table).toBeLessThan(measured.box - 1);
  expect(measured.name).toBeLessThan(12);
});

/** Catches a reorder that is drawn only once the drag is released, and a row that jumps back before the move lands. */
test('while a row is held over the next row\'s place, that row is already where the dragged row was', async ({ page }) => {
  await page.goto('/gallery.html');
  const table = page.getByRole('table', { name: 'Runs, wide' });
  await table.scrollIntoViewIfNeeded();
  const row = (name: string) => table.getByRole('row').filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });
  const top = async (name: string) => (await row(name).boundingBox())!.y;
  const [first, second] = [await top('run-1'), await top('run-2')];

  await row('run-1').hover();
  const handle = (await table.getByRole('img', { name: 'Move run-1', exact: true }).boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2 + (second - first) * 0.75, { steps: 8 });
  await expect.poll(() => top('run-2')).toBeCloseTo(first, 0);

  await page.mouse.up();
  await expect.poll(() => top('run-1')).toBeCloseTo(second, 0);
  await expect(table.getByRole('textbox', { name: /^Rename / }).first()).toHaveAccessibleName('Rename run-2');
  expect(await top('run-1')).toBeCloseTo(second, 0);
  expect(await top('run-2')).toBeCloseTo(first, 0);
});
