import { expect, test, type Locator } from '@playwright/test';

test.use({ trace: 'off' });

/** The least widths the gallery's Runs table declares, in rem. */
const least = [7, 5, 7, 6];

async function box(table: Locator) {
  return table.evaluate((element) => {
    const scroller = element.parentElement!;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    return {
      overflows: scroller.scrollWidth > scroller.clientWidth + 1,
      columns: [...(element as HTMLTableElement).rows[0].cells].map((cell) => cell.getBoundingClientRect().width / rem),
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
