import { expect, test } from './fixtures/studio';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1400, height: 900 } });

// Catches a drawer list whose height cuts through a row (a list a few pixels short of its rows that scrolls a little
// and fades its last row), and a window whose map and shelf leave the Task view too short to show the grid under the
// Variables table; and a list that hides rows marking it by fading the rows it shows, or not marking it at all.
test('the Task tab shows whole rows in its Outline and the grid under the Variables table', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const primary = page.locator('[data-slot="primary"]');
  const drawer = page.locator('[data-studio-focus-region="drawer"]');

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Task Document / }).click();
  await page.getByRole('menu', { name: 'Task Document', exact: true }).getByRole('menuitem', { name: 'New', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Delayed reaches', exact: true }).click();
  await page.keyboard.press('Escape');
  const variables = drawer.getByRole('tree', { name: 'Variables', exact: true });
  await expect(variables.getByRole('treeitem')).toHaveCount(3);
  await expect(primary.getByRole('table', { name: 'Epochs', exact: true })).toBeVisible();

  const bodies = () => drawer.locator('[data-section="drawer"][data-open] > [data-section-body]').evaluateAll((nodes) =>
    nodes.map((body) => {
      const box = body.getBoundingClientRect();
      const rows = [...body.querySelectorAll('[data-list-row]')].map((row) => row.getBoundingClientRect())
        .filter(({ top, bottom }) => bottom > box.top + 0.5 && top < box.bottom - 0.5);
      return {
        name: body.closest('section')!.querySelector('[data-section-header]')!.textContent,
        scrolls: body.scrollHeight - body.clientHeight > 0.5,
        faded: body.hasAttribute('data-overflow-bottom'),
        cut: rows.some(({ bottom }) => bottom > box.bottom + 0.5),
        fade: getComputedStyle(body).getPropertyValue('--fade-bottom').trim(),
        shade: getComputedStyle(body).boxShadow !== 'none',
      };
    }));
  await expect.poll(async () => (await bodies()).find(({ name }) => name === 'Variables'))
    .toMatchObject({ name: 'Variables', scrolls: false, faded: false, cut: false, shade: false });
  for (const body of await bodies()) expect(body, body.name ?? '').toMatchObject({ cut: false, fade: '0px', shade: body.faded });
  expect((await bodies()).find(({ name }) => name === 'Epochs')).toMatchObject({ faded: true, shade: true });

  const view = await primary.locator('.studio-window-slot__body').boundingBox();
  const grid = await primary.getByRole('table', { name: 'Epochs', exact: true }).boundingBox();
  expect(grid!.y).toBeLessThan(view!.y + view!.height);
});
