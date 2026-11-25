import { expect, test } from './fixtures/studio';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const target = { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: 'target_pos' }] };
const squared = { base: 'real', dimensions: { length: 2 } };

// Catches a Window cell whose interval is stored closed (21 samples), anchored at the wrong edge of its epoch, or with
// offsets the shelf does not apply: `[move + 0.1 s, move + 0.3 s)` at a Step of 0.01 s is 20 samples from move + 0.1 s.
test("a term's interval, authored in its Window cell, applies on the shelf over its half-open span", async ({ page, request }) => {
  test.setTimeout(60_000);
  const files = await fixture('delayed-reaches-files') as { document?: { payload: Record<string, any> } }[];
  for (const file of files) {
    const payload = file.document?.payload;
    if (payload?.terms) payload.terms.push({ id: 'pulse', name: 'pulse', measure: target, measure_type: squared });
  }
  const { snapshot } = await seedManagedProject(request, files);
  const clock = page.locator('footer.studio-window-bottom').getByRole('group', { name: 'Clock', exact: true });
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const row = (name: string) => shelf.getByRole('group', { name, exact: true });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await clock.getByRole('button', { name: /^Trial / }).click();
  await page.getByRole('radio', { name: 'nominal', exact: true }).click();
  await expect(row('pulse').locator('[data-start]')).not.toHaveCount(0);
  const move = Number(await row('Epochs').getByRole('button', { name: 'move', exact: true }).getAttribute('data-start'));

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const table = page.locator('[data-slot="primary"]').getByRole('tabpanel').getByRole('table', { name: 'Objective Terms', exact: true });
  const pulse = table.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename pulse' }) });
  await pulse.getByRole('button', { name: 'Window', exact: true }).click();
  const editor = page.getByRole('dialog', { name: 'Window', exact: true });
  await editor.getByRole('button', { name: 'Add Span', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Interval', exact: true }).click();

  const position = (name: 'Start' | 'End') => editor.getByRole('group', { name, exact: true });
  const choose = async (name: 'Start' | 'End', control: 'Anchor' | 'Edge', option: string) => {
    await position(name).getByRole('combobox', { name: `${control} ${name}`, exact: true }).click();
    await page.getByRole('option', { name: option, exact: true }).click();
    await expect(position(name).getByRole('combobox', { name: `${control} ${name}`, exact: true })).toHaveText(option);
  };
  await choose('Start', 'Anchor', 'move');
  await choose('End', 'Anchor', 'move');
  await choose('End', 'Edge', 'Start');
  await enter(position('Start').getByRole('textbox', { name: 'Start', exact: true }), '0.1 s');
  await enter(position('End').getByRole('textbox', { name: 'End', exact: true }), '0.3 s');
  await expect(pulse.getByRole('button', { name: 'Window', exact: true })).toHaveText('[move + 0.1 s, move + 0.3 s)');
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(editor).toBeHidden();

  await expect.poll(async () => row('pulse').locator('[data-start]').evaluateAll((bars) =>
    bars.map((bar) => [Number(bar.getAttribute('data-start')), Number(bar.getAttribute('data-end'))]))).toEqual([
    [expect.closeTo(move + 0.1, 6), expect.closeTo(move + 0.3, 6)],
  ]);
});
