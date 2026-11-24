import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioJobSubmission } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { trialPicker } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const slot = (page: Page) => page.locator('[data-slot="primary"]');
const terms = (page: Page) => slot(page).getByRole('table', { name: 'Objective Terms', exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });
const column = (page: Page, name: string) => terms(page).getByRole('columnheader').allInnerTexts()
  .then((names) => names.indexOf(name));
const measures = (page: Page) => terms(page).getByRole('textbox', { name: 'Measure', exact: true })
  .evaluateAll((inputs: HTMLInputElement[]) => inputs.map(({ value }) => value));

// Catches an Evaluate of a checkpoint that shows no Objective or one other than its Train's, a cell of the inherited
// Objective that takes an edit, and a finished run whose terms have no Value or do not count the trials they cover.
test("an Evaluate of a checkpoint reads its Train's Objective, and after the run each term's Value and Covered",
  { tag: '@round' }, async ({ page, studio }) => {
    test.setTimeout(80_000);
    const projectId = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'objective'));
    await page.goto(`/?project=${encodeURIComponent(projectId)}`);
    await mapStep(page, 'train').click();
    await tab(page, 'Objective').click();
    await expect(terms(page).locator('tbody tr')).not.toHaveCount(0);
    const trained = await measures(page);

    await mapStep(page, 'baseline').click();
    await tab(page, 'Objective').click();
    await expect.poll(() => measures(page)).toEqual(trained);
    await expect(terms(page).locator('input:not([readonly]):not([disabled]), [contenteditable="true"]')).toHaveCount(0);
    const measure = terms(page).getByRole('textbox', { name: 'Measure', exact: true }).first();
    const before = await measure.inputValue();
    await measure.click();
    await page.keyboard.type('x');
    await expect(measure).toHaveValue(before);
    await expect(slot(page).getByRole('button', { name: 'train', exact: true })).toBeVisible();

    const runs = `/api/projects/${projectId}/train-runs`;
    const plans = page.getByRole('table', { name: 'Runs', exact: true });
    for (const step of ['train', 'baseline']) {
      await mapStep(page, step).click();
      await tab(page, 'Runs').click();
      const accepted = page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === runs);
      await page.getByRole('button', { name: 'Run', exact: true }).click();
      const response = await accepted;
      expect(response.status(), await response.text()).toBe(202);
      expect((await response.json() as StudioJobSubmission).items[0].disposition).toBe('admitted');
      await expect(plans.getByText('Done', { exact: true }).first()).toBeVisible({ timeout: 25_000 });
    }
    await plans.getByRole('button', { name: /^\d+ baseline$/ }).click();
    await tab(page, 'Objective').click();
    await expect(terms(page).getByRole('columnheader', { name: 'Covered', exact: true })).toBeVisible();
    const value = await column(page, 'Value');
    const covered = await column(page, 'Covered');
    expect(value).toBeGreaterThan(0);
    const [, run] = (await trialPicker(page).innerText()).match(/^trial 1 of (\d+)$/) ?? [];
    expect(run).toBeDefined();
    const rows = terms(page).locator('tbody tr').filter({ has: page.getByRole('rowheader') });
    await expect(rows).toHaveCount(trained.length);
    for (let index = 0; index < trained.length; index += 1) {
      const cells = rows.nth(index).locator('th, td');
      await expect(cells.nth(value)).toHaveText(/\d/);
      await expect(cells.nth(covered)).toHaveText(`${run} of ${run}`);
    }
    await slot(page).getByRole('button', { name: 'train', exact: true }).click();
    await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  });
