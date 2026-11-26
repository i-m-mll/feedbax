import { expect, test, jobTimeout } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns, StudioSavedTrainRunRequest } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture, seedManagedProject, mapCard } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';

const primary = (page: Page) => page.locator('[data-studio-focus-region="primary"]');
const runs = (page: Page) => primary(page).getByRole('table', { name: 'Runs', exact: true });

async function select(page: Page, step: string) {
  await mapStep(page, step).click();
  await primary(page).focus();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
}

async function savedPlan(request: APIRequestContext, projectPath: string) {
  const { revision } = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: 'true' } });
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioSavedTrainRuns;
}

// Catches a rule field that offers no metric before the producer has run, an unknown metric saved without its Problem,
// a run whose producer fails the condition drawn like any other or as a failure, and a Not chosen run counted on its
// step's card.
test('a metric condition offers the declared metric, refuses an unknown one, and leaves failing runs Not chosen and uncounted',
  { tag: '@round' }, async ({ page, request, studio }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    const evaluation = 'web.e2e.studio.fixtures.canonical_evaluate';
    const { snapshot, projectPath } = await seedManagedProject(request,
      JSON.parse(await runProjectFixture(evaluation, 'measured-subset-files')));
    const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
    const rule = inspector.getByRole('combobox', { name: 'Runs', exact: true });
    const problem = rule.locator('..').getByRole('img', { name: 'Problem', exact: true });
    const save = async (text: string) => {
      await rule.fill(text);
      const saved = page.waitForResponse((response) => response.request().method() === 'PATCH'
        && new URL(response.url()).pathname === projectPath);
      await rule.press('Enter');
      expect((await saved).ok()).toBe(true);
    };

    await page.goto(`/?project=${snapshot.project_id}`);
    await select(page, 'train');
    await mapCard(page, 'train')
      .getByRole('button', { name: 'New Step Train', exact: true }).click();
    await chooseStepKind(page, 'Evaluate');
    await rule.fill('train');
    await page.getByRole('option', { name: 'train_loss', exact: true }).click();
    await expect(rule).toHaveValue('train_loss');
    await rule.press('ArrowDown');
    await expect(page.getByRole('option')).toHaveText(['<', '<=', '>', '>=']);

    await save('train_los < 1');
    await expect(rule).toHaveAttribute('aria-invalid', 'true');
    await expect(problem).toHaveAttribute('data-app-tip', /no declared final metric train_los/);

    await select(page, 'train');
    const trains = (await savedPlan(request, projectPath)).rows.filter(({ key }) => key.step === 'train');
    const posted = page.waitForResponse((response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === projectPath + '/train-runs');
    await primary(page).getByRole('button', { name: 'Run', exact: true }).click();
    expect(((await posted).request().postDataJSON() as StudioSavedTrainRunRequest).run_keys).toHaveLength(trains.length);
    await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(trains.length, { timeout: jobTimeout });
    const finished = (await savedPlan(request, projectPath)).rows.filter(({ key }) => key.step === 'train');
    const measures: Record<string, number> = JSON.parse(await runProjectFixture(evaluation, 'train-measures',
      studio.home, snapshot.project_id, 'train_loss', ...finished.map(({ result }) => result!.job_id)));
    const [lowest, next] = Object.values(measures).sort((a, b) => a - b);
    const threshold = (lowest + next) / 2;

    await select(page, 'evaluate');
    await save(`train_loss < ${threshold}`);
    await expect(rule).not.toHaveAttribute('aria-invalid', 'true');
    const notChosen = runs(page).getByRole('row').filter({ hasText: 'Not chosen' });
    await expect(notChosen).toHaveCount(trains.length - 1);
    await expect(runs(page).getByRole('row').filter({ hasText: 'Not run' })).toHaveCount(1);
    await expect(notChosen.first()).toHaveCSS('opacity', '0.5');
    const card = page.locator('[data-map-step="evaluate"]');
    await expect(card.locator('.studio-map-run-mark')).toHaveCount(trains.length);
    await expect(card.locator('.studio-map-run-mark[aria-label*="Not chosen"]')).toHaveCount(trains.length - 1);
    await expect(card.locator('.studio-map-run-mark[aria-label*="Not chosen"]').first()).toHaveCSS('opacity', '0.5');
    await expect(card.getByRole('img', { name: 'Not chosen', exact: true })).toHaveCount(0);
    await expect(card.getByRole('img', { name: 'Failed', exact: true })).toHaveCount(0);
  });
