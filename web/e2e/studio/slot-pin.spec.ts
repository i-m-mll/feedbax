import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioProjectDocument } from '../../src/generated/studioContracts';
import { appendManagedFiles, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { trialPicker } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const panel = (page: Page, name: string) => page.getByRole('tabpanel', { name, exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });
const footer = (page: Page) => page.locator('[data-studio-focus-region="bottom"]');

// Catches a pin that does not keep its step when another is selected, one that freezes the slot's trial as well as its
// step (the pinned Scene not following the one clock), and one not kept with the split across a reload.
test('a Scene pinned on an Evaluate stays beside its Analyze\'s figures and follows the clock', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'slot-pin',
  ));
  const opened = await request.get(`/api/projects/${project}/open`);
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  await appendManagedFiles(request, snapshot, [
    { kind: 'document', id: 'analysis', document: {
      schema: 'feedbax.analysis_document.v1', label: 'Analysis', payload: { inputs: { states: { kind: 'dataset' } } },
    } },
    { kind: 'step', id: 'analyze', step: {
      kind: 'analyze', label: 'Analyze', base: { document: 'analysis' }, inputs: { states: { step: 'baseline' } },
    } },
  ]);

  await page.goto('/?project=' + encodeURIComponent(project));
  await mapStep(page, 'baseline').click();
  await tab(page, 'Runs').click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await tab(page, 'Scene').click();
  const scene = panel(page, 'Scene');
  await expect(scene.getByRole('group', { name: 'Scene', exact: true })).toBeVisible();
  await scene.getByRole('button', { name: 'Pin', exact: true }).click();

  await mapStep(page, 'analyze').click();
  await panel(page, 'Runs').click();
  await tab(page, 'Figures').click();
  await expect(panel(page, 'Figures')).toBeVisible();
  await expect(scene.getByRole('button', { name: 'Unpin', exact: true })).toBeVisible();
  await expect(scene.getByText('Baseline', { exact: true })).toBeVisible();

  await expect(trialPicker(page)).toHaveText('nominal');
  const target = scene.getByRole('button', { name: 'Select target_pos', exact: true });
  await expect(target).toHaveCount(0);
  await footer(page).getByRole('slider', { name: 'Playback Position' }).press('End');
  await expect(target.locator('[data-scene-shown]')).toHaveCount(1);

  await page.reload();
  await expect(panel(page, 'Figures')).toBeVisible();
  await expect(scene.getByText('Baseline', { exact: true })).toBeVisible();
  await expect(scene.getByRole('group', { name: 'Scene', exact: true })).toBeVisible();
});
