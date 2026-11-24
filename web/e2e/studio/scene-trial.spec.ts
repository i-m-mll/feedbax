import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StudioJobSubmission } from '../../src/generated/studioContracts';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { trialPicker } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const footer = (page: Page) => page.locator('[data-studio-focus-region="bottom"]');
const scene = (page: Page) => page.getByRole('tabpanel', { name: 'Scene', exact: true });
const element = (page: Page, name: string) => scene(page).getByRole('button', { name: `Select ${name}`, exact: true });
const tab = (page: Page, name: string) => page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name, exact: true });

/** A Train step's files with its batch size set, which bounds how many samples the clock draws. */
async function trainFiles(batchSize: number, ...scenario: string[]) {
  const files = await fixture(...scenario) as { kind: string; step?: { base: { inline: { training_config: Record<string, unknown> } } } }[];
  files.find(({ kind }) => kind === 'step')!.step!.base.inline.training_config.batch_size = batchSize;
  return files;
}

async function openScene(page: Page, projectId: string, step: string) {
  await page.goto(`/?project=${projectId}`);
  await mapStep(page, step).click();
  await tab(page, 'Scene').click();
  await expect(scene(page).getByRole('group', { name: 'Scene', exact: true })).toBeVisible();
}

/** Where a solid mark is drawn, in the canvas's coordinates. */
const solid = (mark: Locator) => mark.locator('[data-scene-solid]').evaluate((circle: SVGCircleElement) =>
  [circle.cx.baseVal.value, circle.cy.baseVal.value]);

// Catches a Scene that draws only the mechanism before a run, a task Point placed by name rather than by type, a
// mechanism not posed by the trial's initial values (the point mass apart from the linked start), and hollow marks
// that are not one per sample of the picker's count.
test('with nothing run, the Scene draws the nominal trial solid and each sample hollow', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await trainFiles(8, 'ordinary-files', '2'));
  await openScene(page, snapshot.project_id, 'train');
  await expect(trialPicker(page)).toHaveText('nominal');
  const start = element(page, 'start');
  const target = element(page, 'target');
  await expect(start.locator('[data-scene-solid]')).toHaveCount(1);
  await expect(target.locator('[data-scene-solid]')).toHaveCount(1);
  await expect(target.locator('[data-scene-possible]')).toHaveCount(8);
  const plant = element(page, 'Point mass');
  await expect.poll(() => solid(plant)).toEqual(await solid(start));
  await page.getByRole('complementary', { name: 'Details Drawer', exact: true }).getByRole('tree', { name: 'Outline', exact: true })
    .getByRole('treeitem', { name: 'target', exact: true }).click();
  await expect(target).toHaveAttribute('aria-pressed', 'true');
});

// Catches a signal drawn whether or not it is present at the playhead, and a Scene that does not follow the clock.
test('a Point signal appears in the Scene once the playhead passes its start', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('delayed-reaches-files'));
  await openScene(page, snapshot.project_id, 'train');
  await expect(trialPicker(page)).toHaveText('nominal');
  const target = element(page, 'target_pos');
  await expect(element(page, 'start').locator('[data-scene-solid]')).toHaveCount(1);
  await expect(target).toHaveCount(0);
  await footer(page).getByRole('slider', { name: 'Playback Position' }).press('End');
  await expect(target.locator('[data-scene-solid]')).toHaveCount(1);
});

// Catches a recorded trial drawn at its initial pose rather than at the playhead's recorded state, and a recorded
// pose drawn without the rest pose beneath it.
test('with a Done evaluation the mechanism at the last step is at its path\'s end', { tag: '@round' }, async ({ page, request, studio }) => {
  test.setTimeout(80_000);
  const projectId = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'scene-trial'));
  const runs = `/api/projects/${projectId}/train-runs`;
  const plans = page.getByRole('table', { name: 'Runs', exact: true });
  const run = async (step: string) => {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const accepted = page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === runs);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await accepted;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0].disposition).toBe('admitted');
    await expect(plans.getByText('Done', { exact: true }).first()).toBeVisible({ timeout: 25_000 });
  };
  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await run('train');
  await run('baseline');
  await plans.getByRole('button', { name: /^\d+ baseline$/ }).click();
  await tab(page, 'Scene').click();
  await expect(trialPicker(page)).toHaveText(/^trial 1 of \d+$/);
  const track = scene(page).locator('[data-scene-track]');
  await expect(track).toHaveCount(1);
  await footer(page).getByRole('slider', { name: 'Playback Position' }).press('End');
  const end = await track.evaluate((line: SVGPolylineElement) => {
    const last = line.points.getItem(line.points.numberOfItems - 1);
    return [last.x, last.y];
  });
  const plant = scene(page).locator('[data-scene-layer="bodies"]').first();
  await expect.poll(async () => {
    const [x, y] = await solid(plant);
    return Math.hypot(x - end[0], y - end[1]);
  }).toBeLessThan(0.5);
  await expect(scene(page).locator('[data-scene-rest] > *')).not.toHaveCount(0);
});
