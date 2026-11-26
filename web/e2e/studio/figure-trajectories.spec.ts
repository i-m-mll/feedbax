import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type {
  AnalysisDocument, StoredFigureDocument, StudioProjectDocument, StudioProjectSnapshot,
} from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const position = '/v1/model/state/plant/effector~1position';
const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

// Catches trajectories, colour channels, scales or 3D panels that the inspector does not save, that the Service does not
// draw as authored (no colour bar, no scene), or that a reopened project no longer draws.
test('authors coloured hand paths and 3D hidden-state trajectories in the figure inspector', { tag: '@round' }, async ({ page, request, studio }) => {
  test.setTimeout(120_000);
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.figure_trajectories', 'seed', studio.home, 'figure-trajectories',
  ));
  const path = '/api/projects/' + project;
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const layers = inspector.getByRole('tree', { name: 'Layers', exact: true });
  const choice = (name: string) => inspector.getByRole('combobox', { name, exact: true });
  const box = (name: string) => inspector.getByRole('textbox', { name, exact: true });
  const mutation = (method: string, route: string) => page.waitForResponse((response) =>
    response.request().method() === method && new URL(response.url()).pathname === route);
  async function edit(action: () => Promise<unknown>) {
    const saving = mutation('PATCH', path);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return await response.json() as StudioProjectSnapshot;
  }
  const figureOf = (saved: StudioProjectSnapshot, document: string, view: string) =>
    (rootPayload(saved.project.documents?.[document]) as unknown as AnalysisDocument).views![view]!.figure as StoredFigureDocument;
  async function run(step: string) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', path + '/train-runs');
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Done', { exact: true }).first())
      .toBeVisible({ timeout: 30_000 });
  }
  const row = (name: string) => layers.getByRole('treeitem', { name: new RegExp(`^${name}\\b`) });
  async function removeRow(name: string) {
    await row(name).click({ button: 'right' });
    await edit(() => page.getByRole('menuitem', { name: 'Remove', exact: true }).click());
  }
  async function openFigure(step: string, name: string) {
    await mapStep(page, step).click();
    await tab(page, 'Figures').click();
    await figures.getByRole('button', { name, exact: true }).click();
  }
  async function addTrajectory() {
    await inspector.getByRole('button', { name: 'Add Mark', exact: true }).click();
    await edit(() => page.getByRole('menuitem', { name: 'Trajectory', exact: true }).click());
  }
  const drawn = (name: string): Locator => figures.getByRole('img', { name: `Figure ${name}`, exact: true });

  await page.goto('/?project=' + encodeURIComponent(project));
  await run('train');
  await run('masses');
  await run('baseline');

  await openFigure('hands', 'Hand paths');
  await addTrajectory();
  await removeRow('line');
  await row('trajectory').click();
  await edit(() => pick(choice('X'), position));
  await edit(() => enter(box('X xy'), 'x'));
  await edit(() => enter(box('Y xy'), 'y'));
  await edit(() => pick(choice('Color By'), 'mass.value'));
  await row('scale').click();
  await edit(() => pick(choice('Palette'), 'cividis'));
  const hands = await edit(() => enter(box('Title'), 'Mass (kg)'));
  expect(figureOf(hands, 'hands', 'hand-paths')).toMatchObject({
    color_scales: { scale: { kind: 'continuous', palette: 'cividis', title: 'Mass (kg)' } },
    layers: {
      trajectory: {
        mark: 'trajectory', group: ['member', 'condition', 'trial'],
        encode: {
          x: { variable: position, select: { xy: 'x' } },
          y: { variable: position, select: { xy: 'y' } },
          color: { field: 'mass.value', scale: 'scale' },
        },
      },
    },
  });

  await openFigure('dynamics', 'Hidden');
  await addTrajectory();
  await removeRow('line');
  await row('trajectory').click();
  await edit(() => pick(choice('X'), 'scores'));
  await edit(() => enter(box('X component'), '1'));
  await edit(() => enter(box('Y component'), '2'));
  await edit(() => pick(choice('Z'), 'scores'));
  const hidden = await edit(() => enter(box('Z component'), '3'));
  await edit(() => pick(choice('Color By'), 'condition'));
  await row('scale').click();
  const dynamics = await edit(() => pick(choice('Kind'), 'ordinal'));
  const figure = figureOf(dynamics, 'dynamics', 'hidden');
  expect(figure).toMatchObject({
    color_scales: { scale: { kind: 'ordinal' } },
    layers: {
      trajectory: {
        mark: 'trajectory',
        encode: {
          x: { variable: 'scores', select: { component: 1 } },
          y: { variable: 'scores', select: { component: 2 } },
          z: { variable: 'scores', select: { component: 3 } },
          color: { field: 'condition', scale: 'scale' },
        },
      },
    },
  });
  // Setting Z replaces the panel's 2D labels by the bound fields' labels.
  expect(Object.values(figureOf(hidden, 'dynamics', 'hidden').layout)[0].axes_labels).toEqual({
    x: 'scores · component 1', y: 'scores · component 2', z: 'scores · component 3',
  });

  await run('dynamics');
  await run('hands');
  const draws = async () => {
    await openFigure('dynamics', 'Hidden');
    await expect(drawn('Hidden').locator('.gl-container')).toHaveCount(1);
    await expect(drawn('Hidden').locator('.legend, .colorbar').first()).toBeVisible();
    await openFigure('hands', 'Hand paths');
    await expect(drawn('Hand paths').locator('.colorbar')).toHaveCount(1);
    await expect(drawn('Hand paths').locator('.colorbar')).toContainText('Mass (kg)');
  };
  await draws();
  await page.reload();
  await draws();
  const reopened = await request.get(path + '/open');
  expect(figureOf((await reopened.json() as StudioProjectDocument).snapshot, 'dynamics', 'hidden')).toEqual(figure);
});
