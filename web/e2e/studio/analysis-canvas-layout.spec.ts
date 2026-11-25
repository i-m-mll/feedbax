import { expect, test } from './fixtures/studio';
import type { Locator } from '@playwright/test';
import type { StudioProjectDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { appendManagedFiles, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

type Box = { x: number; y: number; width: number; height: number };
type Point = { x: number; y: number };

async function cardBoxes(analysis: Locator) {
  const cards = analysis.locator('.react-flow__node');
  return Promise.all((await cards.all()).map(async (card) => ({
    name: await card.getAttribute('aria-label'), box: (await card.boundingBox())!,
  })));
}

/** Points along each drawn wire, in page coordinates. */
async function wires(analysis: Locator) {
  return analysis.locator('.react-flow__edge path.react-flow__edge-path').evaluateAll((paths) =>
    (paths as SVGPathElement[]).map((element) => {
      const length = element.getTotalLength();
      const matrix = element.getScreenCTM()!;
      return Array.from({ length: 121 }, (_, index) => {
        const point = element.getPointAtLength((length * index) / 120).matrixTransform(matrix);
        return { x: point.x, y: point.y };
      });
    }));
}

const overlap = (a: Box, b: Box) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
const inside = (box: Box, { x, y }: Point) => x > box.x + 2 && x < box.x + box.width - 2 && y > box.y + 2 && y < box.y + box.height - 2;

async function expectClearCanvas(analysis: Locator) {
  const cards = await cardBoxes(analysis);
  const pane = (await analysis.locator('.react-flow__pane').boundingBox())!;
  for (const { name, box } of cards) {
    expect(box.x >= pane.x && box.y >= pane.y && box.x + box.width <= pane.x + pane.width
      && box.y + box.height <= pane.y + pane.height, `${name} lies past the view's edge`).toBe(true);
  }
  for (const [index, a] of cards.entries()) {
    for (const b of cards.slice(index + 1)) expect(overlap(a.box, b.box), `${a.name} and ${b.name} overlap`).toBe(false);
  }
  for (const points of await wires(analysis)) {
    for (const { name, box } of cards) {
      expect(points.some((point) => inside(box, point)), `a wire runs under ${name}`).toBe(false);
    }
    expect(points[1].x, 'a wire leaves its card rightward').toBeGreaterThan(points[0].x);
    expect(points.at(-2)!.x, 'a wire enters its card rightward').toBeLessThan(points.at(-1)!.x);
  }
}

// Catches a dropped card drawn over the card it was dropped on, a laid-out card left behind a placed one once an input
// is set, a wire to a card behind its source drawn back across the cards rather than around them, and a view that keeps
// its frame after a drop or after the Inspector narrows it, leaving a card past its edge.
test('keeps every Analysis card clear of the others and every wire clear of the cards', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'analysis-layout',
  ));
  const path = '/api/projects/' + project;
  const opened = await request.get(path + '/open');
  expect(opened.ok(), await opened.text()).toBe(true);
  const { snapshot } = await opened.json() as StudioProjectDocument;
  await appendManagedFiles(request, snapshot, [
    { kind: 'document', id: 'analysis', document: { schema: 'feedbax.analysis_document.v1', payload: {
      inputs: { states: { kind: 'dataset' } },
      operations: {
        align: {
          operation: 'feedbax.aligned_velocity',
          params: {
            position: '/v1/model/state/plant/effector~1position', velocity: '/v1/model/state/plant/effector~1velocity',
            target: '/v1/task/output/target_pos',
          },
          inputs: { states: { input: 'states' } },
        },
        summarize: {
          operation: 'feedbax.summarize_velocity', params: { ddof: 1, bounds_scale: 1.0 },
          inputs: { velocity: { element: 'align', product: 'velocity' } },
        },
      },
    } } },
    { kind: 'step', id: 'analyze', step: { kind: 'analyze', base: { document: 'analysis' }, inputs: { states: { step: 'baseline' } } } },
  ]);
  const edit = async (action: () => Promise<unknown>) => {
    const saving = page.waitForResponse((response) => response.request().method() === 'PATCH' && new URL(response.url()).pathname === path);
    await action();
    const response = await saving;
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  };

  await page.goto('/?project=' + encodeURIComponent(project));
  await mapStep(page, 'analyze').click();
  await page.getByRole('tab', { name: 'Analysis', exact: true }).click();
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const align = analysis.getByRole('group', { name: /^Select align$/i });
  await expect(analysis.locator('.react-flow__edge')).toHaveCount(2);
  await expectClearCanvas(analysis);

  await drawer.getByRole('textbox', { name: 'Search Library' }).fill('Trial summary');
  const [pane, target] = await Promise.all([analysis.locator('.react-flow__pane').boundingBox(), align.boundingBox()]);
  await edit(() => drawer.getByRole('option', { name: 'Trial summary', exact: true }).dragTo(analysis.locator('.react-flow__pane'), {
    targetPosition: { x: target!.x - pane!.x + 10, y: target!.y - pane!.y + 5 },
  }));
  const summary = analysis.getByRole('group', { name: 'Select Trial summary', exact: true });
  await expect(summary).toBeVisible();
  await expectClearCanvas(analysis);

  await summary.click();
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();
  await edit(() => pick(inspector.getByRole('combobox', { name: 'Velocity', exact: true }), 'align · velocity'));
  await expect(analysis.locator('.react-flow__edge')).toHaveCount(3);
  await expectClearCanvas(analysis);
  await page.screenshot({ path: test.info().outputPath('canvas.png') });
});

