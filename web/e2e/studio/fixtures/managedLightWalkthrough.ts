import { expect, type APIRequestContext, type Page } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot,
  StudioSavedTrainKey, StudioSavedTrainRuns,
} from '../../../src/generated/studioContracts';
import { authorLearningRateSweep, mapStep } from './canonicalTrain';
import { activate, chooseOption, chooseStepKind, number, openView, palette, tabTo } from './studioKeyboard';
import { pick } from './controls';

type Walkthrough = {
  page: Page;
  request: APIRequestContext;
  projectPath: string;
  attempts(): Promise<string[]>;
};

async function snapshot({ request, projectPath }: Walkthrough) {
  const response = await request.get(projectPath);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioProjectSnapshot;
}

async function plan(context: Walkthrough) {
  const { request, projectPath } = context;
  const { revision } = await snapshot(context);
  const response = await request.get(`${projectPath}/train-runs`, { params: { revision, window_map: 'true' } });
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioSavedTrainRuns;
}

const pane = (page: Page) => page.getByRole('tabpanel', { name: 'Runs', exact: true });
const table = (page: Page) => pane(page).getByRole('table', { name: 'Runs', exact: true });
const drawer = (page: Page) => page.getByRole('complementary', { name: 'Details Drawer', exact: true });
const train = (page: Page) => page.getByRole('article', { name: 'Train train', exact: true });
const variant = (page: Page) => page.getByRole('combobox', { name: 'Variant', exact: true });

async function openRuns(page: Page, step: string) {
  await activate(page, mapStep(page, step));
  await openView(page, 'runs');
}

export async function selectModelComponent(page: Page, componentId: string) {
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const component = model.locator(`[data-id=${JSON.stringify(componentId)}]`);
  await expect(component).toHaveAttribute('aria-label', /.+/);
  const name = (await component.getAttribute('aria-label'))!;
  if (!await drawer(page).isVisible()) await palette(page, 'showDrawer', 'Show Drawer');
  const outline = drawer(page).getByRole('button', { name: 'Outline', exact: true });
  if (await outline.getAttribute('aria-pressed') !== 'true') {
    // The Model Drawer segments rove focus: Tab reaches the pressed Library segment.
    await tabTo(page, drawer(page).getByRole('button', { name: 'Library', exact: true }));
    await page.keyboard.press('ArrowRight');
    await expect(outline).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(outline).toHaveAttribute('aria-pressed', 'true');
  }
  const entry = drawer(page).getByRole('button', { name, exact: true });
  await activate(page, entry);
  await expect(entry).toHaveAttribute('aria-pressed', 'true');
}

function mutation({ page, projectPath }: Walkthrough, method: string, suffix = '') {
  return page.waitForResponse((response) => response.request().method() === method &&
    new URL(response.url()).pathname === projectPath + suffix);
}

export async function authorWalkthroughSweep(context: Walkthrough) {
  const { page } = context;
  await expect(variant(page)).toHaveText('base');
  await openRuns(page, 'train');
  await authorLearningRateSweep(drawer(page).getByRole('region', { name: 'Learning Rate field', exact: true }));
  const sweepRows = (value: StudioSavedTrainRuns) =>
    value.window_map!.rows.filter(({ key }) => key.variant === 'base' && key.step === 'train');
  await expect.poll(async () => sweepRows(await plan(context)).map(({ coordinates }) => coordinates['learning-rate']?.label))
    .toEqual(['Slow', 'Medium', 'Fast']);
  return (await plan(context)).rows.filter(({ key }) => key.variant === 'base' && key.step === 'train');
}

export async function verifyWalkthroughVariant(context: Walkthrough, plantId: string) {
  const { page, attempts } = context;
  const scoped = (value: StudioSavedTrainRuns, name: string) => {
    const window = value.window_map!;
    const requests = window.requests!.filter(({ key }) => key.variant === name);
    const current = [
      ...value.rows.filter(({ key }) => key.variant === name).map(({ result }) => result),
      ...window.evaluations!.filter(({ key }) => key.variant === name).map(({ result }) => result),
      ...window.analysis_elements!.filter(({ key }) => key.variant === name).map(({ result }) => result),
      ...window.rows.filter(({ key }) => key.variant === name && key.step === 'report').map(({ light }) => light?.result),
    ];
    return {
      requests: requests.map(({ key, element, request_id }) => [key.step, key.row, element, request_id]),
      receipts: current.map((result) => result && [result.job_id, result.receipt.manifest_sha256]),
      edges: window.edges.filter(({ to_key }) => to_key.variant === name),
    };
  };
  const original = await plan(context);
  const base = scoped(original, 'base');
  const compare = scoped(original, 'compare');
  expect(compare.receipts).toEqual(base.receipts);
  expect(compare.requests).toEqual(base.requests);
  expect(base.receipts).toHaveLength(9);
  expect(base.receipts.every(Boolean)).toBe(true);
  expect(compare.edges.length).toBeGreaterThan(0);
  expect(compare.edges.every(({ from_key }) => from_key.variant === 'compare')).toBe(true);
  const originalAttempts = await attempts();

  await pick(variant(page), 'compare');
  await activate(page, mapStep(page, 'train'));
  await openView(page, 'model');
  await selectModelComponent(page, plantId);
  const mass = page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('textbox', { name: 'Mass', exact: true });
  await expect(mass).toBeEditable();
  const saved = mutation(context, 'PATCH');
  await number(page, mass, '3');
  const response = await saved;
  expect(response.ok(), await response.text()).toBe(true);
  expect(response.request().postDataJSON().address).toEqual({ kind: 'variant', id: 'compare' });
  const changed = await plan(context);
  expect(scoped(changed, 'base')).toEqual(base);
  const delta = scoped(changed, 'compare');
  expect(delta.edges).toEqual(compare.edges);
  expect(delta.requests.map((entry) => entry.slice(0, 3))).toEqual(compare.requests.map((entry) => entry.slice(0, 3)));
  for (const [index, request] of delta.requests.entries()) expect(request[3]).not.toBe(compare.requests[index][3]);
  expect(delta.receipts.every((receipt) => receipt == null)).toBe(true);
  expect(await attempts()).toEqual(originalAttempts);
  await page.reload();
  await expect(variant(page)).toHaveText('base');
  expect(scoped(await plan(context), 'compare')).toEqual(delta);
}

export async function verifyWalkthroughSuccessors(context: Walkthrough, pairedStep: string) {
  const { page, attempts } = context;
  const stepRows = (value: StudioSavedTrainRuns, step: string) =>
    value.rows.filter(({ key }) => key.variant === 'base' && key.step === step);
  const evaluationRows = (value: StudioSavedTrainRuns, step: string) =>
    value.window_map!.evaluations!.filter(({ key }) => key.variant === 'base' && key.step === step);
  const prefix = stepRows(await plan(context), 'train');
  const sourceUpdates = prefix[0].result!.progress!.completed_updates;
  expect(prefix.slice(0, 2).map(({ result }) => result?.progress?.completed_updates)).toEqual([sourceUpdates, sourceUpdates]);
  const additionalUpdates = 4;
  const childUpdates = sourceUpdates + additionalUpdates;
  const subsetStep = pairedStep === 'evaluate' ? 'evaluate-2' : 'evaluate';
  const outline = page.getByRole('complementary', { name: 'Map Drawer', exact: true });
  const used = pane(page).getByRole('region', { name: 'Used By', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const sourceRow = (index: number) => table(page).getByRole('row').filter({ hasText: prefix[index].label }).getByRole('button');

  async function selectTwo() {
    await openRuns(page, 'train');
    await activate(page, sourceRow(0));
    await activate(page, sourceRow(1), 'Shift+Enter');
    await expect(train(page)).toContainText('2 of 3');
  }

  async function addSuccessor(kind: 'Train' | 'Evaluate', id: string, producer = 'train') {
    const saved = page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === context.projectPath + '/files' && response.request().postDataJSON().file.id === id);
    const card = page.getByRole('article', { name: 'Train ' + producer, exact: true });
    await activate(page, card.locator('button[data-command-id="newStep"]'));
    await chooseStepKind(page, kind, true);
    const response = await saved;
    expect(response.status(), await response.text()).toBe(201);
    await expect(mapStep(page, id)).toHaveAttribute('aria-pressed', 'true');
    return (await response.json() as StudioProjectDocument).snapshot;
  }

  async function runSelected(keys: StudioSavedTrainKey[]) {
    const admitted = mutation(context, 'POST', '/train-runs');
    await palette(page, 'run', 'Run');
    const response = await admitted;
    expect(response.status(), await response.text()).toBe(202);
    expect(response.request().postDataJSON().run_keys).toEqual(keys);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.map(({ disposition }) => disposition)).toEqual(keys.map(() => 'admitted'));
    return items.map(({ job_id }) => job_id!);
  }

  await openRuns(page, 'train');
  if (!await outline.isVisible()) await palette(page, 'showSteps', 'Show Steps');
  await activate(page, sourceRow(2));
  await expect(used).toHaveCount(0);
  await expect(page.getByRole('article', { name: 'Report report', exact: true })).not.toHaveAttribute('data-map-related', /.+/);
  await activate(page, sourceRow(0));
  if (!await inspector.isVisible()) await palette(page, 'showInspector', 'Show Inspector');
  await expect(inspector.getByRole('region', { name: 'Run Inspector', exact: true })
    .getByRole('heading', { name: prefix[0].label, exact: true })).toBeVisible();
  for (const [step, kind, count] of [[pairedStep, 'Evaluate', '2 of 2'], ['analyze', 'Analyze', '1 of 1'], ['report', 'Report', '1 of 1']]) {
    await expect(used).toContainText(step);
    await expect(outline.getByRole('button', { name: `Select ${step} ${kind} ${count}`, exact: true })).toBeVisible();
    await expect(mapStep(page, step)).toHaveAccessibleDescription(new RegExp(count));
  }

  await selectTwo();
  await addSuccessor('Evaluate', subsetStep);
  const input = drawer(page).getByRole('region', { name: 'Inputs', exact: true });
  await expect(input).toContainText('2 of 3');
  const subsetPlan = await plan(context);
  const partial = evaluationRows(subsetPlan, subsetStep);
  const nominal = evaluationRows(subsetPlan, pairedStep)[0];
  expect(partial.map(({ status }) => status)).toEqual(['succeeded', 'not_run']);
  expect(nominal.result?.job_id).toBeDefined();
  expect(partial[0].request_id).toBe(nominal.request_id);
  expect(partial[0].result).toEqual(nominal.result);
  const beforeSubset = await attempts();
  const admitted = await runSelected([partial[1].key]);
  await expect(table(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 24_000 });
  const settled = evaluationRows(await plan(context), subsetStep);
  const beforeWidening = await attempts();
  expect(beforeWidening).toEqual([...beforeSubset, ...admitted].sort());
  const subset = drawer(page).getByRole('group', { name: 'Subject Subset', exact: true });
  const fast = subset.getByRole('checkbox', { name: 'Fast · 0.003', exact: true });
  await activate(page, fast, 'Space');
  await expect(fast).toBeChecked();
  const edited = mutation(context, 'PATCH');
  await activate(page, subset.getByRole('button', { name: 'Apply Subset', exact: true }));
  expect((await edited).ok()).toBe(true);
  await expect(input).toContainText('3 of 3');
  const widened = evaluationRows(await plan(context), subsetStep);
  expect(widened.slice(0, 2)).toEqual(settled);
  expect(widened).toHaveLength(3);
  const extra = await runSelected([widened[2].key]);
  await expect(table(page).getByText('Done', { exact: true })).toHaveCount(3, { timeout: 24_000 });
  expect(await attempts()).toEqual([...beforeWidening, ...extra].sort());

  await selectTwo();
  const child = 'train-continuation';
  const continued = await addSuccessor('Train', child);
  expect(continued.project.steps?.[child]?.kind).toBe('train');
  const carryover = drawer(page).getByRole('combobox', { name: 'Edit Checkpoint carryover', exact: true });
  const carry = mutation(context, 'PATCH');
  await chooseOption(page, carryover, 'Carry training state');
  const carried = await carry;
  expect(carried.ok(), await carried.text()).toBe(true);
  expect(carried.request().postDataJSON().address).toEqual({ kind: 'step', id: child });
  const carriedChild = (await carried.json() as StudioProjectSnapshot).project.steps?.[child];
  expect(carriedChild?.deltas?.at(-1)?.patches).toEqual([{
    op: 'replace', path: 'checkpoint_initialization', value: {
      mode: 'continue_from', source_completed_updates: sourceUpdates,
      state: 'preserve', schedule: 'continue', rng: 'carry',
    },
  }]);
  await expect(carryover).toHaveText('Carry training state');
  const batchCount = drawer(page).getByRole('textbox', { name: 'N Batches', exact: true });
  await expect(batchCount).toBeEditable();
  const batches = mutation(context, 'PATCH');
  await number(page, batchCount, String(additionalUpdates));
  const changed = await batches;
  expect(changed.ok(), await changed.text()).toBe(true);
  expect(changed.request().postDataJSON().address).toEqual({ kind: 'step', id: child });
  const editedChild = (await changed.json() as StudioProjectSnapshot).project.steps[child];
  expect(editedChild.deltas?.at(-1)?.patches).toEqual([
    { op: 'replace', path: 'training_config.n_batches', value: additionalUpdates },
  ]);
  const beforeChildren = await attempts();
  const planned = stepRows(await plan(context), child);
  expect(planned.map(({ key }) => key.row)).toEqual(prefix.slice(0, 2).map(({ key }) => key.row));
  const successors = await runSelected(planned.map(({ key }) => key));
  await expect(table(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 28_000 });
  expect(await attempts()).toEqual([...beforeChildren, ...successors].sort());
  await page.reload();
  await openRuns(page, child);
  const after = await plan(context);
  expect(stepRows(after, 'train')).toEqual(prefix);
  expect(evaluationRows(after, subsetStep).slice(0, 2)).toEqual(settled);
  const children = stepRows(after, child);
  for (const [index, row] of children.entries()) {
    const producer = prefix[index];
    expect(after.window_map?.edges).toContainEqual({ from_key: producer.key, to_key: row.key });
    expect(row.attempts).toEqual([expect.objectContaining({
      job_id: successors[index],
      checkpoint_source: {
        job_id: producer.result?.job_id, request_id: producer.request_id, completed_updates: sourceUpdates,
      },
      progress: { completed_updates: childUpdates, total_updates: childUpdates },
    })]);
    await activate(page, table(page).getByRole('row').filter({ hasText: producer.label }).getByRole('button'));
    if (index === 0) {
      await palette(page, 'splitDetails', 'Split Details');
      await openView(page, 'results');
    }
    const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
    await expect(results.getByRole('img', { name: 'Loss', exact: true }).locator('polyline')).toHaveCount(1);
    await expect(results.getByRole('list', { name: 'Train steps' }).getByRole('listitem')).toHaveText([
      `train 1–${sourceUpdates}`, `${child} ${sourceUpdates + 1}–${childUpdates}`,
    ]);
  }

  async function openFocusedRuns(step: string) {
    await activate(page, mapStep(page, step));
    await palette(page, 'runs', 'Runs');
    const focusedTable = pane(page).and(page.locator('[data-focused]')).getByRole('table', { name: 'Runs', exact: true });
    await expect(focusedTable).toBeVisible();
    return focusedTable;
  }

  const producer = children[0];
  const producerTable = await openFocusedRuns(child);
  await activate(page, producerTable.getByRole('rowheader', { name: producer.label, exact: true }).getByRole('button'));
  const nested = child + '-continuation';
  await addSuccessor('Train', nested, child);
  const nestedTable = await openFocusedRuns(nested);
  const targets = stepRows(await plan(context), nested);
  expect(targets).toHaveLength(1);
  const [target] = targets;
  const [initialized] = await runSelected([target.key]);
  await expect(nestedTable.getByText('Done', { exact: true })).toHaveCount(1, { timeout: 24_000 });
  const [{ result, attempts: nestedAttempts }] = stepRows(await plan(context), nested);
  expect(result?.job_id).toBe(initialized);
  expect(result?.request_id).toBe(target.request_id);
  expect(result?.progress).toEqual({ completed_updates: additionalUpdates, total_updates: additionalUpdates });
  const attempt = nestedAttempts.find(({ job_id }) => job_id === initialized);
  expect(attempt?.successful_receipt).toBeTruthy();
  expect(attempt?.successful_receipt).toEqual(result?.receipt);
  expect(attempt?.checkpoint_source).toEqual({
    job_id: producer.result!.job_id, request_id: producer.request_id, completed_updates: childUpdates,
  });
}
