import { expect, test } from './fixtures/studio';
import { setDocumentSchemaId, type StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { trialField } from './fixtures/trialFields';
import { enter, fixture, mapStep, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type File = { kind: string; id: string; step?: Record<string, unknown>; document?: { payload?: Record<string, unknown> } };

/** Keys the families' seed authors, each of which has a label that the page shows instead. */
const keys = ['effort-k', 'reach-grid', 'side_neg', 'side_pos'];

/**
 * The ordinary project with its Objective's one term a family named Reach effort over `k` from 0 to 5, and its Task's
 * validation trials one family over a text rule (`direction`) and a rule of named values (`side`).
 */
async function families() {
  const files: File[] = await fixture('ordinary-files', '2');
  const payload = (id: string) => files.find((file) => file.id === id)!.document!.payload!;
  const objective = payload('simple-reaches-objective');
  const [term] = Object.values(objective.terms as Record<string, object>);
  objective.terms = { 'effort-k': { ...term, label: 'Reach effort', over: { k: '0..5' }, weight: 'k + 1' } };
  payload('simple-reaches-task').validation_trials = { 'reach-grid': {
    label: 'Reach {direction} {side}',
    over: { direction: '0..3', side: { 'side_neg': { value: -1, label: 'Left' }, 'side_pos': { value: 1, label: 'Right' } } },
    variables: { start: '[0.1 * side, -0.05]', target: '[0.1 * side + 0.1 * direction, 0.2]' },
  } };
  return files;
}

// Catches a family listed as the members it generates or without its count, a count that does not follow its rule, an
// index rule, a named value or its name, or a trial value that cannot be edited on the one member, a reopened project
// that shows any of them differently, and a member or value key shown in place of its label.
test('an Objective term family and a validation trial family each read as one member and are edited on it', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await families());
  const primary = page.locator('[data-slot="primary"]');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const over = inspector.getByRole('region', { name: 'Over', exact: true });
  const terms = primary.getByRole('tabpanel').getByRole('table', { name: 'Objective Terms', exact: true });
  const term = terms.getByRole('row').nth(1).getByRole('rowheader');
  const trials = page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('tree', { name: 'Validation Trials', exact: true });
  const shownKeys = async () => { const text = await page.locator('body').innerText(); return keys.filter((key) => text.includes(key)); };
  const open = async (tab: 'Objective' | 'Task') => {
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: tab, exact: true }).click();
  };
  const openTerm = async () => {
    await open('Objective');
    await terms.getByRole('row').nth(1).getByRole('textbox', { name: 'Weight', exact: true }).click();
  };
  const openTrial = async () => {
    await open('Task');
    await trials.getByRole('treeitem').first().click();
  };

  await page.goto(`/?project=${snapshot.project_id}`);
  await openTerm();
  await expect(terms.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Weight', exact: true }) })).toHaveCount(1);
  await expect(term).toContainText('× 6');
  await expect(over.getByRole('textbox', { name: 'k', exact: true })).toHaveValue('0..5');
  await enter(over.getByRole('textbox', { name: 'k', exact: true }), '0..7');
  await expect(term).toContainText('× 8');
  expect(await shownKeys()).toEqual([]);

  await openTrial();
  await expect(trials.getByRole('treeitem')).toHaveCount(1);
  await expect(trials.getByRole('treeitem')).toContainText('Reach {direction} {side}');
  await expect(trials.getByRole('treeitem')).toContainText('× 8');
  await expect(over.getByRole('textbox', { name: 'direction', exact: true })).toHaveValue('0..3');
  await enter(over.getByRole('textbox', { name: 'Right', exact: true }), '2');
  await enter(over.getByRole('textbox', { name: 'Rename Left', exact: true }), 'West');
  await expect(over.getByRole('textbox', { name: 'West', exact: true })).toHaveValue('-1');
  const target = inspector.getByRole('region', { name: 'Variables', exact: true }).getByRole('textbox', { name: 'target', exact: true });
  await enter(target, '[0.1 * side + 0.2 * direction, 0.2]');
  await expect(target).toHaveValue('[0.1 * side + 0.2 * direction, 0.2]');
  // The field shows what was typed at once; the Service's field shows it once the edit, the last of the three, is stored.
  await expect.poll(async () => (await trialField(request, snapshot.project_id,
    ['task', 'validation_trials', 'reach-grid', 'variables', 'target']))?.text).toBe('[0.1 * side + 0.2 * direction, 0.2]');
  expect(await shownKeys()).toEqual([]);

  await page.reload();
  await openTerm();
  await expect(term).toContainText('× 8');
  await expect(over.getByRole('textbox', { name: 'k', exact: true })).toHaveValue('0..7');
  await openTrial();
  await expect(trials.getByRole('treeitem')).toContainText('× 8');
  await expect(over.getByRole('textbox', { name: 'Right', exact: true })).toHaveValue('2');
  await expect(over.getByRole('textbox', { name: 'West', exact: true })).toHaveValue('-1');
  await expect(target).toHaveValue('[0.1 * side + 0.2 * direction, 0.2]');
  expect(await shownKeys()).toEqual([]);
});

/** The delayed-reaches files with a Set of a Light entry and a family of three loads, which the Train sweeps. */
async function entryFamily() {
  const all = await fixture('delayed-reaches-files') as File[];
  all.find(({ id }) => id === 'train')!.step!.sweep = { axes: { load: { set: 'loads' } } };
  return [...all, { kind: 'document', id: 'loads', document: {
    schema: setDocumentSchemaId, label: 'Loads',
    payload: { entries: { light: { label: 'Light' }, loaded: { label: 'Load {k}', over: { k: '1..3' } } } },
  } }];
}

// Catches a Set family listed as its generated entries or without its count, and a weight typed on the family that
// is not stored on the family's own key (where it gives the weight to each generated entry).
test('a Set entry family is one entry with its count, weighed as one', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await entryFamily());
  const card = page.locator('[data-map-step="train"]');
  const row = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Variables', exact: true }).getByRole('row')
    .filter({ has: page.getByRole('list', { name: 'Values Loads', exact: true }) });
  const step = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps!.train;

  await page.goto(`/?project=${snapshot.project_id}`);
  await sweepLabel(card, 'Loads').click();
  await page.getByRole('menuitem', { name: 'To Field', exact: true }).click();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const entries = row.getByRole('listitem');
  await expect(entries).toHaveCount(3);
  await expect(entries.nth(0)).not.toContainText('×');
  await expect(entries.nth(1)).toHaveText(/^Load \{k\}\s*× 3/);

  const weight = entries.nth(1).getByRole('textbox', { name: /^Weight / });
  await enter(weight, '3');
  await expect.poll(async () => (await step()).set_fields?.load?.sampling).toEqual({ kind: 'weighted', weights: { loaded: 3 } });
  await expect(weight).toHaveValue('3');
});
