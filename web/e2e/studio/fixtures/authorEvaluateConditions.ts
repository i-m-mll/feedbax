import { expect, type APIRequestContext, type Page } from '@playwright/test';
import type {
  StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainKey, StudioSavedTrainRuns,
} from '../../../src/generated/studioContracts';
import { mapStep } from './canonicalTrain';
import { activate, chooseStepKind, number, openView } from './studioKeyboard';

/** Create a shared Evaluate from the current Train selection and author its numeric condition axis. */
export async function authorEvaluateConditions({
  page, request, projectPath, train, selectComponent, field, conditions,
}: {
  page: Page;
  request: APIRequestContext;
  projectPath: string;
  train: StudioSavedTrainKey;
  selectComponent: () => Promise<void>;
  field: string;
  conditions: readonly { label: string; value: number }[];
}) {
  const initial = await request.get(projectPath);
  expect(initial.ok(), await initial.text()).toBe(true);
  const { revision } = await initial.json() as StudioProjectSnapshot;
  const plan = async (revision: string) => {
    const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const upstream = (await plan(revision)).rows.find(({ key }) =>
    key.variant === train.variant && key.step === train.step && key.row === train.row);
  expect(upstream?.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);

  const created = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === projectPath + '/files' &&
    response.request().postDataJSON()?.file?.step?.kind === 'evaluate');
  const card = page.locator(`[data-map-step="${train.step}"]:visible`).first();
  await activate(page, card.locator('button[data-command-id="newStep"]'));
  await chooseStepKind(page, 'Evaluate', true);
  const creation = await created;
  expect(creation.status(), await creation.text()).toBe(201);
  const stepId: string = creation.request().postDataJSON().file.id;
  const { snapshot } = await creation.json() as StudioProjectDocument;
  const initialRows = (await plan(snapshot.revision)).window_map!;
  expect(initialRows.edges.filter(({ to_key }) => to_key.step === stepId && to_key.variant === train.variant)
    .map(({ from_key }) => from_key)).toEqual([train]);

  await expect(mapStep(page, stepId)).toHaveAttribute('aria-pressed', 'true');
  await openView(page, 'model');
  await selectComponent();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const control = inspector.getByRole('region', { name: field + ' field', exact: true });
  const value = control.getByRole('textbox', { name: field, exact: true });
  const edit = async (gesture: () => Promise<void>) => {
    const saved = page.waitForResponse((response) => response.request().method() === 'PATCH' &&
      new URL(response.url()).pathname === projectPath);
    await gesture();
    const response = await saved;
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  };

  await expect(value).toBeEditable();
  await activate(page, control.getByRole('button', { name: field + ' kind: Value', exact: true }));
  await edit(() => activate(page, control.getByRole('menuitem', { name: 'Sweep', exact: true })));
  await expect(control.getByRole('button', { name: field + ' kind: Sweep', exact: true })).toBeVisible();
  await expect(value).toBeEditable();
  let authored = await edit(() => number(page, value, conditions.map(({ value }) => value).join(', ')));
  for (const [index, { label }] of conditions.entries()) {
    const name = control.getByRole('textbox', { name: field + ' value ' + (index + 1) + ' name', exact: true });
    await expect(name).toBeEditable();
    authored = await edit(() => number(page, name, label));
  }

  const projection = (await plan(authored.revision)).window_map!;
  const rows = projection.rows.filter(({ key }) => key.step === stepId && key.variant === train.variant);
  expect(rows).toHaveLength(conditions.length);
  expect(projection.edges.filter(({ to_key }) => to_key.step === stepId && to_key.variant === train.variant))
    .toEqual(rows.map(({ key }) => ({ from_key: train, to_key: key })));
  return { stepId, step: authored.project.steps[stepId], revision: authored.revision, rows };
}
