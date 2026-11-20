import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { appendManagedFiles, fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const evaluation = (id: string, subject: string) => ({ kind: 'step', id, step: {
  kind: 'evaluate',
  base: { inline: {
    evaluation_type: 'feedbax.rollout',
    params: { seed: 731, trial_policy: 'paired', variation: [], trial_set: { kind: 'every_condition' } },
  } },
  inputs: { subject: { step: subject, checkpoint: 'last' } },
} });

/** `train` and `sibling` read `point-mass`, and each of `extra` is added after them. */
async function project(request: APIRequestContext, extra: readonly unknown[] = []) {
  const { snapshot, projectPath } = await seedManagedProject(request, [...await fixture('history-files'), ...extra]);
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  /** The plan of the saved revision; a save between the two reads is read again. */
  async function runs(): Promise<StudioSavedTrainRuns> {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}&window_map=true`);
    if (response.status() === 409) return runs();
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  /** Each step's request id. */
  async function requests() {
    const planned = (await runs()).window_map?.requests ?? [];
    return Object.fromEntries(planned.filter(({ element }) => !element).map(({ key, request_id }) => [key.step, request_id]));
  }
  const documents = async () => (await saved()).project.documents ?? {};
  return { snapshot, saved, runs, requests, documents };
}

function modelView(page: Page) {
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const picker = primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Model Document / });
  const menu = page.getByRole('menu', { name: 'Model Document', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
  const card = (kind: string, step: string) => page.getByRole('article', { name: `${kind} ${step}`, exact: true });
  async function open(step: string) {
    await mapStep(page, step).click();
    await page.getByRole('tab', { name: 'Model', exact: true }).click();
    await primary.locator('.react-flow__node').filter({ hasText: 'Point mass' }).locator('header').click();
    await expect(mass).toBeEditable();
  }
  async function setMass(value: string) {
    await mass.fill(value);
    await mass.press('Enter');
  }
  return { primary, picker, menu, inspector, mass, card, open, setMass };
}

test('a based document marks the value it changes, and Reset of its only one gives the step its runs on the base', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, requests, documents } = await project(request);
  const { picker, mass, open, setMass } = modelView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('train');
  const initial = await requests();
  const base = await mass.inputValue();
  await expect(mass).not.toHaveAccessibleDescription(/Changed/);
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker).toHaveText('point-mass1 of 2');
  await setMass('2.5');
  await expect(picker).toHaveText('point-mass-2');
  // The view stays on the element it showed when the header changed document.
  await expect(mass).toHaveValue('2.5');
  await expect(mass).toHaveAccessibleDescription('Changed');
  expect((await requests()).train).not.toBe(initial.train);

  await page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('button', { name: 'Changed', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Reset', exact: true }).click();
  await expect(mass).toHaveValue(base);
  await expect(mass).not.toHaveAccessibleDescription(/Changed/);
  await expect.poll(async () => (await documents())['point-mass-2']).toMatchObject({ differences: {} });
  expect(await requests()).toEqual(initial);
});

test('an edit in an Evaluate step\'s Model view makes a document for the evaluation alone', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, requests, documents } = await project(request, [evaluation('check', 'train')]);
  const { picker, menu, mass, card, open, setMass } = modelView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('check');
  await expect(picker).toHaveText('point-mass1 of 3');
  await expect(card('Evaluate', 'check')).toHaveAttribute('data-document-scope', 'lit');
  for (const step of ['train', 'sibling']) await expect(card('Train', step)).toHaveAttribute('data-document-scope', 'plain');
  const initial = await requests();
  const files = await documents();
  await setMass('2.5');
  await expect(picker).toHaveText('point-mass-2');
  await expect(mass).toHaveValue('2.5');
  await expect(mass).toHaveAccessibleDescription('Changed by check');
  expect((await documents())['point-mass-2']).toMatchObject({ base: { document: 'point-mass' } });
  expect((await documents())['point-mass']).toEqual(files['point-mass']);
  const edited = await requests();
  expect(edited.check).not.toBe(initial.check);
  expect([edited.train, edited.sibling]).toEqual([initial.train, initial.sibling]);

  await picker.click();
  await expect(menu.getByRole('menuitemradio')).toHaveText(['point-mass', 'point-mass-2']);
  await expect(menu.getByRole('menuitem', { name: 'Detach', exact: true })).toHaveCount(0);
  await menu.getByRole('menuitem', { name: 'Duplicate', exact: true }).click();
  const rename = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header').getByRole('textbox');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('point-mass-3');
  await page.keyboard.press('Escape');
  await expect(picker).toHaveText('point-mass-3');
  expect((await documents())['point-mass-3']).not.toHaveProperty('base');
  await picker.click();
  const third = menu.getByRole('menuitemradio', { name: 'point-mass-3', exact: true }).locator('span').last();
  expect(await third.evaluate((element) => getComputedStyle(element).paddingInlineStart)).toBe('0px');
  await page.keyboard.press('Escape');
});

test('a second Evaluate step and a new Train step can each pick the evaluation\'s document', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, saved } = await project(request, [evaluation('check', 'train'), evaluation('check-2', 'train')]);
  const { picker, menu, open, setMass } = modelView(page);
  const reads = async (step: string) => {
    const base = (await saved()).project.steps?.[step]?.base;
    return base && 'inline' in base ? (base.inline as { trial_documents?: { model?: { document: string } } }).trial_documents?.model?.document : undefined;
  };

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('check');
  await mapStep(page, 'check-2').click({ modifiers: ['Shift'] });
  await setMass('2.5');
  await expect(picker).toHaveText('point-mass-2');
  expect(await reads('check-2')).toBeUndefined();

  for (const step of ['check-2', 'sibling']) {
    await mapStep(page, step).click();
    await page.getByRole('tab', { name: 'Model', exact: true }).click();
    await expect(picker).toHaveText(/^point-mass(\d+ of \d+)?$/);
    await picker.click();
    await menu.getByRole('menuitemradio', { name: 'point-mass-2', exact: true }).click();
    await expect(picker).toHaveText('point-mass-2');
    await expect.poll(() => reads(step)).toBe('point-mass-2');
  }
});

test('a Train step put in an Evaluate step\'s Model view takes the edit, and no document is made', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, requests, documents } = await project(request, [evaluation('check', 'train')]);
  const { picker, card, open, setMass } = modelView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('check');
  await mapStep(page, 'train').click({ modifiers: ['Shift'] });
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  for (const step of ['train', 'sibling']) await expect(card('Train', step)).toHaveAttribute('data-document-scope', 'lit');
  const initial = await requests();
  const names = Object.keys(await documents());
  await setMass('2.5');
  await expect.poll(async () => (await requests()).train).not.toBe(initial.train);
  expect((await requests()).sibling).not.toBe(initial.sibling);
  expect(Object.keys(await documents())).toEqual(names);
  await expect(picker).toHaveText('point-mass');
});

test('an Evaluate step pointed at a Model document unrelated to its subject\'s cannot run, for that document', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, runs } = await project(request, [evaluation('check', 'train')]);
  const { picker, menu } = modelView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'check').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await picker.click();
  await menu.getByRole('menuitem', { name: 'New', exact: true }).click();
  const rename = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header').getByRole('textbox');
  await expect(rename).toBeFocused();
  const created = await rename.inputValue();
  await page.keyboard.press('Escape');
  await expect(picker).toHaveText(created);
  await expect.poll(async () => (await runs()).window_map?.problems.filter(({ step }) => step === 'check')
    .map(({ place }) => place)).toEqual([expect.objectContaining({ document: created })]);
  expect((await runs()).window_map?.rows.find(({ key }) => key.step === 'check')?.status).toBe('cannot_run');
  await page.getByRole('button', { name: /^Problems \d+$/ }).click();
  await expect(page.getByRole('listitem').filter({ hasText: 'Shares no document with point-mass.' })).toContainText('check');
  // The view's refused read is on the document's picker, not a failed view.
  const header = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header');
  await expect(header.getByRole('alert')).toContainText(`${created} shares no document with point-mass.`);
  await expect(page.locator('[data-slot="primary"]').getByRole('button', { name: 'Retry', exact: true })).toHaveCount(0);
});

test('a based Objective document marks the term value it changes, and Reset gives it the base\'s', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, requests, documents } = await project(request);
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const picker = primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Objective Document / });
  const table = primary.getByRole('table', { name: 'Objective Terms', exact: true });
  const name = table.getByRole('textbox', { name: 'Name', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const inspected = inspector.getByRole('textbox', { name: 'Name', exact: true });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await expect(name).toBeEditable();
  await expect(name).not.toHaveAccessibleDescription(/Changed/);
  const initial = await requests();
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await name.fill('Effort');
  await name.press('Enter');
  await expect(picker).toHaveText('simple-reaches-objective-2');
  await name.click();
  await expect(inspected).toHaveValue('Effort');
  await expect(inspected).toHaveAccessibleDescription('Changed');
  await expect(inspector.getByRole('combobox', { name: 'Reduction', exact: true })).not.toHaveAccessibleDescription(/Changed/);
  await inspector.getByRole('button', { name: 'Changed', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Reset', exact: true }).click();
  await expect(name).toHaveValue('');
  await expect(inspected).not.toHaveAccessibleDescription(/Changed/);
  await expect.poll(async () => (await documents())['simple-reaches-objective-2']).toMatchObject({ differences: {} });
  expect(await requests()).toEqual(initial);
});

test('an attachment a based Model document adds is marked as authored there, and not in a document based on it', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot: seeded } = await project(request);
  const { snapshot } = await appendManagedFiles(request, seeded, [
    { kind: 'document', id: 'point-mass-2', document: {
      schema: 'feedbax.spec.model', base: { document: 'point-mass' }, differences: {
        '/interior/wires/decode/attachments': { value: [{ id: 'lag', kind: 'delay', delay: { value: 1, unit: 'st' }, initial: 0 }] },
      },
    } },
    { kind: 'document', id: 'point-mass-3', document: { schema: 'feedbax.spec.model', base: { document: 'point-mass-2' } } },
  ]);
  const pointed = await request.patch(`/api/projects/${snapshot.project_id}`, { data: { revision: snapshot.revision, files: [{
    address: { kind: 'step', id: 'train' },
    patches: [{ op: 'replace', path: '/base/inline/trial_documents/model', value: { document: 'point-mass-3' } }],
  }] } });
  expect(pointed.ok(), await pointed.text()).toBe(true);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer' });
  const wire = drawer.getByRole('treeitem', { name: 'Recurrent controller.output → Motor command.input', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const delay = (marked: boolean) => inspector.getByRole('treeitem', { name: marked ? /^Delay ◇/ : /^Delay\b/ });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await wire.click();
  await expect(delay(false)).toBeVisible();
  await expect(delay(true)).toHaveCount(0);
  const picker = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header')
    .getByRole('button', { name: /^Model Document / });
  await picker.click();
  await page.getByRole('menu', { name: 'Model Document', exact: true })
    .getByRole('menuitemradio', { name: 'point-mass-2', exact: true }).click();
  await expect(picker).toHaveText('point-mass-2');
  await wire.click();
  await expect(delay(true)).toBeVisible();
});

test('a based Training document marks the value it changes, and Reset gives it the base\'s', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot: seeded, saved, requests, documents } = await project(request);
  const keys = ['method', 'settings', 'training_config'] as const;
  const inline = (seeded.project.steps!.train.base as { inline: Record<string, unknown> }).inline;
  const { snapshot } = await appendManagedFiles(request, seeded, [{ kind: 'document', id: 'training', document: {
    schema: 'feedbax.spec.training_document', payload: Object.fromEntries(keys.map((key) => [key, inline[key]])),
  } }]);
  const shared = await request.patch(`/api/projects/${snapshot.project_id}`, { data: { revision: snapshot.revision, files:
    ['train', 'sibling'].map((id) => ({ address: { kind: 'step', id }, patches: keys.map((key) => ({
      op: 'replace', path: `/base/inline/${key}`, value: { document: 'training', pointer: `/${key}` },
    })) })),
  } });
  expect(shared.ok(), await shared.text()).toBe(true);
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const picker = primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Training Document / });
  const batches = primary.getByRole('textbox', { name: 'Batches', exact: true });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(batches).toBeEditable();
  const initial = await requests();
  const base = await batches.inputValue();
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  const revision = (await saved()).revision;
  await batches.fill('9');
  await batches.press('Enter');
  await expect(picker).toHaveText('training-2');
  await expect.poll(async () => (await saved()).revision).not.toBe(revision);
  await expect(batches).toHaveAccessibleDescription('Changed');
  await primary.getByRole('button', { name: 'Changed', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Reset', exact: true }).click();
  await expect(batches).toHaveValue(base);
  await expect(batches).not.toHaveAccessibleDescription(/Changed/);
  await expect.poll(async () => (await documents())['training-2']).toMatchObject({ differences: {} });
  expect(await requests()).toEqual(initial);
});

test('a based Training document marks the class it changes, and Reset gives it the base\'s', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot: seeded, documents } = await project(request);
  const keys = ['method', 'settings', 'training_config'] as const;
  const inline = (seeded.project.steps!.train.base as { inline: Record<string, unknown> }).inline;
  const { snapshot } = await appendManagedFiles(request, seeded, [
    { kind: 'document', id: 'training', document: {
      schema: 'feedbax.spec.training_document', payload: Object.fromEntries(keys.map((key) => [key, inline[key]])),
    } },
    { kind: 'document', id: 'training-2', document: {
      schema: 'feedbax.spec.training_document', base: { document: 'training' },
      differences: { '/settings/optimizer/type': { value: 'adabelief' } },
    } },
  ]);
  const pointed = await request.patch(`/api/projects/${snapshot.project_id}`, { data: { revision: snapshot.revision, files: [{
    address: { kind: 'step', id: 'train' }, patches: keys.map((key) => ({
      op: 'replace', path: `/base/inline/${key}`, value: { document: 'training-2', pointer: `/${key}` },
    })),
  }] } });
  expect(pointed.ok(), await pointed.text()).toBe(true);
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const optimizer = primary.getByRole('region', { name: /Optimizer/ }).first();

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(primary.getByRole('textbox', { name: 'Batches', exact: true })).not.toHaveAccessibleDescription(/Changed/);
  await optimizer.getByRole('button', { name: 'Changed', exact: true }).first().click();
  await page.getByRole('menuitem', { name: 'Reset', exact: true }).click();
  await expect.poll(async () => (await documents())['training-2']).toMatchObject({ differences: {} });
  await expect(optimizer.getByRole('combobox', { name: 'Optimizer', exact: true })).toHaveValue('Adam');
  await expect(optimizer.getByRole('button', { name: 'Changed', exact: true })).toHaveCount(0);
});

test('an Evaluate step\'s Task edit marks the value its document changes, and Reset gives it the base\'s', async ({ page, request }) => {
  // The service refuses a whole-step Evaluate Task edit ("A selected value is controlled by a different authored source.").
  test.fixme();
  test.setTimeout(60_000);
  const files = (JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring')) as { kind: string; id: string }[])
    .filter(({ id }) => id !== 'conditions');
  const { snapshot, projectPath } = await seedManagedProject(request, [...files, evaluation('check', 'train')]);
  const documents = async () => {
    const response = await request.get(projectPath);
    return (await response.json() as StudioProjectSnapshot).project.documents ?? {};
  };
  const before = Object.keys(await documents());
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const target = task.getByRole('textbox', { name: 'Target x (m)', exact: true });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'check').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(target).toBeEditable();
  const base = await target.inputValue();
  await target.fill('0.24');
  await target.press('Enter');
  await expect(target).toHaveAccessibleDescription('Changed by check');
  const created = Object.keys(await documents()).filter((name) => !before.includes(name));
  expect(created).toHaveLength(1);
  await task.getByRole('button', { name: 'Changed by check', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Reset', exact: true }).click();
  await expect(target).toHaveValue(base);
  await expect(target).not.toHaveAccessibleDescription('Changed by check');
  await expect.poll(async () => (await documents())[created[0]]).toMatchObject({ differences: {} });
});

test('a narrowed edit for two of three readers makes one document, and later edits of the base reach it by the document rule', async ({ page, request }) => {
  test.setTimeout(60_000);
  const [...files] = await fixture('history-files');
  const sibling = files.at(-1) as { id: string; step: { base: { inline: { seed: number } } } };
  const third = structuredClone(sibling);
  third.id = 'third';
  third.step.base.inline.seed = 92;
  const { snapshot, requests, documents } = await project(request, [third]);
  const { picker, mass, open, setMass } = modelView(page);
  const changed = async (before: Record<string, string>) => {
    const after = await requests();
    return Object.keys(before).filter((step) => before[step] !== after[step]).sort();
  };

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('train');
  await mapStep(page, 'third').click({ modifiers: ['Shift'] });
  await expect(picker).toHaveText('point-mass2 of 3');
  let before = await requests();
  await setMass('2.5');
  await expect(picker).toHaveText('point-mass-2');
  expect(await changed(before)).toEqual(['sibling', 'train']);
  const variant = (await documents())['point-mass-2'];

  await open('third');
  await expect(picker).toHaveText('point-mass');
  for (const step of ['train', 'sibling']) await mapStep(page, step).click({ modifiers: ['Shift'] });
  before = await requests();
  await setMass('3');
  await expect.poll(() => changed(before)).toEqual(['third']);
  expect((await documents())['point-mass-2']).toEqual(variant);

  for (const step of ['train', 'sibling']) await mapStep(page, step).click({ modifiers: ['Shift'] });
  await expect(picker).toHaveText('point-mass');
  before = await requests();
  await setMass('4');
  await expect.poll(() => changed(before)).toEqual(['sibling', 'third', 'train']);
  await expect(mass).toHaveValue('4');
  expect((await documents())['point-mass-2']).toMatchObject({ differences: {} });
});

test('a subject moved to a new document leaves its evaluation\'s document as it was, and picking the old one back restores the evaluation', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, requests, documents } = await project(request, [evaluation('check', 'train')]);
  const { picker, menu, open, setMass } = modelView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await open('check');
  await setMass('2.5');
  await expect(picker).toHaveText('point-mass-2');
  const evaluated = await requests();
  const files = await documents();

  await open('train');
  await expect(picker).toHaveText('point-mass');
  for (const step of ['sibling', 'check']) await mapStep(page, step).click({ modifiers: ['Shift'] });
  await expect(picker).toHaveText('point-mass1 of 3');
  await setMass('3');
  await expect(picker).toHaveText('point-mass-3');
  const { 'point-mass-3': created, ...rest } = await documents();
  expect(rest).toEqual(files);
  expect(created).toMatchObject({ base: { document: 'point-mass' } });
  const moved = await requests();
  expect(moved.check).toBeDefined();
  expect([moved.train, moved.check]).not.toEqual([evaluated.train, evaluated.check]);
  expect(moved.sibling).toBe(evaluated.sibling);

  await picker.click();
  await menu.getByRole('menuitemradio', { name: 'point-mass', exact: true }).click();
  await expect(picker).toHaveText(/^point-mass(\d+ of \d+)?$/);
  await expect.poll(requests).toEqual(evaluated);
});
