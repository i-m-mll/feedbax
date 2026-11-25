import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { appendManagedFiles, fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { chooseStepKind, palette } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';
import { applyTo } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

function submitted(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

test('the slot header switches, creates, duplicates, renames and re-keys a step\'s documents in place, and Activity keeps a run\'s label', { tag: '@round' }, async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, await fixture('history-files'));
  const runsPath = `${projectPath}/train-runs`;
  const table = page.getByRole('table', { name: 'Runs' });
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const header = primary.locator('.studio-window-slot__header');
  const rename = header.getByRole('textbox');
  const picker = (label: string) => header.getByRole('button', { name: new RegExp(`^${label} Document `) });
  /** Opens a Document menu by keyboard. The menu takes focus on its first item after it mounts; keys sent before that reach the trigger. */
  async function openMenu(label: string) {
    await picker(label).press('Enter');
    const menu = page.getByRole('menu', { name: `${label} Document`, exact: true });
    await expect(menu.getByRole('menuitemradio').first()).toBeFocused();
    return menu;
  }
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function rows() {
    const { revision } = await saved();
    const response = await request.get(`${runsPath}?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  async function uses(step: string) {
    const base = (await saved()).project.steps?.[step]?.base;
    return base && 'inline' in base ? (base.inline as { trial_documents: Record<string, { document: string }> }).trial_documents : undefined;
  }
  const status = async (step: string) => (await rows()).rows.find(({ key }) => key.step === step)?.status;
  async function run(step: string) {
    await mapStep(page, step).click();
    await table.getByRole('button').first().click();
    const admission = submitted(page, runsPath);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admission;
    expect(response.status(), await response.text()).toBe(202);
    expect((await response.json() as StudioJobSubmission).items[0]?.disposition).toBe('admitted');
    await expect.poll(() => status(step), { timeout: 20_000 }).toBe('succeeded');
  }

  async function renameStep(step: string, label: string) {
    await mapStep(page, step).click();
    await mapStep(page, step).press('F2');
    const name = page.locator(`[data-map-step="${step}"]`).getByRole('textbox', { name: /^Rename / });
    await name.fill(label);
    const sent = submitted(page, `${projectPath}/members/command`);
    await name.press('Enter');
    expect((await sent).ok()).toBe(true);
  }

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await run('train');
  await renameStep('sibling', 'Sibling reach');
  await run('sibling');
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();

  await expect(header.getByRole('heading', { name: 'Model', exact: true })).toBeAttached();
  await expect(header.getByRole('heading')).toHaveText('Model');
  await expect(header.getByRole('button')).toHaveCount(2);
  await expect(picker('Model')).toHaveText('Point mass');
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Model')).toHaveText('Point mass1 of 2');
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Model')).toHaveText('Point mass');
  await renameStep('train', 'Reach');
  await page.getByRole('button', { name: /^Activity( \d+)?$/ }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true })
    .getByRole('region', { name: configuration.project, exact: true });
  await expect(activity.getByRole('button', { name: 'Train Train', exact: true })).toBeVisible();
  await expect(activity.getByRole('button', { name: 'Train Sibling reach', exact: true })).toBeVisible();
  await expect(activity.getByRole('button', { name: 'Train Reach', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: /^Activity( \d+)?$/ }).click();

  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(header.getByRole('heading')).toHaveText('Task');
  await expect(picker('Task')).toHaveText('Simple reaches');
  const taskMenu = page.getByRole('menu', { name: 'Task Document', exact: true });
  await picker('Task').click();
  await taskMenu.getByRole('menuitem', { name: 'Duplicate', exact: true }).click();
  await expect(rename).toBeFocused();
  const copy = await rename.inputValue();
  expect(copy).not.toBe('Simple reaches');
  await page.keyboard.press('Escape');
  await expect(picker('Task')).toHaveText(copy);
  await expect.poll(async () => (await uses('train'))?.task.document).not.toBe('simple-reaches-task');
  expect((await uses('sibling'))?.task.document).toBe('simple-reaches-task');
  expect(await status('sibling')).toBe('succeeded');

  await picker('Task').press('Enter');
  await expect(taskMenu.getByRole('menuitemradio')).toHaveText(['Simple reaches', copy]);
  await taskMenu.getByRole('menuitemradio', { name: 'Simple reaches', exact: true }).press('Enter');
  await expect(picker('Task')).toHaveText('Simple reaches');
  await expect.poll(async () => (await uses('train'))?.task.document).toBe('simple-reaches-task');

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await expect(picker('Objective')).toHaveText('Simple reaches');
  const objectiveMenu = await openMenu('Objective');
  await page.keyboard.press('End');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('ArrowUp');
  await expect(objectiveMenu.getByRole('menuitem', { name: 'New', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('Objective');
  await page.keyboard.type('fresh');
  await page.keyboard.press('Enter');
  await expect(picker('Objective')).toHaveText('fresh');
  await expect.poll(async () => (await uses('train'))?.objective.document).toBe('objective');
  expect((await saved()).project.documents?.objective?.label).toBe('fresh');
  await picker('Objective').click();
  await page.getByRole('menuitemradio', { name: 'Simple reaches', exact: true }).click();
  await expect.poll(async () => (await uses('train'))?.objective.document).toBe('simple-reaches-objective');

  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const menu = await openMenu('Model');
  await expect(menu.getByRole('menuitemradio')).toHaveText(['Point mass']);
  await page.keyboard.press('End');
  await page.keyboard.press('ArrowUp');
  await expect(menu.getByRole('menuitem', { name: 'Rename', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('Point mass');
  await page.keyboard.press('Escape');
  await expect(picker('Model')).toHaveText('Point mass');
  const requests = (await rows()).rows.map(({ request_id }) => request_id);
  await openMenu('Model');
  await page.keyboard.press('End');
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Enter');
  await page.keyboard.type('arm');
  await page.keyboard.press('Enter');
  await expect(picker('Model')).toHaveText('arm');
  await expect.poll(async () => (await saved()).project.documents?.['point-mass']?.label).toBe('arm');
  expect([(await uses('train'))?.model.document, (await uses('sibling'))?.model.document]).toEqual(['point-mass', 'point-mass']);
  expect((await rows()).rows.map(({ request_id }) => request_id)).toEqual(requests);

  await openMenu('Model');
  await page.keyboard.press('End');
  await expect(menu.getByRole('menuitem', { name: 'Change Key', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await expect(rename).toHaveValue('point-mass');
  await page.keyboard.type((await uses('train'))!.task.document);
  await expect(rename).toHaveAttribute('aria-invalid', 'true');
  await page.keyboard.press('Enter');
  await expect(rename).toBeFocused();
  await page.keyboard.type('arm-model');
  await page.keyboard.press('Enter');
  await expect(picker('Model')).toHaveText('arm');
  await expect.poll(async () => [(await uses('train'))?.model.document, (await uses('sibling'))?.model.document])
    .toEqual(['arm-model', 'arm-model']);
  expect((await saved()).project.documents?.['arm-model']?.label).toBe('arm');
  expect((await rows()).rows.map(({ request_id }) => request_id)).toEqual(requests);
  expect((await rows()).window_map?.problems).toEqual([]);
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(picker('Model')).toHaveText('arm');

  // The renamed Model's point mass is a mechanism the Scene draws, so its tab stays offered.
  await expect(page.getByRole('tab', { name: 'Scene', exact: true })).not.toHaveAttribute('aria-disabled', 'true');

  await header.getByRole('button', { name: 'Split Details', exact: true }).click();
  const slots = page.locator('[data-slot]');
  await expect(slots.getByRole('button', { name: 'Close Slot', exact: true })).toHaveCount(2);
  await page.locator('[data-slot="primary"]').getByRole('button', { name: 'Close Slot', exact: true }).click();
  await expect(slots).toHaveCount(1);
  await expect(header.getByRole('button', { name: 'Split Details', exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('a Train step keeps a new Model with its Problems and a duplicated Model without any, and one Undo restores each', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('history-files'));
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const header = primary.locator('.studio-window-slot__header');
  const picker = (label: string) => header.getByRole('button', { name: new RegExp(`^${label} Document `) });
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function trainProblems() {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const runs = await response.json() as StudioSavedTrainRuns;
    return {
      status: runs.window_map!.rows.find(({ key }) => key.step === 'train')?.status,
      details: runs.window_map!.problems.filter(({ step }) => step === 'train').map(({ detail }) => detail),
    };
  }
  async function model() {
    const base = (await saved()).project.steps?.train?.base;
    return base && 'inline' in base ? (base.inline as { trial_documents: { model: { document: string } } }).trial_documents.model.document : undefined;
  }
  async function command(name: string) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: new RegExp(`^${name}\\b`) }).click();
  }
  async function stillAuthored() {
    await page.getByRole('tab', { name: 'Task', exact: true }).click();
    await expect(picker('Task')).toBeEnabled();
    await expect(page.getByRole('complementary', { name: 'Details Drawer', exact: true })
      .getByRole('region', { name: 'Signals' }).getByRole('treeitem').first()).toBeVisible();
    await page.getByRole('tab', { name: 'Objective', exact: true }).click();
    await expect(picker('Objective')).toBeEnabled();
    await expect(primary.getByRole('textbox', { name: 'Weight', exact: true }).first()).toBeEditable();
    await page.getByRole('tab', { name: 'Model', exact: true }).click();
    await expect(picker('Model')).toBeEnabled();
    await expect(primary.getByRole('alert')).toHaveCount(0);
  }

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(picker('Model')).toHaveText('Point mass');
  const menu = page.getByRole('menu', { name: 'Model Document', exact: true });

  for (const choice of ['New', 'Duplicate'] as const) {
    const keys = Object.keys((await saved()).project.documents ?? {});
    await picker('Model').click();
    await menu.getByRole('menuitem', { name: choice, exact: true }).click();
    const rename = header.getByRole('textbox');
    await expect(rename).toBeFocused();
    const created = await rename.inputValue();
    await page.keyboard.press('Escape');
    await expect(picker('Model')).toHaveText(created);
    await expect.poll(async () => {
      const key = await model();
      return key && ((await saved()).project.documents?.[key]?.label ?? key);
    }).toBe(created);
    const { status, details } = await trainProblems();
    if (choice === 'New') {
      expect(status).toBe('cannot_run');
      expect(details.length).toBeGreaterThan(0);
      for (const detail of details) expect(detail).toMatch(/is linked to a field the Model does not have|is not in scope here/);
      await expect(page.getByRole('button', { name: /^Problems \d+$/ })).toBeVisible();
    } else {
      // Model addresses are relative to their document, so a copy's links and measures resolve as the original's.
      expect(details).toEqual([]);
    }
    await stillAuthored();

    await command('Undo');
    await expect(picker('Model')).toHaveText('Point mass');
    await expect.poll(model).toBe('point-mass');
    expect(Object.keys((await saved()).project.documents ?? {})).toEqual(keys);
    expect((await trainProblems()).details).toEqual([]);
  }

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const base = (await saved()).project.steps?.train?.base;
  const objective = base && 'inline' in base
    ? (base.inline as { trial_documents: { objective: { document: string } } }).trial_documents.objective.document : undefined;
  const measure = page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByRole('textbox', { name: 'Measure', exact: true });
  await primary.getByRole('textbox', { name: 'Measure', exact: true }).first().click();
  await measure.fill('sqnorm(absent)');
  await measure.press('Enter');
  // `sibling` shares the Objective, so the edit asks where it applies.
  await page.getByRole('menuitem', { name: 'All 2 Steps', exact: true }).click();
  // A measure that names no signal is stored as typed, and its Problem keeps the step from running.
  await expect.poll(async () => (rootPayload((await saved()).project.documents?.[objective!]) as {
    terms: { measure: unknown }[];
  }).terms[0].measure).toEqual({ kind: 'unresolved', text: 'sqnorm(absent)' });
  expect((await trainProblems()).status).toBe('cannot_run');
});

test('New in the Document header gives an empty report that cannot run', async ({ page, request, studio }) => {
  const project = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.managed_report', 'seed', studio.home, 'empty-report'));
  const path = '/api/projects/' + project;
  const document = page.getByRole('tabpanel', { name: 'Document', exact: true });
  const header = document.locator('.studio-window-slot__header');
  async function saved() {
    const response = await request.get(path);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }

  await page.goto('/?project=' + encodeURIComponent(project));
  await mapStep(page, 'analyze').click();
  await page.getByRole('tab', { name: 'Figures', exact: true }).click();
  await page.getByRole('tabpanel', { name: 'Figures', exact: true }).getByRole('button', { name: 'profile', exact: true }).click();
  await palette(page, 'newStep', 'New Step');
  await chooseStepKind(page, 'Report', true);
  await expect(document.getByRole('listitem')).toHaveCount(1);

  await header.getByRole('button', { name: /^Report Document / }).click();
  await page.getByRole('menu', { name: 'Report Document', exact: true }).getByRole('menuitem', { name: 'New', exact: true }).click();
  const rename = header.getByRole('textbox');
  await expect(rename).toBeFocused();
  const created = await rename.inputValue();
  await page.keyboard.press('Escape');
  await expect(header.getByRole('button', { name: /^Report Document / })).toHaveText(created);
  const [key] = Object.entries((await saved()).project.documents ?? {}).find(([, file]) => file.label === created)!;
  const base = (await saved()).project.steps?.report?.base;
  expect(base && 'document' in base ? base.document : undefined).toBe(key);
  expect(rootPayload((await saved()).project.documents?.[key])).toMatchObject({ blocks: [] });
  await expect(document.getByRole('listitem')).toHaveCount(0);
  await expect(document.getByRole('button', { name: 'Add Text', exact: true })).toBeEnabled();
  await expect(document.getByRole('status')).toContainText("Can't run");
  await expect(document.getByText('The report has no blocks.')).toBeVisible();
});

/** A project whose Train steps `train` and `sibling`, and any `extra` steps, read the Training document `training`. */
async function sharedTraining(request: APIRequestContext, extra: readonly unknown[] = []) {
  const { snapshot, projectPath } = await seedManagedProject(request, [...await fixture('history-files'), ...extra]);
  const keys = ['method', 'settings', 'training_config'] as const;
  const inline = (snapshot.project.steps!.train.base as { inline: Record<string, unknown> }).inline;
  const { snapshot: withDocument } = await appendManagedFiles(request, snapshot, [{ kind: 'document', id: 'training', document: {
    schema: 'feedbax.spec.training_document', payload: Object.fromEntries(keys.map((key) => [key, inline[key]])),
  } }]);
  const shared = await request.patch(projectPath, { data: { revision: withDocument.revision, files:
    ['train', 'sibling'].map((id) => ({ address: { kind: 'step', id }, patches: keys.map((key) => ({
      op: 'replace', path: `/base/inline/${key}`, value: { document: 'training', pointer: `/${key}` },
    })) })),
  } });
  expect(shared.ok(), await shared.text()).toBe(true);
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function requests() {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const { rows } = await response.json() as StudioSavedTrainRuns;
    return Object.fromEntries(rows.map(({ key, request_id }) => [key.step, request_id]));
  }
  return { snapshot, saved, requests };
}

function trainingView(page: Page) {
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const picker = (label: string) => primary.locator('.studio-window-slot__header')
    .getByRole('button', { name: new RegExp(`^${label} Document `) });
  const batches = primary.getByRole('textbox', { name: 'Batches', exact: true });
  const menu = page.getByRole('menu', { name: 'Training Document', exact: true });
  /** How far the picker's menu indents `name` under its base. */
  async function indent(name: string) {
    await picker('Training').click();
    const item = menu.getByRole('menuitemradio', { name, exact: true }).locator('span').last();
    const padding = await item.evaluate((element) => getComputedStyle(element).paddingInlineStart);
    await page.keyboard.press('Escape');
    return padding;
  }
  return { primary, picker, batches, menu, indent };
}

test('a narrowed edit in the Training view makes a document based on the shown one, and one Undo restores the files', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, saved, requests } = await sharedTraining(request);
  const { primary, picker, batches, indent } = trainingView(page);
  /** Commits a Batches value and waits for the plan of the revision it saved. */
  async function setBatches(value: string) {
    const before = (await saved()).revision;
    await batches.fill(value);
    await batches.press('Enter');
    await expect.poll(async () => (await saved()).revision).not.toBe(before);
    return requests();
  }
  const document = (name: string) => saved().then(({ project }) => project.documents?.[name]);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(picker('Training')).toHaveText('training');
  const initial = await requests();
  const edited = await setBatches('7');
  expect(edited.train).not.toBe(initial.train);
  expect(edited.sibling).not.toBe(initial.sibling);
  expect(rootPayload(await document('training'))).toMatchObject({ training_config: { n_batches: 7 } });

  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Training')).toHaveText('training1 of 2');
  const files = (await saved()).project;
  const alone = await setBatches('9');
  expect(alone.train).not.toBe(edited.train);
  expect(alone.sibling).toBe(edited.sibling);
  expect(await document('training')).toEqual(files.documents?.training);
  expect(await document('training-2')).toMatchObject({ base: { document: 'training' } });
  await expect(picker('Training')).toHaveText('training 2');
  await expect(batches).toHaveValue('9');
  expect(await indent('training')).toBe('0px');
  expect(await indent('training 2')).not.toBe('0px');

  await primary.focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => (await saved()).project).toEqual(files);
  await expect(picker('Training')).toHaveText('training1 of 2');

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const objective = picker('Objective');
  const name = (await objective.textContent())!;
  const [key] = Object.entries((await saved()).project.documents ?? {}).find(([, file]) => file.label === name)!;
  const weight = primary.getByRole('textbox', { name: 'Weight', exact: true }).first();
  await expect(weight).toBeEditable();
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(objective).toHaveText(`${name}1 of 2`);
  const before = await document(key);
  const revision = (await saved()).revision;
  await weight.fill('0.5');
  await weight.press('Enter');
  await page.getByRole('menu', { name: 'Apply To', exact: true }).getByRole('menuitem', { name: 'Only This Step', exact: true }).click();
  await expect.poll(async () => (await saved()).revision).not.toBe(revision);
  const reached = await requests();
  expect(reached.train).not.toBe(edited.train);
  expect(reached.sibling).toBe(edited.sibling);
  expect(await document(key)).toEqual(before);
  await expect(objective).toHaveText(`${name} 2`);
});

test('a continuation taken out keeps its runs, and Detach cuts a based document from its base', async ({ page, request }) => {
  test.setTimeout(60_000);
  const continuation = { kind: 'step', id: 'train-2', step: { kind: 'train', base: { step: 'train' } } };
  const { snapshot, saved, requests } = await sharedTraining(request, [continuation]);
  const { picker, batches, menu, indent } = trainingView(page);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await mapStep(page, 'train-2').click({ modifiers: ['Shift'] });
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker('Training')).toHaveText('training1 of 3');
  const before = await requests();
  await batches.fill('9');
  await batches.press('Enter');
  await expect(picker('Training')).toHaveText('training 2');
  const after = await requests();
  expect(after.train).not.toBe(before.train);
  expect(after['train-2']).toBe(before['train-2']);
  expect(after.sibling).toBe(before.sibling);

  await picker('Training').click();
  await menu.getByRole('menuitem', { name: 'Detach', exact: true }).click();
  await expect.poll(async () => 'base' in ((await saved()).project.documents?.['training-2'] ?? {})).toBe(false);
  expect(await indent('training 2')).toBe('0px');
  expect(await requests()).toEqual(after);
  await picker('Training').click();
  await expect(menu.getByRole('menuitem', { name: 'Detach', exact: true })).toHaveCount(0);
});

test('a narrowed edit in the Model view makes a document based on the shown one, and one Undo restores the files', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('history-files'));
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const picker = primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Model Document / });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const mass = inspector.getByRole('textbox', { name: 'Mass', exact: true });
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function requests() {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const { rows } = await response.json() as StudioSavedTrainRuns;
    return Object.fromEntries(rows.map(({ key, request_id }) => [key.step, request_id]));
  }

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await expect(picker).toHaveText('Point mass');
  await primary.locator('.react-flow__node').filter({ hasText: 'Point mass' }).locator('header').click();
  await expect(mass).toBeEditable();
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(picker).toHaveText('Point mass1 of 2');
  await expect(mass).toBeEditable();
  const files = (await saved()).project;
  const before = await requests();
  await mass.fill('2.5');
  await mass.press('Enter');
  await applyTo(page).getByRole('menuitem', { name: 'Selected Readers', exact: true }).click();
  await expect(picker).toHaveText('Point mass 2');
  await expect.poll(async () => (await saved()).project.documents?.['point-mass-2']).toMatchObject({ base: { document: 'point-mass' } });
  expect((await saved()).project.documents?.['point-mass']).toEqual(files.documents?.['point-mass']);
  const after = await requests();
  expect(after.train).not.toBe(before.train);
  expect(after.sibling).toBe(before.sibling);
  await primary.locator('.react-flow__node').filter({ hasText: 'Point mass' }).locator('header').click();
  await expect(mass).toHaveValue('2.5');

  await primary.focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => (await saved()).project).toEqual(files);
  await expect(picker).toHaveText('Point mass1 of 2');
});

test('Change Key takes a key only a dangling reference spells: its bound reader follows and the dangling reader keeps its Problem', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('history-files'));
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const header = primary.locator('.studio-window-slot__header');
  const picker = (label: string) => header.getByRole('button', { name: new RegExp(`^${label} Document `) });
  const key = header.getByRole('textbox');
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function objectiveOf(step: string) {
    const base = (await saved()).project.steps?.[step]?.base;
    return base && 'inline' in base ? (base.inline as { trial_documents: { objective: unknown } }).trial_documents.objective : undefined;
  }
  async function problems(step: string) {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!.problems.filter((problem) => problem.step === step);
  }
  const { document: objective } = await objectiveOf('train') as { document: string };
  const dangling = await request.patch(projectPath, { data: { revision: snapshot.revision, files: [{
    address: { kind: 'step', id: 'sibling' },
    patches: [{ op: 'replace', path: '/base/inline/trial_documents/objective', value: { document: 'missing' } }],
  }] } });
  expect(dangling.ok(), await dangling.text()).toBe(true);
  expect(await problems('sibling')).not.toEqual([]);
  expect(await problems('train')).toEqual([]);

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await picker('Objective').click();
  await page.getByRole('menu', { name: 'Objective Document', exact: true })
    .getByRole('menuitem', { name: 'Change Key', exact: true }).click();
  await expect(key).toHaveValue(objective);
  await key.fill('missing');
  await expect(key).not.toHaveAttribute('aria-invalid', 'true');
  await key.press('Enter');
  await expect.poll(() => objectiveOf('train')).toEqual({ document: 'missing' });
  expect(await objectiveOf('sibling')).toEqual({ document: 'missing', unresolved: true });
  expect(await problems('train')).toEqual([]);
  expect(await problems('sibling')).not.toEqual([]);

  await expect(page.getByRole('button', { name: /^Problems [1-9]/ })).toBeVisible();
  await expect(picker('Objective')).toHaveText('Simple reaches');
  await mapStep(page, 'sibling').click();
  await expect(page.getByRole('tab', { name: 'Objective', exact: true })).toHaveAttribute('aria-disabled', 'true');

  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Undo\b/ }).click();
  await expect.poll(() => objectiveOf('train')).toEqual({ document: objective });
  expect(await objectiveOf('sibling')).toEqual({ document: 'missing' });
});
