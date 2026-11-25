import { expect, test } from './fixtures/studio';
import { type Locator, type Page } from '@playwright/test';
import {
  type ModelDocument, type StudioJobSubmission, type StudioSavedTrainRuns,
  type TaskDocument,
} from '../../src/generated/studioContracts';
import { enter, mapStep } from './fixtures/canonicalTrain';
import { post, readSaved, startProject } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

/** The Model edit a gesture sent, after checking that it was applied. */
async function applied(edit: ReturnType<typeof post>) {
  const response = await edit;
  expect(response.ok(), await response.text()).toBe(true);
  const { result } = await response.json();
  expect(result.status, result.problem?.detail).toBe('applied');
  return result.snapshot.project.documents.model.payload as ModelDocument;
}

/** A handle grows while a wire is dragged, so the pointer goes to its centre rather than waiting for it to settle. */
async function centre(target: Locator) {
  const box = (await target.boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function drag(page: Page, from: Locator, to: Locator | { x: number; y: number }) {
  const start = await centre(from);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x + 20, start.y + 20, { steps: 2 });
  // The backend's verdicts arrive after the drag starts; a drop before them makes no wire.
  if (!('x' in to)) await expect(to.locator('xpath=..')).toHaveAttribute('data-verdict', 'allowed');
  const end = 'x' in to ? to : await centre(to);
  await page.mouse.move(end.x, end.y, { steps: 8 });
  await page.mouse.up();
}

/**
 * Authors a Train's Model with the pointer in the Model view, from an empty model, and trains it.
 * Catches a Model view that cannot author a model a Train can run: a gesture that writes nothing, a wire stored on the
 * wrong ports, or a document that does not survive a reload.
 */
test('authors a Model with the pointer and trains it @round', async ({ page, request }) => {
  test.setTimeout(58_000);
  const { snapshot } = await startProject(page);
  const projectPath = '/api/projects/' + snapshot.project_id;
  const runsPath = projectPath + '/train-runs';
  const saved = () => readSaved(request, projectPath);
  const savedModel = async () => rootPayload((await saved()).project.documents?.model) as unknown as ModelDocument;
  const edit = () => post(page, projectPath + '/model/edit');

  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true }).click();
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const task = page.getByRole('tabpanel', { name: 'Task', exact: true });
  const objective = page.getByRole('tabpanel', { name: 'Objective', exact: true });
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const canvas = model.locator('.react-flow__pane');
  const inputs = model.locator('[data-id="@inputs"]');
  const outputs = model.locator('[data-id="@outputs"]');
  const card = (id: string) => model.locator(`.react-flow__node[data-id="${id}"]`);
  const handle = (id: string, key: string) => card(id).locator(`[data-handleid="${key}"]`);

  // The empty model is its two boundary cards; the task's signals are hollow ports on `inputs`.
  await expect(model.locator('.react-flow__node')).toHaveCount(2);
  await expect(inputs.locator('[data-handleid^="output:@task:"]')).not.toHaveCount(0);

  // The search at the cursor, by its key, places the component there, selected.
  const box = (await canvas.boundingBox())!;
  const searched = page.getByRole('dialog', { name: 'Add Component', exact: true });
  async function place(name: string, open: () => Promise<void>) {
    await open();
    await searched.getByRole('textbox').fill(name);
    const adding = edit();
    await searched.getByRole('button', { name: 'Add ' + name, exact: true }).last().click();
    const stored = await applied(adding);
    const [id] = Object.entries(stored.interior.components ?? {})
      .filter(([, component]) => component.kind === 'leaf' && component.type === 'feedbax.' + name).map(([key]) => key);
    await expect(card(id)).toHaveClass(/selected/);
    return id;
  }
  const cursor = { x: box.x + box.width * 0.35, y: box.y + box.height * 0.6 };
  const cell = await place('GRU', async () => {
    await page.mouse.click(cursor.x, cursor.y);
    await page.keyboard.press('Shift+A');
  });
  const cellBox = (await card(cell).boundingBox())!;
  expect(Math.abs(cellBox.x - cursor.x) + Math.abs(cellBox.y - cursor.y)).toBeLessThan(4);
  await enter(inspector.getByRole('textbox', { name: 'Input size', exact: true }), '2');
  await expect.poll(async () => (await savedModel()).interior.components?.[cell]).toMatchObject({ fields: { input_size: 2 } });

  // A double click on empty canvas opens the same search.
  const readout = await place('Linear', () => canvas.dblclick({ position: { x: box.width * 0.6, y: box.height * 0.6 } }));
  await enter(inspector.getByRole('textbox', { name: 'Input size', exact: true }), '4');
  await enter(inspector.getByRole('textbox', { name: 'Output size', exact: true }), '2');
  await expect.poll(async () => (await savedModel()).interior.components?.[readout])
    .toMatchObject({ fields: { input_size: 4, output_size: 2 } });

  // Dropping a wire on empty canvas offers the components with an input that fits, and places the chosen one wired.
  let joining = edit();
  await drag(page, handle(readout, 'output:output'), { x: box.x + box.width * 0.85, y: box.y + box.height * 0.3 });
  await searched.getByRole('button', { name: 'PointMass · force', exact: true }).click();
  let stored = await applied(joining);
  const [plant] = Object.entries(stored.interior.components ?? {})
    .filter(([, component]) => component.kind === 'leaf' && component.type === 'feedbax.PointMass').map(([key]) => key);
  const readoutOutput = { owner: 'component', component: readout, port: 'output' };
  expect(Object.values(stored.interior.wires ?? {})).toEqual([expect.objectContaining({
    source: readoutOutput, target: { owner: 'component', component: plant, port: 'force' },
  })]);

  // While a wire is dragged the inputs it can feed are offered; a drop on one makes the wire.
  const start = await centre(handle(cell, 'output:output'));
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.15, { steps: 4 });
  await expect(card(readout).locator('.studio-model-port[data-verdict="allowed"]')).toHaveCount(1);
  joining = edit();
  const end = await centre(handle(readout, 'input:input'));
  await page.mouse.move(end.x, end.y, { steps: 4 });
  await page.mouse.up();
  await applied(joining);

  // Wiring a hollow task port creates the Model input it feeds, which then carries the Task mark.
  joining = edit();
  await drag(page, inputs.locator('[data-handleid="output:@task:target_pos"]'), handle(cell, 'input:input'));
  stored = await applied(joining);
  const target = stored.ports.inputs?.find(({ name }) => name === 'target_pos');
  expect(target).toBeDefined();
  await expect(inputs.getByRole('img', { name: 'Task', exact: true })).toHaveCount(1);

  // Wiring to the empty handle of `outputs` adds a port named after the wire's source, and an Objective term on it marks it.
  joining = edit();
  await drag(page, handle(readout, 'output:output'), outputs.locator('[data-handleid="input:__new_port__"]'));
  stored = await applied(joining);
  const command = stored.ports.outputs?.find(({ name }) => name === 'output');
  expect(command).toBeDefined();
  expect(Object.values(stored.interior.wires ?? {})).toContainEqual(expect.objectContaining({
    source: readoutOutput, target: { owner: 'outputs', port: command!.id },
  }));
  await tab('Objective').click();
  await objective.getByRole('button', { name: 'Add Term', exact: true }).click();
  await page.getByRole('menu', { name: 'Add Term', exact: true }).getByRole('menuitem', { name: 'Output', exact: true }).click();
  await expect(objective.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(output)');
  await tab('Model').click();
  await expect(outputs.getByRole('img', { name: 'Objective', exact: true })).toHaveCount(1);

  // Simple reaches starts nothing until `start` is chosen to initialize this Model's point mass.
  const plantLink = JSON.stringify({ composites: [], element: {
    kind: 'field', parent: { id: plant, kind: 'component' }, path: ['state', 'effector/position'],
  } });
  const links = async () => {
    const { project } = await saved();
    const { document } = (project.steps?.train?.base as { inline: { trial_documents: { task: { document: string } } } })
      .inline.trial_documents.task;
    return (rootPayload(project.documents?.[document]) as unknown as TaskDocument).links;
  };
  await tab('Task').click();
  const initializes = task.getByRole('combobox', { name: 'Initializes start', exact: true });
  await initializes.click();
  await page.keyboard.type('effector position');
  await page.getByRole('option', { name: 'PointMass · effector · position', exact: true }).click();
  await page.keyboard.press('Escape');
  await expect.poll(links).toEqual([{ model_field: plantLink, variable: 'start' }]);

  await tab('Runs').click();
  await enter(inspector.getByRole('textbox', { name: 'Batches', exact: true }), '2');
  await enter(inspector.getByRole('textbox', { name: 'Batch Size', exact: true }), '4');
  await expect.poll(async () => (rootPayload((await saved()).project.documents?.training) as {
    training_config?: { n_batches?: number; batch_size?: number };
  }).training_config).toMatchObject({ n_batches: 2, batch_size: 4 });

  const authored = await saved();
  const authoredModel = rootPayload(authored.project.documents?.model) as unknown as ModelDocument;
  expect(Object.values(authoredModel.interior.wires ?? {})).toEqual(expect.arrayContaining([
    expect.objectContaining({ source: { owner: 'inputs', port: target!.id }, target: { owner: 'component', component: cell, port: 'input' } }),
    expect.objectContaining({ source: { owner: 'component', component: cell, port: 'output' },
      target: { owner: 'component', component: readout, port: 'input' } }),
    expect.objectContaining({ source: readoutOutput, target: { owner: 'component', component: plant, port: 'force' } }),
    expect.objectContaining({ source: readoutOutput, target: { owner: 'outputs', port: command!.id } }),
  ]));
  const reviewed = async () => {
    const response = await request.get(runsPath, { params: { revision: authored.revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const before = await reviewed();
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toHaveLength(1);

  await page.reload();
  await mapStep(page, 'train').click();
  await tab('Model').click();
  await expect(model.locator('.react-flow__edge')).toHaveCount(4);
  expect(await savedModel()).toEqual(authoredModel);
  await tab('Runs').click();
  await runs.getByRole('table', { name: 'Runs', exact: true }).getByRole('button').first().click();
  const submitted = post(page, runsPath);
  await runs.getByRole('button', { name: 'Run', exact: true }).click();
  const response = await submitted;
  expect(response.status(), await response.text()).toBe(202);
  const { items } = await response.json() as StudioJobSubmission;
  expect(items).toEqual([expect.objectContaining({ disposition: 'admitted', job_id: expect.any(String) })]);
  await expect.poll(async () => (await reviewed()).rows[0]?.attempts[0]?.status, { timeout: 20_000 }).toBe('succeeded');
  expect((await reviewed()).rows[0].attempts[0]).toMatchObject({
    job_id: items[0].job_id, progress: { completed_updates: 2, total_updates: 2 },
  });
});
