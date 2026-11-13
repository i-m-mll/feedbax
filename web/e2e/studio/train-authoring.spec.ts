import { expect, test, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { parseContract, type StudioProjectCreateFile } from '../../src/generated/studioContracts';
import type { mountTrainAuthoring } from './fixtures/trainAuthoringProbe';
import { reachingDocument } from './reachingProject';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
type Probe = Window & { trainAuthoring: Awaited<ReturnType<typeof mountTrainAuthoring>> };

async function mount(page: Page, project: string | StudioProjectCreateFile['file'][]) {
  return page.evaluate(async (target) => {
    const entry = '/e2e/studio/fixtures/trainAuthoringProbe.tsx';
    const { mountTrainAuthoring } = await import(entry) as { mountTrainAuthoring: typeof mountTrainAuthoring };
    const container = document.createElement('div');
    document.body.replaceChildren(container);
    const mounted = await mountTrainAuthoring(container, target);
    (window as Probe).trainAuthoring = mounted;
    return mounted.snapshot().saved;
  }, project);
}

async function plan(root: string, revision: string) {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', 'web/e2e/studio/fixtures/train_plan.py', root, revision,
  ], { cwd: repo, timeout: 10_000 });
  return JSON.parse(stdout);
}

test('Train authoring completes the real reaching Plan through the folder controller', async ({ page, request }, info) => {
  const surfaceResponse = await request.get('/api/projects/authoring/schema');
  expect(surfaceResponse.status()).toBe(200);
  const surface = parseContract('StudioAuthoringSurface', await surfaceResponse.json());
  expect(Object.keys(surface.step_payloads)).toEqual(['train']);
  expect(surface.project_grammar).toHaveProperty('$defs.Step.properties.kind.$ref', '#/$defs/StepKind');
  expect(surface.project_grammar).toHaveProperty('$defs.StepKind.enum', ['train', 'evaluate', 'analyze', 'report']);
  expect(surface.step_payloads.train).toHaveProperty('required', [
    'graph', 'task', 'objective', 'method_payload', 'method_ref', 'training_config',
  ]);
  expect(surface.training_methods).toEqual(expect.arrayContaining([expect.objectContaining({
    key: 'feedbax/studio_graph/v1', owner: 'feedbax.training.studio_executor',
    payload_schema: expect.objectContaining({ $defs: expect.objectContaining({ OptimizerSpec: expect.any(Object) }) }),
  })]));

  const fixture = await reachingDocument();
  const { graph } = fixture;
  graph.retained_observables = [{
    id: 'observable:endpoint-position', label: 'Endpoint position', selector: 'path:plant.effector.pos',
    value_schema: { id: 'endpoint-position', label: 'Endpoint position', kind: 'vector', shape: [2], origin: 'declared' },
  }];
  const scenario = fixture.workspace!.scenarios['scenario:train'];
  await page.goto('/');
  const { project_id: id } = await mount(page, [
    { kind: 'document', id: 'model', document: { schema: 'feedbax.spec.graph', payload: JSON.parse(JSON.stringify(graph)) } },
    { kind: 'document', id: 'stale-model', document: { schema: 'feedbax.spec.graph', payload: JSON.parse(JSON.stringify(graph)) } },
    { kind: 'document', id: 'task', document: { schema: 'feedbax.spec.task', payload: scenario.task_spec! } },
    { kind: 'step', id: 'train', step: { kind: 'train', base: { inline: {
      task_binding_spec: JSON.parse(JSON.stringify(scenario.task_binding_spec)),
    } } } },
  ]);
  const root = join(String(info.config.metadata.home), 'state', 'web', 'projects', id);
  const saved = () => page.evaluate(() => (window as Probe).trainAuthoring.snapshot().saved);
  const model = page.getByRole('combobox', { name: 'Edit Model', exact: true });
  const task = page.getByRole('combobox', { name: 'Edit Task', exact: true });
  const folderPath = `**/api/projects/${id}`;
  let competed = false;
  await page.route(folderPath, async (route) => {
    if (route.request().method() === 'PATCH' && !competed) {
      competed = true;
      await unlink(join(root, 'documents', 'stale-model.json'));
    }
    await route.continue();
  });
  await model.selectOption({ label: 'stale-model' });
  await expect.poll(() => page.evaluate(() => Boolean((window as Probe).trainAuthoring.snapshot().outside))).toBe(true);
  const kept = await page.evaluate(async () => {
    const probe = (window as Probe).trainAuthoring;
    try { await probe.keep('edits'); }
    catch (error) {
      await probe.keep('disk');
      return (error as Error).message;
    }
    return null;
  });
  expect(kept).toBe('The project changed before the pending edits could be saved.');
  expect((await saved()).project.documents).not.toHaveProperty('stale-model');
  await page.unroute(folderPath);

  await model.selectOption({ label: 'model' });
  await expect(model).toHaveValue('0');
  await task.selectOption({ label: 'task' });
  await expect(task).toHaveValue('0');
  const baseline = await saved();
  const before = await plan(root, baseline.revision);
  expect(before).toEqual({
    revision: baseline.revision, semanticRoot: baseline.semantic_root, storeExists: false, rows: 1, runs: 0,
    diagnostics: ['method_payload', 'method_ref', 'objective', 'training_config'].map(
      (field) => ['invalid_payload', `payload.${field}`],
    ),
  });
  const authoringPath = `/api/projects/${id}/authoring/train/train`;
  const bad = await request.post(authoringPath, { data: {
    revision: baseline.revision, change: { field: '/method_ref/name', value: 'feedbax/unknown/v1' },
  } });
  expect(bad.status()).toBe(422);
  expect((await saved()).revision).toBe(baseline.revision);

  const objective = page.getByRole('combobox', { name: 'Edit Final-state objective' });
  await objective.selectOption({ label: 'Endpoint position → Reach target' });
  await expect(objective).toHaveValue('0');
  const method = page.getByRole('combobox', { name: 'Edit Training method' });
  await method.selectOption({ label: 'Studio graph' });
  await expect(method).toHaveValue('0');
  const count = page.getByRole('spinbutton', { name: 'Edit N Batches' });
  await expect(count).toHaveValue('');
  await count.fill('24');
  await count.press('Enter');
  await expect(count).toHaveValue('24');
  await expect(count).toBeEnabled();
  const complete = await plan(root, (await saved()).revision);
  expect(complete).toMatchObject({ rows: 1, runs: 1, diagnostics: [] });

  const undone = await page.evaluate(() => (window as Probe).trainAuthoring.undo());
  expect(await plan(root, undone)).toMatchObject({
    rows: 1, runs: 0, diagnostics: [['invalid_payload', 'payload.training_config']],
  });
  const redone = await page.evaluate(() => (window as Probe).trainAuthoring.redo());
  expect(redone).toBe(complete.revision);
  expect((await plan(root, redone)).requestId).toBe(complete.requestId);

  const rate = page.getByRole('textbox', { name: 'Learning Rate', exact: true });
  await rate.fill('0.002');
  await rate.press('Enter');
  await expect(rate).toHaveValue('0.002');
  const optimizer = page.getByRole('combobox', { name: 'Edit Optimizer' });
  await optimizer.selectOption({ label: 'sgd' });
  const seed = page.getByRole('textbox', { name: 'Seed', exact: true });
  await seed.fill('7');
  await seed.press('Enter');
  await expect(seed).toHaveValue('7');
  await expect(seed).toBeEnabled();
  const interval = page.getByRole('textbox', { name: 'Snapshot Interval', exact: true });
  await interval.fill('12');
  await interval.press('Enter');
  await expect(interval).toHaveValue('12');
  await expect(interval).toBeEnabled();
  const finished = await saved();
  expect(finished.project.documents).toEqual(baseline.project.documents);
  const stale = await request.post(authoringPath, { data: { revision: baseline.revision } });
  expect(stale.status()).toBe(409);
  expect(await stale.json()).toMatchObject({ detail: { current: finished.revision, resource: 'project' } });

  await page.evaluate(() => (window as Probe).trainAuthoring.close());
  await page.reload();
  expect(await mount(page, id)).toEqual(finished);
  const after = await plan(root, finished.revision);
  expect(after).toEqual({
    revision: finished.revision, semanticRoot: finished.semantic_root, storeExists: false,
    rows: 1, runs: 1, diagnostics: [],
    requestId: 'train:4460007977edbeb5724fb129d505b3b7d211a3a4610454eb9c2b69264f015028',
    model: 'feedbax.PointMass', task: { type: 'SimpleReaches', params: { n_steps: 100 } }, binding: 'reach_target',
    objective: ['target_state', 'Endpoint position', 'path:plant.effector.pos', 'task_data:reach_target', 'final'],
    method: 'feedbax/studio_graph/v1', optimizer: 'sgd', seed: 7,
    config: { n_batches: 24, batch_size: 1, learning_rate: 0.002, n_reach_steps: 100, snapshot_interval: 12 },
    sources: [
      ['document', 'model', '/nodes/plant/type'],
      ['document', 'task', '/type'],
      ['step', 'train', '/base/inline/objective/loss/selector'],
      ['step', 'train', '/base/inline/method_ref/name'],
      ['step', 'train', '/base/inline/method_payload/payload/target_total'],
      ['step', 'train', '/base/inline/training_config/n_batches'],
    ],
  });
  await page.evaluate(() => (window as Probe).trainAuthoring.close());
  console.info('Train Plan', JSON.stringify({ before, after }));
});
