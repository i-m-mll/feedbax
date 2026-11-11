import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type {
  GraphCreateResponse,
  GraphDetailResponse,
  GraphProject,
  StudioPersistenceDocument,
} from '../../src/generated/studioContracts';
import type { StudioTrainingExecutionPreparation } from '../../src/types/workspace';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
const sweepRates = [0.001, 0.0005];

async function reachingDocument(): Promise<StudioPersistenceDocument> {
  const { stdout } = await run('uv', [
    'run', '--no-sync', 'python', '-B', 'web/e2e/studio/fixtures/reaching_project.py',
  ], { cwd: repo, timeout: 10_000 });
  return JSON.parse(stdout);
}

async function saveFixture(request: APIRequestContext, document: StudioPersistenceDocument) {
  const response = await request.post('/api/graphs', { data: document });
  expect(response.ok(), await response.text()).toBe(true);
  const { data } = await response.json() as GraphCreateResponse;
  return data;
}

function openOverlay(page: Page) {
  const navigation = page.getByRole('navigation').filter({
    has: page.getByRole('button', { name: 'Projects', exact: true }),
  });
  return navigation.locator('xpath=..');
}

async function openSavedProject(page: Page, name: string) {
  await page.getByTitle('Open project', { exact: true }).click();
  const overlay = openOverlay(page);
  await overlay.getByRole('button', { name: new RegExp(`^${RegExp.escape(name)}(?: |$)`) }).click();
  await expect(overlay).toHaveCount(0);
  await expect(page.getByRole('banner').getByText(name, { exact: true }).first()).toBeVisible();
}

async function stageSweep(page: Page) {
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  const matrix = page.getByText('Matrix builder', { exact: true }).locator('xpath=ancestor::section[1]');
  await matrix.getByRole('textbox', { name: /^Path\b/ }).fill('training_spec.optimizer.params.learning_rate');
  await matrix.getByRole('textbox', { name: /^Values\b/ }).fill(sweepRates.join(', '));
  await matrix.getByRole('button', { name: 'Add', exact: true }).click();
  const plan = page.waitForResponse((response) =>
    response.url().endsWith('/api/provider/studio/training/plan') && response.request().method() === 'POST',
    { timeout: 8_000 });
  await matrix.getByRole('button', { name: 'Stage 2 runs', exact: true }).click();
  return plan;
}

test('adds a composite with Train open and reopens the saved graph from another browser context',
  async ({ page, browser, baseURL, request }) => {
    const name = 'Walkthrough authoring';
    await page.goto('/');
    await page.getByTitle('New project', { exact: true }).click();
    await page.getByPlaceholder('Tab name...').fill(name);
    await page.getByPlaceholder('Tab name...').press('Enter');
    await page.getByRole('button', { name: 'Train', exact: true }).click();
    await page.getByRole('button', { name: 'Templates', exact: true }).click();
    const search = page.getByPlaceholder('Search templates...');
    await search.fill('SimpleFeedbackLoop');
    const library = search.locator('xpath=../..');
    const template = library.getByTitle('Add feedbax.SimpleFeedbackLoop', { exact: true });
    const components = await request.get('/api/components');
    expect(components.ok()).toBe(true);
    const { data } = await components.json();
    const { category, template_graph: model } = data.components.find(({ name }: { name: string }) =>
      name === 'feedbax.SimpleFeedbackLoop');
    const count = Object.keys(model.nodes).length;
    const group = library.getByRole('button', { name: category, exact: true });
    await expect(group).toBeVisible();
    if (!await template.isVisible()) {
      await group.click();
    }
    const saving = page.waitForResponse((response) => {
      const request = response.request();
      if (!/\/api\/graphs(?:\/[^/]+)?$/.test(new URL(response.url()).pathname) ||
          !['POST', 'PUT'].includes(request.method())) return false;
      const graph = (request.postDataJSON() as StudioPersistenceDocument).graph;
      return graph?.metadata?.name === name && Object.keys(graph.nodes).length === count;
    });
    await template.click();
    await expect(page.locator('.react-flow__node')).toHaveCount(count);
    await expect(page.getByRole('button', { name: /^Stage / })).toBeVisible();
    await page.getByTitle('Save', { exact: true }).click();
    const save = await saving;
    expect(save.ok(), await save.text()).toBe(true);
    const id = save.request().method() === 'POST'
      ? ((await save.json()) as GraphCreateResponse).data.id
      : new URL(save.url()).pathname.split('/').at(-1);
    const response = await request.get(`/api/graphs/${id}`);
    expect(response.ok()).toBe(true);
    const { data: persisted } = await response.json() as GraphDetailResponse;
    expect(Object.keys(persisted.graph.nodes)).toHaveLength(count);
    const interior = Object.values(persisted.graph.subgraphs ?? {}).find((graph) => 'wires' in graph);
    expect(interior).toBeDefined();

    const fresh = await browser.newPage({ baseURL, serviceWorkers: 'block' });
    try {
      const loaded = fresh.waitForResponse((response) => response.url().endsWith(`/api/graphs/${id}`));
      await fresh.goto('/');
      await openSavedProject(fresh, name);
      expect((await (await loaded).json() as GraphDetailResponse).data.graph).toEqual(persisted.graph);
      await expect(fresh.locator('.react-flow__node')).toHaveCount(count);
    } finally {
      await fresh.close();
    }
  });

test('stages two pending rows at the authored learning rates', async ({ page, request }) => {
  const document = await reachingDocument();
  const { metadata } = await saveFixture(request, document);
  await page.goto('/');
  await openSavedProject(page, metadata.name);
  const plan = await stageSweep(page);
  expect(plan.ok(), await plan.text()).toBe(true);
  const { workspace } = await plan.json() as StudioTrainingExecutionPreparation;
  const rows = workspace.stages.find((stage) => stage.kind === 'train')?.output_collections
    .find((collection) => collection.kind === 'training_runs')?.item_refs;
  const labels = sweepRates.map((rate) => `Learning rate=${rate}`);
  expect(rows?.map((row) => [row.metadata.label, row.metadata.status])).toEqual(
    expect.arrayContaining(labels.map((label) => [label, 'pending'])),
  );
  expect(rows).toHaveLength(labels.length);
  await expect(page.getByRole('button', { name: /^Show details for Learning rate=/ })).toHaveCount(labels.length);
  for (const label of labels) {
    await expect(page.getByRole('button', { name: `Show details for ${label}`, exact: true })).toBeVisible();
  }
});

test('reports the project object that prevents staging', async ({ page, request }) => {
  const document = await reachingDocument();
  document.graph!.metadata!.name = 'Unwired reach';
  const training = document.workspace!.scenarios['scenario:train'];
  training.task_binding_spec!.bindings[0].target_port = 'missing_input';
  const { metadata } = await saveFixture(request, document);
  await page.goto('/');
  await openSavedProject(page, metadata.name);
  const plan = await stageSweep(page);
  expect(plan.status()).toBe(422);
  const problems = page.getByRole('list', { name: 'Problems' });
  await expect(problems.getByRole('listitem').filter({ hasText: /cell\.missing_input/i }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: /^Launch / })).toHaveCount(0);
});

test('opens a readable project beside an older unreadable one', async ({ page, request }, testInfo) => {
  const document = await reachingDocument();
  document.graph!.metadata!.name = 'Readable reach';
  const { id, metadata } = await saveFixture(request, document);
  const root = join(testInfo.config.metadata.home, 'state', 'web', 'graphs');
  const legacyPath = join(root, `${randomUUID()}.json`);
  const project = JSON.parse(await readFile(join(root, `${id}.json`), 'utf8')) as GraphProject;
  const legacyName = 'Legacy reach';
  project.metadata.name = legacyName;
  delete project.graph.nodes.readout.revision;
  await writeFile(legacyPath, JSON.stringify(project), { flag: 'wx' });
  try {
    await page.goto('/');
    await page.getByTitle('Open project', { exact: true }).click();
    const overlay = openOverlay(page);
    const unreadable = overlay.getByText(legacyName, { exact: true });
    await expect(unreadable).toBeVisible();
    await expect(unreadable.locator('xpath=..').getByText(/unversioned.*Linear/i)).toBeVisible();
    await expect(overlay.getByRole('button', { name: new RegExp(`^${legacyName}`) })).toHaveCount(0);
    await overlay.getByRole('button', { name: new RegExp(`^${metadata.name}`) }).click();
    await expect(overlay).toHaveCount(0);
    await expect(page.getByRole('banner').getByText(metadata.name, { exact: true }).first()).toBeVisible();
  } finally {
    await rm(legacyPath, { force: true });
  }
});
