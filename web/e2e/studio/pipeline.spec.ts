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

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);

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

async function openSavedProject(page: Page, name: string) {
  await page.getByTitle('Open project', { exact: true }).click();
  await page.getByRole('button', { name: new RegExp(`^${RegExp.escape(name)}(?: |$)`) }).click();
  await expect(page.getByRole('button', { name: 'Projects', exact: true })).toHaveCount(0);
  await expect(page.getByRole('banner').getByText(name, { exact: true }).first()).toBeVisible();
}

async function stageSweep(page: Page) {
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await page.getByLabel('Path', { exact: true }).fill('training_spec.optimizer.params.learning_rate');
  await page.getByLabel('Values', { exact: true }).fill('0.001, 0.0003');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const plan = page.waitForResponse((response) =>
    response.url().endsWith('/api/provider/studio/training/plan') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Stage 2 runs', exact: true }).click();
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
    await page.getByPlaceholder('Search templates...').fill('SimpleFeedbackLoop');
    const template = page.getByTitle('Add feedbax.SimpleFeedbackLoop', { exact: true });
    const components = await request.get('/api/components');
    expect(components.ok()).toBe(true);
    const { data } = await components.json();
    const category = data.components.find(({ name }: { name: string }) =>
      name === 'feedbax.SimpleFeedbackLoop').category;
    const group = page.getByRole('button', { name: category, exact: true });
    await expect(group).toBeVisible();
    if (!await template.isVisible()) {
      await group.click();
    }
    await template.click();
    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(1);
    await expect(page.getByRole('button', { name: /^Stage / })).toBeVisible();

    const saving = page.waitForResponse((response) => {
      const request = response.request();
      if (!/\/api\/graphs(?:\/[^/]+)?$/.test(new URL(response.url()).pathname) ||
          !['POST', 'PUT'].includes(request.method())) return false;
      const graph = (request.postDataJSON() as StudioPersistenceDocument).graph;
      return graph?.metadata?.name === name && Object.keys(graph.nodes).length === 1;
    });
    await page.getByTitle('Save', { exact: true }).click();
    const save = await saving;
    expect(save.ok(), await save.text()).toBe(true);
    const id = save.request().method() === 'POST'
      ? ((await save.json()) as GraphCreateResponse).data.id
      : new URL(save.url()).pathname.split('/').at(-1);
    const response = await request.get(`/api/graphs/${id}`);
    expect(response.ok()).toBe(true);
    const { data: persisted } = await response.json() as GraphDetailResponse;
    expect(Object.keys(persisted.graph.nodes)).toHaveLength(1);
    const interior = Object.values(persisted.graph.subgraphs ?? {}).find((graph) => 'wires' in graph);
    expect(interior).toBeDefined();

    const fresh = await browser.newPage({ baseURL, serviceWorkers: 'block' });
    try {
      await fresh.goto('/');
      const loaded = fresh.waitForResponse((response) => response.url().endsWith(`/api/graphs/${id}`));
      await openSavedProject(fresh, name);
      expect((await (await loaded).json() as GraphDetailResponse).data.graph).toEqual(persisted.graph);
      await expect(fresh.locator('.react-flow__node')).toHaveCount(1);
    } finally {
      await fresh.close();
    }
  });

test('stages two real rows for the CPU reaching project', async ({ page, request }) => {
  const { metadata } = await saveFixture(request, await reachingDocument());
  await page.goto('/');
  await openSavedProject(page, metadata.name);
  const plan = await stageSweep(page);
  expect(plan.ok(), await plan.text()).toBe(true);
  await expect(page.getByRole('button', { name: /^Launch / })).toHaveCount(2);
  for (const launch of await page.getByRole('button', { name: /^Launch / }).all()) {
    await expect(launch).toBeEnabled();
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
  await expect(problems.getByRole('listitem').filter({ hasText: /cell.*missing_input/i })).toBeVisible();
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
    const unreadable = page.getByText(legacyName, { exact: true });
    await expect(unreadable).toBeVisible();
    await expect(page.getByText(/unversioned.*Linear/i)).toBeVisible();
    await expect(page.getByRole('button', { name: new RegExp(`^${legacyName}`) })).toHaveCount(0);
    await page.getByRole('button', { name: new RegExp(`^${metadata.name}`) }).click();
    await expect(page.getByRole('button', { name: 'Projects', exact: true })).toHaveCount(0);
    await expect(page.getByRole('banner').getByText(metadata.name, { exact: true }).first()).toBeVisible();
  } finally {
    await rm(legacyPath, { force: true });
  }
});
