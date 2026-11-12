import { expect, test } from '@playwright/test';
import { randomUUID } from 'node:crypto';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type {
  GraphCreateResponse,
  GraphDetailResponse,
  GraphProject,
  StudioPersistenceDocument,
} from '../../src/generated/studioContracts';
import { openOverlay, openSavedProject } from './projectControls';

import { reachingDocument, saveFixture } from './reachingProject';

test('adds a composite with Train open and reopens the saved graph from another browser context',
  async ({ page, browser, baseURL, request }) => {
    const name = 'Walkthrough authoring';
    await page.goto('/');
    await page.getByRole('button', { name: 'New project', exact: true }).click();
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
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Save', exact: true }).click();
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
    await page.getByRole('button', { name: 'Open project', exact: true }).click();
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
