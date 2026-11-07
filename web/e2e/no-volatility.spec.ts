import { expect, test, type Page } from '@playwright/test';
import type { StudioWorkspaceSpec, WorkspaceDocument } from '../src/generated/studioContracts';

const TRANSPORT_SCHEMA = 'feedbax.spec.studio.api_transport';
const TRANSPORT_VERSION = 'feedbax.spec.studio.api_transport.v3';
const LOCAL_TABS_KEY = 'feedbax:studio-local-tabs';
const LAST_PROJECT_KEY = 'feedbax:lastProjectId';
const CAUSAL_DOMAIN_ID = 'feedbax.domain.causal';
const ACAUSAL_DOMAIN_ID = 'feedbax.domain.acausal';

function metadata(name: string) {
  return {
    name,
    created_at: '2026-01-01T00:00:00.000Z',
    updated_at: '2026-01-01T00:00:00.000Z',
    version: '1.0.0',
    save_revision: 0,
  };
}

function transport<T extends Record<string, unknown>>(data: T) {
  return {
    schema_id: TRANSPORT_SCHEMA,
    schema_version: TRANSPORT_VERSION,
    data: {
      schema_id: TRANSPORT_SCHEMA,
      schema_version: TRANSPORT_VERSION,
      ...data,
    },
  };
}

function graphFixture(hiddenSize: number) {
  return {
    schema_id: 'feedbax.spec.graph',
    schema_version: 'feedbax.spec.graph.v6',
    nodes: {
      cell: {
        type: 'GRUCell',
        params: {
          input_size: 6,
          hidden_size: hiddenSize,
        },
        input_ports: ['input', 'hidden'],
        output_ports: ['hidden'],
      },
      system: {
        type: 'feedbax.AcausalSystem',
        params: {},
        input_ports: ['input'],
        output_ports: ['state'],
      },
    },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    subgraphs: {
      system: {
        schema_id: 'feedbax.spec.acausal_graph',
        schema_version: 'feedbax.spec.acausal_graph.v1',
        physical_domain: 'translational',
        nodes: {},
        connections: [],
        solver: { solver_type: 'euler', dt: 0.01 },
      },
    },
    metadata: metadata('No volatility smoke'),
  };
}

async function installMockApi(page: Page, initialWorkspace: StudioWorkspaceSpec | null = null) {
  page.on('console', (message) => {
    if (message.type() === 'error') console.error(message.text());
  });
  page.on('pageerror', (error) => console.error(error));
  let savedGraph = graphFixture(100);
  const uiState: any = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      cell: {
        position: { x: 200, y: 120 },
        collapsed: false,
        selected: false,
      },
      system: {
        position: { x: 480, y: 120 },
        collapsed: false,
        selected: false,
      },
    },
    subgraph_states: {
      system: {
        viewport: { x: 0, y: 0, zoom: 1 },
        node_states: {},
      },
    },
  };
  let savedWorkspace = initialWorkspace;
  let savedDocument: WorkspaceDocument = {
    schema_id: 'feedbax.workspace_document',
    schema_version: '1',
    semantic_root: {
      semantic_document_sha256: 'a'.repeat(64),
      authored_path: '/graph',
    },
    semantic_anchors: {},
    graph_ui_state: uiState,
    analysis_pages: [],
    active_analysis_page_id: null,
  };

  await page.route('**/api/components', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify(transport({
        components: [
          {
            name: 'GRUCell',
            category: 'Networks',
            description: 'Recurrent cell',
            param_schema: [
              { name: 'input_size', type: 'int', default: 6 },
              { name: 'hidden_size', type: 'int', default: 100 },
            ],
            input_ports: ['input', 'hidden'],
            output_ports: ['hidden'],
            default_params: { input_size: 6, hidden_size: 100 },
          },
          {
            name: 'feedbax.Subgraph',
            category: 'Structure',
            description: 'Nested causal graph',
            param_schema: [],
            input_ports: ['input'],
            output_ports: ['output'],
            icon: 'Layers',
            default_params: {},
            domain: CAUSAL_DOMAIN_ID,
            interior_domain: CAUSAL_DOMAIN_ID,
            is_composite: true,
          },
          {
            name: 'feedbax.AcausalSystem',
            category: 'Mechanics',
            description: 'Assembled acausal system',
            param_schema: [],
            input_ports: ['input'],
            output_ports: ['state'],
            icon: 'Cog',
            default_params: {},
            domain: CAUSAL_DOMAIN_ID,
            interior_domain: ACAUSAL_DOMAIN_ID,
            is_composite: true,
          },
        ],
      })),
    });
  });

  await page.route('**/api/domains', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify(transport({
        schema_id: 'feedbax.spec.domain',
        schema_version: 'feedbax.spec.domain.v1',
        domains: [
          {
            id: CAUSAL_DOMAIN_ID,
            display_name: 'Causal',
            interior_schema_id: 'feedbax.spec.graph',
            edge_semantics: 'directed',
            allows_multi_edge_per_port: false,
            nestable_domains: [CAUSAL_DOMAIN_ID, ACAUSAL_DOMAIN_ID],
            editor: { kind: 'canvas', editable: true },
            theme: { color: 'causal', icon: 'Layers', edge_style: 'directed' },
            compiler_id: null,
          },
          {
            id: ACAUSAL_DOMAIN_ID,
            display_name: 'Acausal',
            interior_schema_id: 'feedbax.spec.acausal_graph',
            edge_semantics: 'undirected',
            allows_multi_edge_per_port: true,
            nestable_domains: [ACAUSAL_DOMAIN_ID],
            editor: { kind: 'canvas', editable: true },
            theme: { color: 'acausal', icon: 'Cog', edge_style: 'undirected' },
            compiler_id: 'feedbax.compiler.acausal',
          },
        ],
      })),
    });
  });

  await page.route('**/api/provider/studio/schemas', async (route) => {
    const { graph } = route.request().postDataJSON();
    expect(graph.schema_id).not.toBe('feedbax.spec.acausal_graph');
    expect(Array.isArray(graph.wires)).toBe(true);
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        kind: 'studio_schema_registry',
        schema_version: 'feedbax.studio.schema_registry.v1',
        generated_at: '2026-01-01T00:00:00.000Z',
        workspace_id: null,
        scenario_id: null,
        ports: [],
        task_data: [],
        selector_targets: [],
        issues: [],
        metadata: {},
      }),
    });
  });

  await page.route('**/api/graphs', async (route) => {
    if (route.request().method() !== 'GET') {
      return route.fallback();
    }
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify(
        transport({
          graphs: [
            {
              schema_id: TRANSPORT_SCHEMA,
              schema_version: TRANSPORT_VERSION,
              id: 'graph-1',
              metadata: metadata('No volatility smoke'),
            },
          ],
        }),
      ),
    });
  });

  await page.route('**/api/graphs/graph-1', async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify(
          transport({
            graph: savedGraph,
            workspace_document: savedDocument,
            metadata: savedGraph.metadata,
            workspace: savedWorkspace,
            demo_training_data: null,
          }),
        ),
      });
      return;
    }

    if (route.request().method() === 'PUT') {
      const payload = route.request().postDataJSON() as {
        graph?: typeof savedGraph;
        workspace?: StudioWorkspaceSpec;
        workspace_document?: WorkspaceDocument;
      };
      if (payload.graph) {
        savedGraph = payload.graph;
      }
      savedWorkspace = payload.workspace ?? savedWorkspace;
      savedDocument = payload.workspace_document ?? savedDocument;
      savedGraph.metadata.save_revision += 1;
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify(transport({ success: true, metadata: savedGraph.metadata })),
      });
      return;
    }

    await route.fallback();
  });

  return () => ({ graph: savedGraph, workspace: savedWorkspace });
}

test('parameter editing preserves geometry and survives save/reload', async ({ page }) => {
  await installMockApi(page);
  await page.addInitScript(
    ({ localTabsKey, lastProjectKey }) => {
      localStorage.removeItem(localTabsKey);
      localStorage.setItem(lastProjectKey, 'graph-1');
    },
    { localTabsKey: LOCAL_TABS_KEY, lastProjectKey: LAST_PROJECT_KEY },
  );

  await page.goto('/');
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.graphId()))
    .toBe('graph-1');
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.nodeParam('cell', 'hidden_size')))
    .toBe(100);

  await page.locator('.react-flow__node[data-id="cell"]').click();
  const hiddenSize = page.getByRole('textbox', { name: /^hidden size$/i });
  await expect(hiddenSize).toHaveValue('100');
  await hiddenSize.scrollIntoViewIfNeeded();
  await expect(hiddenSize).toBeInViewport({ ratio: 1 });
  const nodeLabel = page.getByRole('textbox', { name: 'Node label', exact: true });
  const fieldBounds = await hiddenSize.boundingBox();
  const labelBounds = await nodeLabel.boundingBox();
  expect(fieldBounds).not.toBeNull();
  expect(labelBounds).not.toBeNull();
  const expectStableGeometry = async () => {
    expect(await hiddenSize.boundingBox()).toEqual(fieldBounds);
    expect(await nodeLabel.boundingBox()).toEqual(labelBounds);
  };

  await hiddenSize.hover();
  await expectStableGeometry();
  await hiddenSize.focus();
  await expect(hiddenSize).toBeFocused();
  await expectStableGeometry();
  await hiddenSize.fill('128');
  await expect(hiddenSize).toHaveValue('128');
  await expectStableGeometry();

  const saved = page.waitForResponse((response) =>
    response.url().endsWith('/api/graphs/graph-1')
    && response.request().method() === 'PUT'
    && response.request().postDataJSON().graph?.nodes.cell.params.hidden_size === 128
  );
  await hiddenSize.press('Enter');
  await expect(hiddenSize).not.toBeFocused();
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.nodeParam('cell', 'hidden_size')))
    .toBe(128);
  await expectStableGeometry();
  await page.getByTitle('Save', { exact: true }).click();
  expect((await saved).ok()).toBe(true);

  await page.evaluate((localTabsKey) => localStorage.removeItem(localTabsKey), LOCAL_TABS_KEY);
  await page.reload();

  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.graphId()))
    .toBe('graph-1');
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.nodeParam('cell', 'hidden_size')))
    .toBe(128);
  await page.locator('.react-flow__node[data-id="cell"]').click();
  await expect(page.getByRole('textbox', { name: /^hidden size$/i })).toHaveValue('128');
});

test('keyboard history restores a deleted model node and its task binding together', async ({ page }) => {
  const scenarioId = 'scenario:train';
  const binding = {
    id: 'task:inputs->cell:hidden',
    source_data_id: 'inputs',
    target_node_id: 'cell',
    target_port: 'hidden',
    role: 'model_input',
    metadata: { label: 'Recurrent input' },
  };
  const savedProject = await installMockApi(page, {
    id: 'workspace-1',
    schema_id: 'feedbax.spec.studio.workspace',
    schema_version: 'feedbax.spec.studio.workspace.v2',
    label: 'Bound node history',
    active_stage_id: 'stage:train',
    stages: [],
    metadata: {},
    collections: [],
    manifest_refs: [],
    artifact_refs: [],
    scenarios: {
      [scenarioId]: {
        id: scenarioId,
        label: 'Training scenario',
        stage_id: 'stage:train',
        task_binding_spec: { exposed_data: [], bindings: [binding] },
      },
    },
  });
  await page.addInitScript(
    ({ localTabsKey, lastProjectKey }) => {
      localStorage.removeItem(localTabsKey);
      localStorage.setItem(lastProjectKey, 'graph-1');
    },
    { localTabsKey: LOCAL_TABS_KEY, lastProjectKey: LAST_PROJECT_KEY },
  );

  await page.goto('/');
  const node = page.locator('.react-flow__node[data-id="cell"]');
  const task = page.getByRole('button', { name: 'Task', exact: true });
  const model = page.getByRole('button', { name: 'Model', exact: true });
  const bindingSource = page.locator(`[data-task-data-port-id="${binding.source_data_id}"]`);
  const bindingPath = page.locator(`[data-task-binding-visual-id="${binding.id}"]`).last();
  await expect(node).toBeVisible();
  await task.click();
  await expect(bindingPath).toHaveAttribute('d', /^M /);
  await model.click();
  await node.click();

  const deleteWithConfirmation = async (key: 'Backspace' | 'Delete', accept: boolean) => {
    const confirmDeletion = page.waitForEvent('dialog').then(async (dialog) => {
      if (accept) await dialog.accept();
      else await dialog.dismiss();
      expect(dialog.message()).toContain('1 task binding');
    });
    await Promise.all([page.keyboard.press(key), confirmDeletion]);
  };

  await deleteWithConfirmation('Backspace', false);
  await expect(node).toBeVisible();
  await deleteWithConfirmation('Delete', true);
  await expect(node).toHaveCount(0);
  await task.click();
  await expect(bindingSource).toBeVisible();
  await expect(bindingPath).toHaveCount(0);

  const savedNodeAndBindings = () => {
    const { graph, workspace } = savedProject();
    return {
      node: graph.nodes.cell,
      bindings: workspace?.scenarios?.[scenarioId]?.task_binding_spec?.bindings,
    };
  };
  const save = page.getByTitle('Save', { exact: true });
  const deleted = { node: undefined, bindings: [] };
  await save.click();
  await expect.poll(savedNodeAndBindings).toEqual(deleted);

  await page.keyboard.press('ControlOrMeta+z');
  await expect(node).toBeVisible();
  await expect(bindingPath).toHaveAttribute('d', /^M /);
  await save.click();
  await expect.poll(savedNodeAndBindings).toEqual({
    node: graphFixture(100).nodes.cell,
    bindings: [binding],
  });

  await page.keyboard.press('ControlOrMeta+Shift+z');
  await expect(node).toHaveCount(0);
  await expect(bindingPath).toHaveCount(0);
  await save.click();
  await expect.poll(savedNodeAndBindings).toEqual(deleted);

  await page.reload();
  await expect(page.locator('.react-flow__node[data-id="system"]')).toBeVisible();
  await task.click();
  await expect(bindingSource).toBeVisible();
  await expect(node).toHaveCount(0);
  await expect(bindingPath).toHaveCount(0);
});

test('reload restores acausal domain context and rejects causal subgraph drops', async ({ page }) => {
  await installMockApi(page);
  await page.addInitScript(
    ({ localTabsKey, lastProjectKey }) => {
      localStorage.removeItem(localTabsKey);
      localStorage.setItem(lastProjectKey, 'graph-1');
    },
    { localTabsKey: LOCAL_TABS_KEY, lastProjectKey: LAST_PROJECT_KEY },
  );

  await page.goto('/');
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.graphId()))
    .toBe('graph-1');

  await page.getByRole('button', { name: 'Task', exact: true }).click();
  const taskHandles = page.locator('[data-task-data-port-id]');
  await expect(taskHandles).not.toHaveCount(0);
  await page.evaluate(() => window.feedbaxE2E?.enterSubgraph('system'));
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.currentContext()))
    .toBe(ACAUSAL_DOMAIN_ID);
  await expect(taskHandles).toHaveCount(0);
  await expect(page.getByTitle('Wrap this graph in a new parent')).toBeDisabled();

  const autosave = page.waitForResponse(
    (response) =>
      response.url().endsWith('/api/graphs/graph-1') &&
      response.request().method() === 'PUT'
  );
  await page.evaluate(() => window.feedbaxE2E?.markDirty());
  await autosave;
  await page.evaluate((localTabsKey) => localStorage.removeItem(localTabsKey), LOCAL_TABS_KEY);
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.currentContext()))
    .toBe(ACAUSAL_DOMAIN_ID);

  await page.evaluate(() => {
    const target = document.querySelector('.react-flow');
    if (!target) throw new Error('React Flow canvas not found');
    const dataTransfer = new DataTransfer();
    dataTransfer.setData('application/feedbax-component', 'feedbax.Subgraph');
    target.dispatchEvent(new DragEvent('drop', {
      bubbles: true,
      cancelable: true,
      clientX: 320,
      clientY: 260,
      dataTransfer,
    }));
  });

  await expect(page.getByText(
    "Acausal interiors accept acausal-domain components only; 'feedbax.Subgraph' is causal."
  )).toBeVisible();
  await page.evaluate(() => window.feedbaxE2E?.exitToRoot());
  await expect(taskHandles).not.toHaveCount(0);
  await expect(page.getByTitle('Wrap this graph in a new parent')).toBeEnabled();
});
