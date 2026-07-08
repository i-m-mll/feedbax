import { expect, test, type Page } from '@playwright/test';

const TRANSPORT_SCHEMA = 'feedbax.spec.studio.api_transport';
const TRANSPORT_VERSION = 'feedbax.spec.studio.api_transport.v1';
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
        type: 'AcausalSystem',
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
        nodes: {},
        wires: [],
        input_ports: [],
        output_ports: [],
        input_bindings: {},
        output_bindings: {},
      },
    },
    metadata: metadata('No volatility smoke'),
  };
}

async function installMockApi(page: Page) {
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
  let savedWorkspace: any = null;

  await page.route('**/api/components', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify(transport({
        components: [
          {
            name: 'Subgraph',
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
            name: 'AcausalSystem',
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
            ui_state: uiState,
            metadata: metadata('No volatility smoke'),
            analysis_pages: null,
            active_analysis_page_id: null,
            workspace: savedWorkspace,
            demo_training_data: null,
          }),
        ),
      });
      return;
    }

    if (route.request().method() === 'PUT') {
      const payload = route.request().postDataJSON() as { graph?: typeof savedGraph };
      if (payload.graph) {
        savedGraph = payload.graph;
      }
      savedWorkspace = (payload as { workspace?: any }).workspace ?? savedWorkspace;
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify(transport({ success: true })),
      });
      return;
    }

    await route.fallback();
  });
}

test('save/reload persists an edited graph parameter', async ({ page }) => {
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

  await page.evaluate(() => {
    window.feedbaxE2E?.updateNodeParam('cell', 'hidden_size', 128);
  });
  await page.getByTitle('Save').click();
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.nodeParam('cell', 'hidden_size')))
    .toBe(128);

  await page.evaluate((localTabsKey) => localStorage.removeItem(localTabsKey), LOCAL_TABS_KEY);
  await page.reload();

  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.graphId()))
    .toBe('graph-1');
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.nodeParam('cell', 'hidden_size')))
    .toBe(128);
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

  await page.evaluate(() => window.feedbaxE2E?.enterSubgraph('system'));
  await expect
    .poll(() => page.evaluate(() => window.feedbaxE2E?.currentContext()))
    .toBe(ACAUSAL_DOMAIN_ID);

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
    dataTransfer.setData('application/feedbax-component', 'Subgraph');
    target.dispatchEvent(new DragEvent('drop', {
      bubbles: true,
      cancelable: true,
      clientX: 320,
      clientY: 260,
      dataTransfer,
    }));
  });

  await expect(page.getByText(
    "Acausal interiors accept acausal-domain components only; 'Subgraph' is causal."
  )).toBeVisible();
});
