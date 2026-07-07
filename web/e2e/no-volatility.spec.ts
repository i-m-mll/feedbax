import { expect, test, type Page } from '@playwright/test';

const TRANSPORT_SCHEMA = 'feedbax.spec.studio.api_transport';
const TRANSPORT_VERSION = 'feedbax.spec.studio.api_transport.v1';
const LOCAL_TABS_KEY = 'feedbax:studio-local-tabs';
const LAST_PROJECT_KEY = 'feedbax:lastProjectId';

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
    },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    metadata: metadata('No volatility smoke'),
  };
}

async function installMockApi(page: Page) {
  let savedGraph = graphFixture(100);
  const uiState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      cell: {
        position: { x: 200, y: 120 },
        collapsed: false,
        selected: false,
      },
    },
  };

  await page.route('**/api/components', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify(transport({ components: [] })),
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
            workspace: null,
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
