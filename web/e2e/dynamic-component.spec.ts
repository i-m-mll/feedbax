import { expect, test } from '@playwright/test';
import type { GraphSpec, StudioWorkspaceSpec, WorkspaceDocument } from '../src/generated/studioContracts';

test('creates and edits an external component using its served port policy', async ({ page }) => {
  const schema_id = 'feedbax.spec.studio.api_transport';
  const schema_version = 'feedbax.spec.studio.api_transport.v3';
  const transport = (data: object) => ({
    schema_id, schema_version, data: { schema_id, schema_version, ...data },
  });
  const domain = 'feedbax.domain.causal';
  const graphId = 'external-policy';
  const timestamp = '2026-01-01T00:00:00.000Z';
  const metadata = { name: 'External policy', created_at: timestamp, updated_at: timestamp, version: '1.0.0', save_revision: 0 };
  let graph: GraphSpec = {
    schema_id: 'feedbax.spec.graph', schema_version: 'feedbax.spec.graph.v6', metadata,
    nodes: {}, wires: [], input_ports: [], output_ports: [], input_bindings: {}, output_bindings: {},
  };
  let workspace: StudioWorkspaceSpec | null = null;
  let workspace_document: WorkspaceDocument = {
    schema_id: 'feedbax.workspace_document', schema_version: '1',
    semantic_anchors: {}, analysis_pages: [], active_analysis_page_id: null,
    semantic_root: { semantic_document_sha256: 'a'.repeat(64), authored_path: '/graph' },
    graph_ui_state: { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} },
  };
  await page.route('**/api/components', (route) => route.fulfill({ json: transport({
    components: [{
      name: 'example.Fanout',
      revision: 2, supported_revisions: [1, 2],
      category: 'External',
      description: 'External example with configurable output arity.',
      domain,
      param_schema: [{ name: 'slots', type: 'int', default: 1 }],
      default_params: { slots: 1 },
      input_ports: ['value'],
      output_ports: ['branch-4'],
      dynamic_port_policy: {
        direction: 'output',
        count_mode: 'integer',
        count_param: 'slots',
        fixed_input_ports: ['value'],
        generated_name_template: 'branch-{index}',
        generated_index_origin: 4,
        dynamic_port_type: { dtype: 'scalar' },
      },
    }],
  }) }));
  await page.route('**/api/domains', (route) => route.fulfill({ json: transport({
    schema_id: 'feedbax.spec.domain',
    schema_version: 'feedbax.spec.domain.v1',
    domains: [{
      id: domain, display_name: 'Causal', interior_schema_id: 'feedbax.spec.graph',
      edge_semantics: 'directed', allows_multi_edge_per_port: false, nestable_domains: [domain],
      editor: { kind: 'canvas', editable: true }, theme: { color: 'causal', icon: 'Layers', edge_style: 'directed' },
      compiler_id: null,
    }],
  }) }));
  await page.route('**/api/graphs', (route) => route.fulfill({ json: transport({
    graphs: [{ schema_id, schema_version, id: graphId, metadata }],
  }) }));
  await page.route(`**/api/graphs/${graphId}`, async (route) => {
    if (route.request().method() === 'PUT') {
      const update = route.request().postDataJSON();
      graph = update.graph;
      workspace = update.workspace;
      workspace_document = update.workspace_document;
      metadata.save_revision += 1;
      return route.fulfill({ json: transport({ success: true, metadata }) });
    }
    return route.fulfill({ json: transport({ graph, metadata, workspace, workspace_document }) });
  });
  await page.addInitScript((id) => {
    localStorage.removeItem('feedbax:studio-local-tabs');
    localStorage.setItem('feedbax:lastProjectId', id);
  }, graphId);

  await page.goto('/');
  await expect.poll(() => page.evaluate(() => window.feedbaxE2E?.graphId())).toBe(graphId);
  await page.getByRole('button', { name: 'External', exact: true }).click();
  await page.getByTitle('Add example.Fanout', { exact: true }).click();
  const nodeId = 'fanout';
  const node = page.locator(`.react-flow__node[data-id="${nodeId}"]`);
  await expect(node.getByRole('button', { name: 'branch-4', exact: true })).toBeVisible();
  await node.click();
  const slots = page.getByRole('textbox', { name: /^slots$/i });
  await expect(slots).toHaveValue('1');
  const saved = page.waitForResponse((response) =>
    response.url().endsWith(`/api/graphs/${graphId}`)
      && response.request().method() === 'PUT'
      && response.request().postDataJSON().graph?.nodes[nodeId]?.params.slots === 2
  );
  await slots.fill('2');
  await slots.press('Enter');
  await expect(node.getByRole('button', { name: 'branch-5', exact: true })).toBeVisible();
  await page.getByTitle('Save', { exact: true }).click();
  expect((await saved).ok()).toBe(true);
  expect(graph.nodes[nodeId]).toMatchObject({ type: 'example.Fanout', revision: 2, params: { slots: 2 } });
  await page.reload();
  await expect(node.getByRole('button', { name: 'branch-5', exact: true })).toBeVisible();
});
