// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactFlowProvider } from '@xyflow/react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { compileGraphNode } from '@/api/client';
import { StatusBar } from '@/components/layout/StatusBar';
import { ValidationPanel } from '@/components/panels/ValidationPanel';
import { stableGraphHash } from '@/features/domains/acausal';
import type { AcausalGraphSpec, DomainCompileReport } from '@/generated/studioContracts';
import { useCompileStatusStore } from '@/stores/compileStatusStore';
import { createBlankGraph, useGraphStore } from '@/stores/graphStore';
import { useProjectsStore } from '@/stores/projectsStore';
import { Canvas } from './Canvas';

vi.mock('@/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/client')>()),
  compileGraphNode: vi.fn(),
}));

const compile = vi.mocked(compileGraphNode);
const path = ['system'];
const components = [{
  name: "feedbax.AcausalSystem",
  category: "Mechanics",
  description: "Acausal interior",
  input_ports: [],
  output_ports: [],
  default_params: {},
  interior_domain: "feedbax.domain.acausal",
}];


function makeInterior(): AcausalGraphSpec {
  return {
    schema_id: 'feedbax.spec.acausal_graph',
    schema_version: 'feedbax.spec.acausal_graph.v2',
    physical_domain: 'translational',
    nodes: {},
    connections: [],
    solver: { solver_type: 'implicit_euler', dt: 0.01 },
  };
}

const acausalSystem = { type: 'feedbax.AcausalSystem', params: {}, input_ports: [], output_ports: [] };

function openDocument(id: string) {
  const graph = createBlankGraph();
  graph.nodes.system = acausalSystem;
  graph.subgraphs = { system: makeInterior() };
  const tabId = useProjectsStore.getState().openProjectInTab(id, graph, {});
  useGraphStore.getState().enterSubgraph('system');
  return tabId;
}

function renderEditor() {
  const client = new QueryClient();
  client.setQueryData(["components"], components);
  client.setQueryData(['domains'], []);
  render(
    <QueryClientProvider client={client}>
      <ReactFlowProvider>
        <Canvas />
        <ValidationPanel />
        <StatusBar />
      </ReactFlowProvider>
    </QueryClientProvider>,
  );
}

function reportFor(interior: AcausalGraphSpec, diagnostic?: string): DomainCompileReport {
  return {
    schema_id: 'feedbax.spec.domain_compile_report',
    schema_version: 'feedbax.spec.domain_compile_report.v1',
    interior_content_hash: stableGraphHash(interior),
    status: diagnostic ? 'error' : 'ok',
    diagnostics: diagnostic ? [{ severity: 'error', code: 'test.diagnostic', message: diagnostic }] : [],
    summary: { equations: diagnostic ? 13 : 2, unknowns: 2 },
  };
}

async function elapseDebounce() {
  await act(async () => vi.advanceTimersByTimeAsync(500));
}

function changeTimestep(dt: number) {
  act(() => {
    const graph = useGraphStore.getState().graph as AcausalGraphSpec;
    useGraphStore.setState({ graph: { ...graph, solver: { solver_type: 'implicit_euler', dt } } });
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  compile.mockReset();
  useGraphStore.getState().setComponentRegistry(components);

});

afterEach(() => {
  cleanup();
  useCompileStatusStore.setState(useCompileStatusStore.getInitialState());
  useProjectsStore.setState(useProjectsStore.getInitialState());
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

it('compiles two tabs independently and shows diagnostics only in their owning tab', async () => {
  let finishOutgoing!: (report: DomainCompileReport) => void;
  compile.mockImplementationOnce(() => new Promise((resolve) => { finishOutgoing = resolve; }));
  compile.mockImplementation(async (_id, _path, interior) => reportFor(interior));
  const outgoing = openDocument('outgoing');
  renderEditor();
  await elapseDebounce();
  expect(compile.mock.calls.map(([id, layer]) => [id, layer])).toEqual([['outgoing', path]]);
  const outgoingReport = reportFor(compile.mock.calls[0][2], 'Outgoing diagnostic');

  act(() => { openDocument('incoming'); });
  await elapseDebounce();
  expect.soft(compile.mock.calls.map(([id, layer]) => [id, layer])).toEqual([
    ['outgoing', path],
    ['incoming', path],
  ]);

  await act(async () => finishOutgoing(outgoingReport));
  expect.soft(screen.queryByText('Outgoing diagnostic')).toBeNull();
  expect.soft(screen.queryByText('eq 13 / unk 2')).toBeNull();
  await elapseDebounce();
  expect.soft(compile).toHaveBeenCalledTimes(2);

  act(() => useProjectsStore.getState().switchTab(outgoing));
  expect(screen.getByText('Outgoing diagnostic')).toBeVisible();
  expect(screen.getByText('eq 13 / unk 2')).toBeVisible();

  changeTimestep(0.02);
  expect(screen.getByText('stale')).toBeVisible();
  expect(screen.getByText('Outgoing diagnostic')).toBeVisible();
});

it('attempts an unchanged failed draft once, then allows retry and a semantic edit', async () => {
  const failed = new Error('Compile request failed');
  compile.mockRejectedValue(failed);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  openDocument('failing');
  renderEditor();
  await elapseDebounce();
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(1);

  fireEvent.click(screen.getByRole('button', { name: 'Retry compile' }));
  expect(compile).toHaveBeenCalledTimes(2);
  await act(async () => {});
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(2);

  changeTimestep(0.02);
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(3);
});

it('compiles a semantic edit while the older draft is pending and keeps the newer result', async () => {
  let finishOlder!: (report: DomainCompileReport) => void;
  compile.mockImplementationOnce(() => new Promise((resolve) => { finishOlder = resolve; }));
  compile.mockImplementation(async (_id, _path, interior) => reportFor(interior));
  openDocument('editing');
  renderEditor();
  await elapseDebounce();

  changeTimestep(0.02);
  await elapseDebounce();
  expect(compile.mock.calls.map(([, , interior]) => interior.solver?.dt)).toEqual([0.01, 0.02]);

  const olderReport = reportFor(compile.mock.calls[0][2], 'Older draft diagnostic');
  await act(async () => finishOlder(olderReport));
  expect(screen.queryByText('Older draft diagnostic')).toBeNull();
  expect(screen.getByText('eq 2 / unk 2')).toBeVisible();
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(2);
});

it('uses the matching in-flight result after editing and reverting the draft', async () => {
  let finishOriginal!: (report: DomainCompileReport) => void;
  compile.mockImplementationOnce(() => new Promise((resolve) => { finishOriginal = resolve; }));
  compile.mockImplementation(async (_id, _path, interior) => reportFor(interior));
  openDocument('reverting');
  renderEditor();
  await elapseDebounce();

  changeTimestep(0.02);
  await elapseDebounce();
  changeTimestep(0.01);
  expect(screen.getByText('Compiling…')).toBeVisible();

  const original = reportFor(compile.mock.calls[0][2], 'Reverted draft diagnostic');
  await act(async () => finishOriginal(original));
  expect(screen.getByText('Reverted draft diagnostic')).toBeVisible();
  expect(screen.getByText('eq 13 / unk 2')).toBeVisible();
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(2);
});

it('skips ambiguous saved reports and compiles slash-containing and nested paths separately', async () => {
  compile.mockImplementation(async (_id, layer, interior) => reportFor(interior, JSON.stringify(layer)));
  const graph = createBlankGraph();
  const direct = makeInterior();
  const nested = makeInterior();
  graph.nodes = { 'a/b': acausalSystem, a: acausalSystem, ordinary: acausalSystem };
  graph.subgraphs = {
    'a/b': direct,
    a: { ...makeInterior(), nodes: { b: acausalSystem }, subgraphs: { b: nested } },
  };
  const documentId = useProjectsStore.getState().openProjectInTab('paths', graph, {});
  useCompileStatusStore.getState().setReports(documentId, graph, {
    'a/b': reportFor(direct, 'Ambiguous diagnostic'),
    ordinary: reportFor(direct, 'Ordinary component diagnostic'),
  });
  useGraphStore.getState().enterSubgraph('a/b');
  renderEditor();
  expect(screen.queryByText('Ambiguous diagnostic')).toBeNull();
  expect(screen.getByText('Ordinary component diagnostic')).toBeVisible();
  await elapseDebounce();

  act(() => {
    const editor = useGraphStore.getState();
    editor.exitToBreadcrumb(0);
    editor.enterSubgraph('a');
    editor.enterSubgraph('b');
  });
  await elapseDebounce();
  expect(compile.mock.calls.map(([, layer]) => layer)).toEqual([['a/b'], ['a', 'b']]);
  expect(screen.getByText('["a/b"]')).toBeVisible();
  expect(screen.getByText('["a","b"]')).toBeVisible();
});

it('does not auto-retry when a reverted in-flight draft fails', async () => {
  let failOriginal!: (error: Error) => void;
  let failNewer!: (error: Error) => void;
  compile.mockImplementationOnce(() => new Promise((_resolve, reject) => { failOriginal = reject; }));
  compile.mockImplementationOnce(() => new Promise((_resolve, reject) => { failNewer = reject; }));
  openDocument('reverting-failure');
  renderEditor();
  await elapseDebounce();

  changeTimestep(0.02);
  await elapseDebounce();
  changeTimestep(0.01);
  await act(async () => failOriginal(new Error('Compile request failed')));
  await act(async () => failNewer(new Error('Newer request failed')));
  expect(screen.getByRole('button', { name: 'Retry compile' })).toBeVisible();
  await elapseDebounce();
  expect(compile).toHaveBeenCalledTimes(2);
});

it('shows a local error and sends no request for an unhashable draft', async () => {
  openDocument('unhashable');
  renderEditor();
  changeTimestep(Infinity);

  expect(screen.getByText(/finite/i)).toBeVisible();
  await elapseDebounce();
  expect(compile).not.toHaveBeenCalled();
});
