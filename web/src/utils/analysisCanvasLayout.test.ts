import { describe, expect, it } from 'vitest';
import type { WorkspaceDocument } from '@/generated/studioContracts';
import type { AnalysisGraphSpec, AnalysisSnapshot } from '@/types/analysis';
import {
  ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
  ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
  analysisSnapshotFromWorkspaceDocument,
  buildAnalysisCanvasLayoutDocument,
} from '@/utils/analysisCanvasLayout';

const graphSpec: AnalysisGraphSpec = {
  nodes: {
    'analysis:a': {
      id: 'analysis:a',
      type: 'ActivityPlot',
      label: 'Activity plot',
      category: 'Figures',
      inputPorts: [],
      outputPorts: [],
      params: {},
      role: 'analysis',
    },
  },
  wires: [],
  dataSourceId: '__data_source__',
};

function document(layout: WorkspaceDocument['analysis_canvas_layout']): WorkspaceDocument {
  return {
    schema_id: 'feedbax.workspace_document',
    schema_version: '1',
    semantic_root: {
      semantic_document_sha256: 'a'.repeat(64),
      authored_path: '/graph',
    },
    analysis_pages: [{
      id: 'page:a',
      name: 'Page A',
      graph_spec: graphSpec as unknown as Record<string, unknown>,
      eval_params: {},
      eval_run_id: null,
      expanded_field_paths: [],
    }],
    active_analysis_page_id: 'page:a',
    analysis_canvas_layout: layout,
  };
}

describe('Analysis Canvas presentation layout', () => {
  it('round-trips exact node positions and viewport through the versioned document', () => {
    const snapshot: AnalysisSnapshot = {
      pages: [{
        id: 'page:a',
        name: 'Page A',
        graphSpec,
        inputRequirements: [],
        evalParams: {},
        viewport: { x: -120, y: 64, zoom: 1.35 },
        nodePositions: {
          'analysis:a': { x: 432.5, y: -17.25 },
          __data_source__: { x: 16, y: 80 },
        },
        evalRunId: null,
        expandedFieldPaths: [],
      }],
      activePageId: 'page:a',
    };

    const layout = buildAnalysisCanvasLayoutDocument(undefined, null, snapshot);
    const restored = analysisSnapshotFromWorkspaceDocument(document(layout), null);

    expect(layout).toMatchObject({
      schema_id: ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
      schema_version: ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    });
    expect(restored?.pages[0]).toMatchObject({
      viewport: { x: -120, y: 64, zoom: 1.35 },
      nodePositions: {
        'analysis:a': { x: 432.5, y: -17.25 },
        __data_source__: { x: 16, y: 80 },
      },
    });
  });

  it('prunes stale layout keys and never materializes them as semantic nodes', () => {
    const restored = analysisSnapshotFromWorkspaceDocument(
      document({
        schema_id: ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        schema_version: ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        stages: {
          'stage:analysis': {
            pages: {
              'page:a': {
                node_positions: {
                  'analysis:a': { x: 20, y: 30 },
                  'stale:deleted': { x: 999, y: 999 },
                },
                viewport: { x: 0, y: 0, zoom: 1 },
              },
            },
          },
        },
      }),
      null,
    );

    expect(restored?.pages[0].nodePositions).toEqual({
      'analysis:a': { x: 20, y: 30 },
    });
    expect(restored?.pages[0].graphSpec.nodes).toEqual(graphSpec.nodes);
  });

  it('fails visibly for malformed geometry instead of substituting a layout', () => {
    const malformed = document({
      schema_id: ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
      schema_version: ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
      stages: {
        'stage:analysis': {
          pages: {
            'page:a': {
              node_positions: {
                'analysis:a': { x: Number.NaN, y: 30 },
              },
              viewport: { x: 0, y: 0, zoom: 1 },
            },
          },
        },
      },
    });

    expect(() => analysisSnapshotFromWorkspaceDocument(malformed, null))
      .toThrow('Analysis Canvas position for analysis:a is not finite and bounded');
  });
});
