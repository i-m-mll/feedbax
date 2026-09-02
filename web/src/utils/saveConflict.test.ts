import { describe, expect, it } from 'vitest';
import {
  changedSaveConflictSections,
  summarizeSaveConflict,
  type SaveConflictSnapshot,
} from '@/utils/saveConflict';
import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';

const metadata: GraphMetadata = {
  name: 'Server project',
  created_at: '2026-07-07T00:00:00+00:00',
  updated_at: '2026-07-07T12:00:00+00:00',
  version: '1.0.0',
  save_revision: 8,
};

const baseGraph: GraphSpec = {
  nodes: {},
  wires: [],
  input_ports: [],
  output_ports: [],
  input_bindings: {},
  output_bindings: {},
  metadata,
};

const baseUi: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {},
};

function snapshot(overrides: Partial<SaveConflictSnapshot> = {}): SaveConflictSnapshot {
  return {
    graph: baseGraph,
    uiState: baseUi,
    workspace: null,
    analysisPages: null,
    activeAnalysisPageId: null,
    ...overrides,
  };
}

describe('save conflict summaries', () => {
  it('reports concrete server truth and local expected revision', () => {
    const message = summarizeSaveConflict({
      expectedRevision: 4,
      serverMetadata: metadata,
      local: snapshot(),
      server: snapshot({
        graph: {
          ...baseGraph,
          nodes: {
            serverOnly: {
              type: 'Gain',
              params: {},
              input_ports: ['input'],
              output_ports: ['output'],
            },
          },
        },
      }),
    });

    expect(message).toContain('Server project');
    expect(message).toContain('revision 8');
    expect(message).toContain('expected revision 4');
    expect(message).toContain('updated 2026-07-07T12:00:00+00:00');
    expect(message).toContain('Server differs in: graph.');
    expect(message).toContain('local edits are still unsaved');
  });

  it('summarizes high-level sections that differ', () => {
    const sections = changedSaveConflictSections(
      snapshot({
        analysisPages: [{ id: 'local-page' }],
        activeAnalysisPageId: 'local-page',
      }),
      snapshot({
        uiState: { ...baseUi, viewport: { x: 2, y: 0, zoom: 1 } },
        workspace: {
          id: 'workspace:server',
          schema_id: 'feedbax.spec.studio.workspace',
          schema_version: 'feedbax.spec.studio.workspace.v2',
          label: 'Server workspace',
          active_stage_id: null,
          stages: [],
          scenarios: {},
          collections: [],
          manifest_refs: [],
          artifact_refs: [],
          validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
          ui_state: {},
          metadata: {},
        },
        analysisPages: [{ id: 'server-page' }],
        activeAnalysisPageId: 'server-page',
      }),
    );

    expect(sections).toEqual(['UI state', 'workspace', 'analysis pages']);
  });
});
