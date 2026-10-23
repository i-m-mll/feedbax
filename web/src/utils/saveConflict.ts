import type { GraphMetadata, GraphSpec, GraphUIState } from '@/types/graph';
import type { StudioWorkspaceSpec } from '@/types/workspace';

export interface SaveConflictSnapshot {
  graph: GraphSpec | null;
  uiState: GraphUIState | null;
  workspace: StudioWorkspaceSpec | null;
  analysisPages: unknown[] | null;
  activeAnalysisPageId: string | null;
}

export interface SaveConflictSummaryInput {
  expectedRevision: number | null;
  serverMetadata: GraphMetadata | null;
  local: SaveConflictSnapshot;
  server: SaveConflictSnapshot;
}

function stableJson(value: unknown): string {
  return JSON.stringify(value ?? null);
}

export function changedSaveConflictSections(
  local: SaveConflictSnapshot,
  server: SaveConflictSnapshot,
): string[] {
  const sections: string[] = [];
  if (stableJson(local.graph) !== stableJson(server.graph)) sections.push('graph');
  if (stableJson(local.uiState) !== stableJson(server.uiState)) sections.push('UI state');
  if (stableJson(local.workspace) !== stableJson(server.workspace)) sections.push('workspace');
  if (
    stableJson(local.analysisPages) !== stableJson(server.analysisPages) ||
    local.activeAnalysisPageId !== server.activeAnalysisPageId
  ) {
    sections.push('analysis pages');
  }
  return sections;
}

export function summarizeSaveConflict(input: SaveConflictSummaryInput): string {
  const changedSections = changedSaveConflictSections(input.local, input.server);
  const serverName = input.serverMetadata?.name ?? 'untitled project';
  const serverRevision =
    input.serverMetadata?.save_revision !== undefined
      ? String(input.serverMetadata.save_revision)
      : 'unknown';
  const expectedRevision =
    input.expectedRevision !== null ? String(input.expectedRevision) : 'unknown';
  const updatedAt = input.serverMetadata?.updated_at ?? 'unknown time';
  const sectionSummary =
    changedSections.length > 0
      ? `Server differs in: ${changedSections.join(', ')}.`
      : 'No high-level graph, UI, workspace, or analysis-page differences were detected.';

  return (
    `Save conflict: server "${serverName}" is at revision ${serverRevision} ` +
    `(updated ${updatedAt}); this tab expected revision ${expectedRevision}. ` +
    `${sectionSummary} Your local edits are still unsaved.`
  );
}
