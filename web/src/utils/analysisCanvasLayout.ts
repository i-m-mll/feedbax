import type {
  AnalysisCanvasLayoutDocument,
  AnalysisCanvasPosition,
  AnalysisCanvasViewport,
  AnalysisPageSpec as AnalysisPageWire,
  WorkspaceDocument,
} from '@/generated/studioContracts';
import type {
  AnalysisGraphSpec,
  AnalysisInputRequirement,
  AnalysisPageSpec,
  AnalysisSnapshot,
  AnalysisViewport,
} from '@/types/analysis';
import type { StudioWorkspaceSpec } from '@/types/workspace';

export const ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID =
  'feedbax.spec.studio.analysis_canvas_layout' as const;
export const ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION =
  'feedbax.spec.studio.analysis_canvas_layout.v1' as const;

const DEFAULT_VIEWPORT: AnalysisViewport = { x: 0, y: 0, zoom: 1 };
const DEFAULT_ANALYSIS_STAGE_ID = 'stage:analysis';

function analysisStageId(workspace: StudioWorkspaceSpec | null | undefined): string {
  return workspace?.stages.find((stage) => stage.kind === 'analysis')?.id
    ?? DEFAULT_ANALYSIS_STAGE_ID;
}

function semanticNodeIds(graphSpec: AnalysisGraphSpec): Set<string> {
  const ids = new Set(Object.keys(graphSpec.nodes));
  ids.add(graphSpec.dataSourceId);
  for (const wire of graphSpec.wires) {
    if (wire.transform?.id) ids.add(wire.transform.id);
  }
  return ids;
}

function safePosition(value: AnalysisCanvasPosition | undefined): value is AnalysisCanvasPosition {
  return Boolean(
    value
    && Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Math.abs(value.x) <= 10_000_000
    && Math.abs(value.y) <= 10_000_000
  );
}

export function reconcileAnalysisNodePositions(
  graphSpec: AnalysisGraphSpec,
  positions: Record<string, AnalysisCanvasPosition> | undefined,
): Record<string, AnalysisCanvasPosition> {
  const validIds = semanticNodeIds(graphSpec);
  const reconciled: Record<string, AnalysisCanvasPosition> = {};
  for (const [nodeId, position] of Object.entries(positions ?? {})) {
    if (!safePosition(position)) {
      throw new Error(`Analysis Canvas position for ${nodeId} is not finite and bounded`);
    }
    if (validIds.has(nodeId)) reconciled[nodeId] = position;
  }
  return reconciled;
}

function normalizedViewport(viewport: AnalysisCanvasViewport | undefined): AnalysisViewport {
  const x = viewport?.x ?? DEFAULT_VIEWPORT.x;
  const y = viewport?.y ?? DEFAULT_VIEWPORT.y;
  const zoom = viewport?.zoom ?? DEFAULT_VIEWPORT.zoom;
  if (
    !Number.isFinite(x)
    || !Number.isFinite(y)
    || !Number.isFinite(zoom)
    || Math.abs(x) > 10_000_000
    || Math.abs(y) > 10_000_000
    || zoom < 0.1
    || zoom > 2.5
  ) {
    throw new Error('Analysis Canvas viewport is not finite and bounded');
  }
  return { x, y, zoom };
}

function graphFromWire(page: AnalysisPageWire): AnalysisGraphSpec {
  return (page.graph_spec ?? {
    nodes: {},
    wires: [],
    dataSourceId: '__data_source__',
  }) as unknown as AnalysisGraphSpec;
}

function pageFromWire(
  page: AnalysisPageWire,
  layout: AnalysisCanvasLayoutDocument | undefined,
  stageId: string,
): AnalysisPageSpec {
  const graphSpec = graphFromWire(page);
  const pageLayout = layout?.stages?.[stageId]?.pages?.[page.id];
  return {
    id: page.id,
    name: page.name,
    graphSpec,
    inputRequirements: (page.input_requirements ?? []) as unknown as AnalysisInputRequirement[],
    evalParams: page.eval_params ?? {},
    viewport: normalizedViewport(pageLayout?.viewport),
    nodePositions: reconcileAnalysisNodePositions(graphSpec, pageLayout?.node_positions),
    evalRunId: page.eval_run_id ?? null,
    expandedFieldPaths: page.expanded_field_paths ?? [],
  };
}

export function analysisSnapshotFromWorkspaceDocument(
  document: WorkspaceDocument,
  workspace: StudioWorkspaceSpec | null | undefined,
): AnalysisSnapshot | null {
  const wirePages = document.analysis_pages ?? [];
  if (wirePages.length === 0) return null;
  const stageId = analysisStageId(workspace);
  const pages = wirePages.map((page) =>
    pageFromWire(page, document.analysis_canvas_layout, stageId)
  );
  const requestedActiveId = document.active_analysis_page_id;
  return {
    pages,
    activePageId: requestedActiveId && pages.some((page) => page.id === requestedActiveId)
      ? requestedActiveId
      : pages[0].id,
  };
}

export function buildAnalysisCanvasLayoutDocument(
  current: AnalysisCanvasLayoutDocument | undefined,
  workspace: StudioWorkspaceSpec | null | undefined,
  snapshot: AnalysisSnapshot | null,
): AnalysisCanvasLayoutDocument {
  const stageId = analysisStageId(workspace);
  const pages = Object.fromEntries(
    (snapshot?.pages ?? []).map((page) => [
      page.id,
      {
        node_positions: reconcileAnalysisNodePositions(page.graphSpec, page.nodePositions),
        viewport: normalizedViewport(page.viewport),
      },
    ])
  );
  return {
    schema_id: ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    schema_version: ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    stages: {
      ...(current?.stages ?? {}),
      [stageId]: { pages },
    },
  };
}
