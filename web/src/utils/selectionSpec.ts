import type {
  ManifestPredicate,
  ParentRef,
  SelectionRefreshDiff,
  SelectionSpec,
} from '@/generated/studioContracts';

export const SELECTION_SPEC_SCHEMA_ID = 'feedbax.spec.selection';
export const SELECTION_SPEC_SCHEMA_VERSION = 'feedbax.spec.selection.v2';

export function explicitSelectionSpec(
  ids: string[],
  manifestKind = 'TrainingRunManifest',
): SelectionSpec {
  return {
    schema_id: SELECTION_SPEC_SCHEMA_ID,
    schema_version: SELECTION_SPEC_SCHEMA_VERSION,
    mode: 'explicit',
    manifest_kind: manifestKind,
    ids,
    frozen_refs: [],
    metadata: {},
  };
}

export function querySelectionSpec(query: ManifestPredicate): SelectionSpec {
  return {
    schema_id: SELECTION_SPEC_SCHEMA_ID,
    schema_version: SELECTION_SPEC_SCHEMA_VERSION,
    mode: 'query',
    manifest_kind: query.manifest_kind,
    ids: [],
    query,
    frozen_refs: [],
    metadata: {},
  };
}

export function frozenSelectionSpec(
  query: ManifestPredicate,
  refs: ParentRef[],
  frozenAt: string,
): SelectionSpec {
  return {
    schema_id: SELECTION_SPEC_SCHEMA_ID,
    schema_version: SELECTION_SPEC_SCHEMA_VERSION,
    mode: 'frozen',
    manifest_kind: query.manifest_kind,
    ids: [],
    query,
    frozen_refs: refs,
    frozen_at: frozenAt,
    metadata: {},
  };
}

export function migrateLegacySelectionSpec(
  payload: Record<string, unknown>,
): SelectionSpec | null {
  const direct = selectionIds(payload, 'ids');
  const training = selectionIds(payload, 'training_run_ids');
  const evaluation = selectionIds(payload, 'eval_run_ids')
    .concat(selectionIds(payload, 'evaluation_run_ids'));
  const analysis = selectionIds(payload, 'analysis_run_ids');
  const reports = selectionIds(payload, 'report_ids');
  if (direct.length > 0) {
    return explicitSelectionSpec(
      direct,
      stringValue(payload.manifest_kind) ?? 'TrainingRunManifest',
    );
  }
  if (training.length > 0) return explicitSelectionSpec(training, 'TrainingRunManifest');
  if (evaluation.length > 0) return explicitSelectionSpec(evaluation, 'EvaluationRunManifest');
  if (analysis.length > 0) return explicitSelectionSpec(analysis, 'AnalysisRunManifest');
  if (reports.length > 0) return explicitSelectionSpec(reports, 'ReportManifest');
  return null;
}

export function selectedParentIds(spec: SelectionSpec): string[] {
  if (spec.mode === 'explicit') return spec.ids;
  if (spec.mode === 'frozen') return spec.frozen_refs.map((ref) => ref.id);
  return [];
}

export function selectionRefreshCounts(diff: SelectionRefreshDiff): {
  newCount: number;
  goneCount: number;
  missing: number;
  missingFailed: number;
  all: number;
  stale: number;
} {
  return {
    newCount: diff.new_refs.length,
    goneCount: diff.gone_refs.length,
    missing: diff.reprocess_counts.missing ?? 0,
    missingFailed: diff.reprocess_counts.missing_failed ?? 0,
    all: diff.reprocess_counts.all ?? 0,
    stale: diff.reprocess_counts.stale ?? 0,
  };
}

function selectionIds(payload: Record<string, unknown>, key: string): string[] {
  const value = payload[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}
