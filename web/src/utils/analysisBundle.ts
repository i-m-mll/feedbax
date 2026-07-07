import type { AnalysisPageWire, StudioScenarioSpec, StudioStageSpec } from '@/types/workspace';
import type {
  AnalysisBundleDryRunResult,
  BundleStageDryRunRecord,
  ManifestPredicate,
  SelectionSpec,
} from '@/generated/studioContracts';
import {
  explicitSelectionSpec,
  migrateLegacySelectionSpec,
  querySelectionSpec,
  selectedParentIds,
} from '@/utils/selectionSpec';

export const ANALYSIS_BUNDLE_SCHEMA_ID = 'feedbax.spec.analysis_bundle';
export const ANALYSIS_BUNDLE_SCHEMA_VERSION = 'feedbax.spec.analysis_bundle.v2';

export type AnalysisBundleStageStatus =
  | 'would_run'
  | 'would_skip'
  | 'missing'
  | 'not_applicable';

export interface AnalysisBundleSpecWire {
  schema_id: typeof ANALYSIS_BUNDLE_SCHEMA_ID;
  schema_version: typeof ANALYSIS_BUNDLE_SCHEMA_VERSION;
  name: string;
  description?: string | null;
  predicate: ManifestPredicate;
  templates: unknown[];
  stages: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
}

export interface AnalysisBundleCard {
  id: string;
  bundle: AnalysisBundleSpecWire;
  title: string;
  description: string | null;
  stageCount: number;
  pageCount: number;
  dryRun: AnalysisBundleDryRunResult | null;
}

export function selectionSpecForAnalysisStage(
  stage: StudioStageSpec | null | undefined,
): SelectionSpec {
  const migrated = migrateLegacySelectionSpec(stage?.selection_spec ?? {});
  if (migrated) return migrated;
  const evalRunIds = selectionIds(stage?.selection_spec, 'eval_run_ids')
    .concat(selectionIds(stage?.selection_spec, 'evaluation_run_ids'));
  return explicitSelectionSpec(evalRunIds, 'EvaluationRunManifest');
}

export function bundlePredicateFromSelection(selectionSpec: SelectionSpec): ManifestPredicate {
  if (selectionSpec.query) return selectionSpec.query;
  return {
    manifest_kind: selectionSpec.manifest_kind ?? 'EvaluationRunManifest',
    run_ids: selectedParentIds(selectionSpec),
    source_set_ids: [],
    statuses: [],
    tags: [],
    metadata_equals: {},
    params_equals: {},
    path_equals: {},
  };
}

export function selectionSpecWithPredicate(predicate: ManifestPredicate): SelectionSpec {
  return querySelectionSpec({
    manifest_kind: predicate.manifest_kind ?? 'EvaluationRunManifest',
    run_ids: predicate.run_ids ?? [],
    source_set_ids: predicate.source_set_ids ?? [],
    statuses: predicate.statuses ?? [],
    has_checkpoint: predicate.has_checkpoint ?? null,
    tags: predicate.tags ?? [],
    metadata_equals: predicate.metadata_equals ?? {},
    params_equals: predicate.params_equals ?? {},
    path_equals: predicate.path_equals ?? {},
    expression: predicate.expression ?? null,
    top_k_by_metric_per_group: predicate.top_k_by_metric_per_group ?? null,
  });
}

export function analysisBundleCards(
  scenario: StudioScenarioSpec | null | undefined,
  stage: StudioStageSpec | null | undefined,
): AnalysisBundleCard[] {
  const analysisSpec = recordValue(scenario?.analysis_spec);
  const authored = authoredBundles(analysisSpec);
  const bundles = authored.length > 0 ? authored : [synthesizedBundle(analysisSpec, stage)];
  return bundles.map((bundle, index) => ({
    id: `${bundle.name}:${index}`,
    bundle,
    title: bundle.name,
    description: bundle.description ?? null,
    stageCount: bundle.stages.length,
    pageCount: analysisPages(analysisSpec).length,
    dryRun: null,
  }));
}

export function bundleWithPredicate(
  bundle: AnalysisBundleSpecWire,
  predicate: ManifestPredicate,
): AnalysisBundleSpecWire {
  return {
    ...bundle,
    predicate,
    metadata: {
      ...bundle.metadata,
      predicate_updated_from: 'studio_analysis_bundle_panel',
    },
  };
}

export function statusLabel(status: AnalysisBundleStageStatus | string): string {
  switch (status) {
    case 'would_run':
      return 'would run';
    case 'would_skip':
      return 'skipped';
    case 'not_applicable':
      return 'not applicable';
    case 'missing':
      return 'missing';
    default:
      return status.replace(/_/g, ' ');
  }
}

export function stageReason(stage: BundleStageDryRunRecord): string | null {
  if (stage.reason) return stage.reason;
  return stage.missing_roles?.[0]?.reason ?? null;
}

function authoredBundles(spec: Record<string, unknown> | null): AnalysisBundleSpecWire[] {
  const bundles = arrayValue(spec?.bundles).map(coerceBundle).filter(Boolean);
  const single = coerceBundle(spec?.bundle);
  return single ? [single, ...bundles] : bundles;
}

function coerceBundle(value: unknown): AnalysisBundleSpecWire | null {
  const record = recordValue(value);
  if (!record || typeof record.name !== 'string') return null;
  return {
    schema_id: ANALYSIS_BUNDLE_SCHEMA_ID,
    schema_version: ANALYSIS_BUNDLE_SCHEMA_VERSION,
    name: record.name,
    description: typeof record.description === 'string' ? record.description : null,
    predicate: manifestPredicateValue(record.predicate),
    templates: arrayValue(record.templates),
    stages: arrayValue(record.stages).filter(isRecord),
    metadata: recordValue(record.metadata) ?? {},
  };
}

function synthesizedBundle(
  analysisSpec: Record<string, unknown> | null,
  stage: StudioStageSpec | null | undefined,
): AnalysisBundleSpecWire {
  const pages = analysisPages(analysisSpec);
  const selectionSpec = selectionSpecForAnalysisStage(stage);
  return {
    schema_id: ANALYSIS_BUNDLE_SCHEMA_ID,
    schema_version: ANALYSIS_BUNDLE_SCHEMA_VERSION,
    name: 'studio-analysis-dag',
    description: 'Analysis DAG draft',
    predicate: bundlePredicateFromSelection(selectionSpec),
    templates: [],
    stages: [
      {
        name: 'analysis-dag',
        kind: 'analysis',
        mode: 'grouped',
        analysis_type: 'studio.analysis_dag',
        params: {
          page_count: pages.length,
          active_page_id:
            typeof analysisSpec?.active_page_id === 'string' ? analysisSpec.active_page_id : null,
        },
        outputs: [{ role: 'manifest', required: true }],
      },
    ],
    metadata: {
      source: 'studio_analysis_stage',
      page_count: pages.length,
    },
  };
}

function analysisPages(spec: Record<string, unknown> | null): AnalysisPageWire[] {
  return arrayValue(spec?.pages).filter(isRecord) as unknown as AnalysisPageWire[];
}

function manifestPredicateValue(value: unknown): ManifestPredicate {
  const record = recordValue(value);
  return {
    manifest_kind: stringValue(record?.manifest_kind) ?? 'EvaluationRunManifest',
    run_ids: stringArray(record?.run_ids),
    source_set_ids: stringArray(record?.source_set_ids),
    statuses: stringArray(record?.statuses),
    has_checkpoint: typeof record?.has_checkpoint === 'boolean' ? record.has_checkpoint : null,
    tags: stringArray(record?.tags),
    metadata_equals: recordValue(record?.metadata_equals) ?? {},
    params_equals: recordValue(record?.params_equals) ?? {},
    path_equals: recordValue(record?.path_equals) ?? {},
    expression: recordValue(record?.expression),
    top_k_by_metric_per_group: recordValue(record?.top_k_by_metric_per_group) as
      | ManifestPredicate['top_k_by_metric_per_group']
      | null,
  };
}

function selectionIds(value: unknown, key: string): string[] {
  return stringArray(recordValue(value)?.[key]);
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}
