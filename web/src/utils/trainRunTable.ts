import type { TrainingRun } from '@/types/runs';
import type { StudioManifestRef, StudioStageSpec } from '@/types/workspace';
import { trainingRunMetricValue, type TrainingRunSummary } from '@/utils/pipelineCollections';

export interface TrainingRunSortState {
  key: string;
  direction: 'asc' | 'desc';
}

export interface ProgressLike {
  job_id?: string | null;
  batch: number;
  total_batches: number;
}

export interface ProgressBindings {
  byRunId: Map<string, string>;
  byGroupId: Map<string, string>;
}

export const UNGROUPED_RUN_GROUP_ID = 'ungrouped';

export function trainingRunGroupId(row: TrainingRunSummary): string {
  return row.runSetId ?? UNGROUPED_RUN_GROUP_ID;
}

export function sortTrainingRows(
  rows: TrainingRunSummary[],
  sort: TrainingRunSortState
): TrainingRunSummary[] {
  return [...rows].sort((a, b) => {
    const direction = sort.direction === 'asc' ? 1 : -1;
    return compareSortableValue(sortableValue(a, sort.key), sortableValue(b, sort.key)) * direction;
  });
}

function sortableValue(row: TrainingRunSummary, key: string): number | string | null {
  if (key === 'progress') return row.status === 'completed' ? row.warmupBatches ?? 1 : 0;
  if (key.startsWith('axis:')) {
    const value = row.axisCoordinates[key.slice('axis:'.length)];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim()) return value;
    if (typeof value === 'boolean') return value ? 1 : 0;
    return value === undefined || value === null ? null : JSON.stringify(value);
  }
  return trainingRunMetricValue(row, key);
}

function compareSortableValue(a: number | string | null, b: number | string | null): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function progressBindingsForRuns(
  rows: TrainingRunSummary[],
  progress: ProgressLike | null | undefined,
  activeJobId?: string | null
): ProgressBindings {
  const byRunId = new Map<string, string>();
  const byGroupId = new Map<string, string>();
  const jobId = progress?.job_id ?? activeJobId ?? null;
  if (!progress || !jobId) return { byRunId, byGroupId };
  const label = `${progress.batch}/${progress.total_batches}`;
  const matches = rows.filter((row) => row.id === jobId || row.jobId === jobId);
  if (matches.length === 1) {
    byRunId.set(matches[0].id, label);
    return { byRunId, byGroupId };
  }
  for (const row of matches) {
    byGroupId.set(trainingRunGroupId(row), label);
  }
  return { byRunId, byGroupId };
}

export function stageWithTrainingRunLifecyclePatch(
  stage: StudioStageSpec,
  action: 'update' | 'remove',
  run: TrainingRun
): StudioStageSpec {
  const patch = (ref: StudioManifestRef): StudioManifestRef => {
    if (ref.id !== run.id) return ref;
    return {
      ...ref,
      uri: run.uri ?? ref.uri,
      metadata: {
        ...ref.metadata,
        name: run.name,
        status: run.status,
        planned: run.planned ?? ref.metadata.planned,
        checkpoint_available: run.checkpointAvailable ?? ref.metadata.checkpoint_available,
        source_issue: run.sourceIssue ?? ref.metadata.source_issue,
        provenance_id: run.provenanceId ?? ref.metadata.provenance_id,
        superseded_by: run.supersededBy ?? ref.metadata.superseded_by,
      },
    };
  };
  const keep = (ref: StudioManifestRef) => ref.id !== run.id;
  return {
    ...stage,
    output_collections: stage.output_collections.map((collection) => ({
      ...collection,
      item_refs:
        action === 'remove'
          ? collection.item_refs.filter(keep)
          : collection.item_refs.map(patch),
    })),
    manifest_refs:
      action === 'remove'
        ? stage.manifest_refs.filter(keep)
        : stage.manifest_refs.map(patch),
  };
}
