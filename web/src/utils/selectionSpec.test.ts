import { describe, expect, it } from 'vitest';
import type { ManifestPredicate, ParentRef, SelectionRefreshDiff } from '@/generated/studioContracts';
import {
  explicitSelectionSpec,
  frozenSelectionSpec,
  migrateLegacySelectionSpec,
  querySelectionSpec,
  selectedParentIds,
  selectionRefreshCounts,
} from '@/utils/selectionSpec';

describe('SelectionSpec utilities', () => {
  it('migrates legacy id-list payloads into explicit SelectionSpec v2', () => {
    expect(migrateLegacySelectionSpec({ training_run_ids: ['run-a'] })).toMatchObject({
      schema_id: 'feedbax.spec.selection',
      schema_version: 'feedbax.spec.selection.v2',
      mode: 'explicit',
      manifest_kind: 'TrainingRunManifest',
      ids: ['run-a'],
    });
    expect(migrateLegacySelectionSpec({ eval_run_ids: ['eval-a'] })?.manifest_kind)
      .toBe('EvaluationRunManifest');
    expect(migrateLegacySelectionSpec({ selected: ['unknown'] })).toBeNull();
  });

  it('constructs query and frozen specs without deriving P5 selection state', () => {
    const query: ManifestPredicate = {
      manifest_kind: 'TrainingRunManifest',
      source_set_ids: ['sweep-a'],
      statuses: ['completed'],
      has_checkpoint: true,
      run_ids: [],
      tags: [],
      metadata_equals: {},
      params_equals: {},
      path_equals: {},
    };
    const ref: ParentRef = {
      kind: 'TrainingRunManifest',
      id: 'run-a',
      role: 'training_run',
      metadata: {},
    };

    expect(querySelectionSpec(query)).toMatchObject({ mode: 'query', query });
    expect(frozenSelectionSpec(query, [ref], '2026-07-07T00:00:00Z')).toMatchObject({
      mode: 'frozen',
      frozen_refs: [ref],
    });
    expect(selectedParentIds(explicitSelectionSpec(['run-a']))).toEqual(['run-a']);
    expect(selectedParentIds(frozenSelectionSpec(query, [ref], '2026-07-07T00:00:00Z')))
      .toEqual(['run-a']);
  });

  it('summarizes refresh reprocess counts', () => {
    const diff: SelectionRefreshDiff = {
      frozen_refs: [],
      current_refs: [],
      new_refs: [{ kind: 'TrainingRunManifest', id: 'run-new', metadata: {} }],
      gone_refs: [{ kind: 'TrainingRunManifest', id: 'run-gone', metadata: {} }],
      unchanged_refs: [],
      reprocess_counts: { missing: 1, missing_failed: 2, all: 3, stale: 1 },
    };

    expect(selectionRefreshCounts(diff)).toEqual({
      newCount: 1,
      goneCount: 1,
      missing: 1,
      missingFailed: 2,
      all: 3,
      stale: 1,
    });
  });
});
