import { describe, expect, it } from 'vitest';
import type { ManifestPredicate, SelectionSpec } from '@/generated/studioContracts';
import {
  explicitSelectionSpec,
  migrateLegacySelectionSpec,
  querySelectionSpec,
  selectedParentIds,
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

  it('constructs a query and reads selected IDs from explicit and saved frozen specs', () => {
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
    expect(querySelectionSpec(query)).toMatchObject({ mode: 'query', query });
    expect(selectedParentIds(explicitSelectionSpec(['run-a']))).toEqual(['run-a']);
    const frozen: SelectionSpec = {
      ...querySelectionSpec(query),
      mode: 'frozen',
      frozen_refs: [{ kind: 'TrainingRunManifest', id: 'run-a', metadata: {} }],
      frozen_at: '2026-07-07T00:00:00Z',
    };
    expect(selectedParentIds(frozen)).toEqual(['run-a']);
  });
});
