import { describe, expect, it } from 'vitest';
import {
  analysisBundleCards,
  analysisSpecWithRetargetedBundle,
  bundlePredicateFromSelection,
  selectionSpecForAnalysisStage,
  selectionSpecWithPredicate,
  statusLabel,
} from '@/utils/analysisBundle';
import type { StudioScenarioSpec, StudioStageSpec } from '@/types/workspace';

function stage(selectionSpec: Record<string, unknown>): StudioStageSpec {
  return {
    id: 'stage:analysis',
    kind: 'analysis',
    label: 'Analysis',
    status: 'draft',
    scenario_id: 'scenario:analysis',
    input_collections: [],
    output_collections: [],
    manifest_refs: [],
    artifact_refs: [],
    execution_spec: null,
    selection_spec: selectionSpec,
    validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
    ui_state: {},
    metadata: {},
  };
}

function scenario(analysisSpec: Record<string, unknown> | null): StudioScenarioSpec {
  return {
    id: 'scenario:analysis',
    schema_version: 'feedbax.spec.studio.scenario.v2',
    label: 'Analysis',
    stage_id: 'stage:analysis',
    parent_scenario_id: null,
    training_spec: null,
    task_spec: null,
    task_binding_spec: null,
    objective_spec: null,
    probe_specs: [],
    temporal_spec: null,
    biomechanics_spec: null,
    analysis_spec: analysisSpec,
    report_spec: null,
    validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
    ui_state: {},
    metadata: {},
  };
}

describe('analysis bundle UI helpers', () => {
  it('normalizes legacy eval selection into a SelectionSpec predicate', () => {
    const spec = selectionSpecForAnalysisStage(stage({ eval_run_ids: ['eval-a'] }));

    expect(spec).toMatchObject({
      mode: 'explicit',
      manifest_kind: 'EvaluationRunManifest',
      ids: ['eval-a'],
    });
    expect(bundlePredicateFromSelection(spec)).toMatchObject({
      manifest_kind: 'EvaluationRunManifest',
      run_ids: ['eval-a'],
    });
  });

  it('uses authored bundle cards and synthesizes a DAG card when absent', () => {
    const authored = analysisBundleCards(
      scenario({
        bundles: [
          {
            schema_id: 'feedbax.spec.analysis_bundle',
            schema_version: 'feedbax.spec.analysis_bundle.v5',
            name: 'authored',
            predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
            stages: [
              {
                name: 'stage-a',
                kind: 'analysis',
                evaluation_states_policy: 'require_durable',
              },
              { name: 'stage-b', kind: 'materialization' },
              { name: 'stage-c', kind: 'report' },
            ],
          },
          {
            schema_id: 'feedbax.spec.analysis_bundle',
            schema_version: 'feedbax.spec.analysis_bundle.v5',
            name: 'authored-templates',
            predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
            templates: [
              {
                name: 'strict-template',
                analysis_type: 'strict',
                evaluation_states_policy: 'require_durable',
              },
              { name: 'legacy-template', analysis_type: 'legacy' },
            ],
          },
        ],
        pages: [{ id: 'page-a', name: 'Page A' }],
      }),
      stage({ eval_run_ids: ['eval-a'] }),
    );
    const synthesized = analysisBundleCards(
      scenario({ pages: [{ id: 'page-a', name: 'Page A' }] }),
      stage({ eval_run_ids: ['eval-a'] }),
    );

    expect(authored[0]).toMatchObject({ title: 'authored', stageCount: 3, pageCount: 1 });
    expect(authored[0].bundle).toMatchObject({
      schema_version: 'feedbax.spec.analysis_bundle.v5',
      stages: [
        { evaluation_states_policy: 'require_durable' },
        { evaluation_states_policy: 'recompute' },
        { kind: 'report' },
      ],
    });
    expect(authored[0].bundle.stages[2]).not.toHaveProperty('evaluation_states_policy');
    expect(authored[1].bundle.templates).toMatchObject([
      { evaluation_states_policy: 'require_durable' },
      { evaluation_states_policy: 'recompute' },
    ]);
    expect(synthesized[0].bundle).toMatchObject({
      schema_version: 'feedbax.spec.analysis_bundle.v6',
      name: 'studio-analysis-dag',
      predicate: { run_ids: ['eval-a'] },
      params_base: { params: {} },
      stages: [
        {
          evaluation_states_policy: 'recompute',
          local_params: { page_count: 1 },
          params_patches: [],
        },
      ],
    });
    expect(synthesized[0].bundle.stages[0]).not.toHaveProperty('params');
  });

  it('preserves legacy bundle versions for server-owned migration', () => {
    const cards = analysisBundleCards(
      scenario({
        bundle: {
          schema_id: 'feedbax.spec.analysis_bundle',
          schema_version: 'feedbax.spec.analysis_bundle.v2',
          name: 'legacy-v2',
          stages: [
            {
              name: 'analysis',
              kind: 'analysis',
              analysis_type: 'studio.analysis_dag',
              params: { page_count: 1 },
            },
          ],
        },
      }),
      stage({}),
    );

    expect(cards[0].bundle.schema_version).toBe('feedbax.spec.analysis_bundle.v2');
    expect(cards[0].bundle.stages[0]).toMatchObject({ params: { page_count: 1 } });
    expect(cards[0].bundle.stages[0]).not.toHaveProperty('local_params');
    expect(cards[0].bundle.stages[0]).not.toHaveProperty('evaluation_states_policy');
  });

  it('keeps distinct predicates for authored bundle cards', () => {
    const cards = analysisBundleCards(
      scenario({
        bundles: [
          {
            name: 'bundle-a',
            predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
            stages: [{ name: 'stage-a', kind: 'analysis' }],
          },
          {
            name: 'bundle-b',
            predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-b'] },
            stages: [{ name: 'stage-b', kind: 'analysis' }],
          },
        ],
      }),
      stage({ eval_run_ids: ['active-stage-eval'] }),
    );

    expect(cards).toHaveLength(2);
    expect(cards.map((card) => card.bundle.predicate.run_ids)).toEqual([['eval-a'], ['eval-b']]);
    expect(cards.map((card) => card.source)).toEqual([
      { kind: 'array', index: 0 },
      { kind: 'array', index: 1 },
    ]);
  });

  it('retargets predicates as query SelectionSpec values', () => {
    const selection = selectionSpecWithPredicate({
      manifest_kind: 'EvaluationRunManifest',
      statuses: ['completed'],
      run_ids: [],
      source_set_ids: [],
      tags: [],
      metadata_equals: {},
      params_equals: {},
      path_equals: {},
    });

    expect(selection).toMatchObject({
      mode: 'query',
      manifest_kind: 'EvaluationRunManifest',
      query: { statuses: ['completed'] },
    });
    expect(statusLabel('not_applicable')).toBe('not applicable');
  });

  it('retargets a singleton analysis bundle in place', () => {
    const analysisSpec = {
      bundle: {
        name: 'single',
        predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
        stages: [{ name: 'stage-a', kind: 'analysis' }],
      },
      pages: [{ id: 'page-a' }],
    };
    const card = analysisBundleCards(scenario(analysisSpec), stage({}))[0];

    const next = analysisSpecWithRetargetedBundle(analysisSpec, card, {
      manifest_kind: 'EvaluationRunManifest',
      run_ids: ['eval-b'],
      source_set_ids: [],
      statuses: [],
      tags: [],
      metadata_equals: {},
      params_equals: {},
      path_equals: {},
    });

    expect(next.bundle).toMatchObject({
      name: 'single',
      predicate: { run_ids: ['eval-b'] },
      metadata: { predicate_updated_from: 'studio_analysis_bundle_panel' },
    });
    expect(next).not.toHaveProperty('bundles');
  });

  it('retargets an array analysis bundle entry in place', () => {
    const analysisSpec = {
      bundles: [
        {
          name: 'bundle-a',
          predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
          stages: [{ name: 'stage-a', kind: 'analysis' }],
        },
        {
          name: 'bundle-b',
          predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-b'] },
          stages: [{ name: 'stage-b', kind: 'analysis' }],
        },
      ],
    };
    const cards = analysisBundleCards(scenario(analysisSpec), stage({}));

    const next = analysisSpecWithRetargetedBundle(analysisSpec, cards[1], {
      manifest_kind: 'EvaluationRunManifest',
      run_ids: ['eval-c'],
      source_set_ids: [],
      statuses: [],
      tags: [],
      metadata_equals: {},
      params_equals: {},
      path_equals: {},
    });

    expect(next).not.toHaveProperty('bundle');
    expect(next.bundles).toMatchObject([
      { name: 'bundle-a', predicate: { run_ids: ['eval-a'] } },
      {
        name: 'bundle-b',
        predicate: { run_ids: ['eval-c'] },
        metadata: { predicate_updated_from: 'studio_analysis_bundle_panel' },
      },
    ]);
  });
});
