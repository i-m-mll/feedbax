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
    schema_version: 'feedbax.studio.scenario.v1',
    label: 'Analysis',
    stage_id: 'stage:analysis',
    parent_scenario_id: null,
    graph: null,
    graph_ui_state: null,
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
        bundle: {
          name: 'authored',
          predicate: { manifest_kind: 'EvaluationRunManifest', run_ids: ['eval-a'] },
          stages: [{ name: 'stage-a', kind: 'analysis' }],
        },
        pages: [{ id: 'page-a', name: 'Page A' }],
      }),
      stage({ eval_run_ids: ['eval-a'] }),
    );
    const synthesized = analysisBundleCards(
      scenario({ pages: [{ id: 'page-a', name: 'Page A' }] }),
      stage({ eval_run_ids: ['eval-a'] }),
    );

    expect(authored[0]).toMatchObject({ title: 'authored', stageCount: 1, pageCount: 1 });
    expect(synthesized[0].bundle).toMatchObject({
      name: 'studio-analysis-dag',
      predicate: { run_ids: ['eval-a'] },
    });
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
