import type {
  AnalysisPageWire,
  StudioArtifactRef,
  StudioManifestRef,
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioSpec,
  StudioStageKind,
  StudioStageSpec,
  StudioWorkspaceSpec,
} from '@/types/workspace';

export type ScenarioMetricSource = 'objective' | 'analysis' | 'manifest' | 'task_default';
export type ScenarioOverlaySource = 'artifact' | 'analysis' | 'evaluation';

export interface ScenarioMetricSpec {
  id: string;
  label: string;
  role: string;
  source: ScenarioMetricSource;
  selector: string | null;
  units: string | null;
  stageId: string | null;
  scenarioId: string | null;
  sourceId: string;
  summary: string | null;
  metadata: Record<string, unknown>;
}

export interface ScenarioArtifactOverlay {
  id: string;
  label: string;
  source: ScenarioOverlaySource;
  role: string;
  stageId: string | null;
  artifactId: string | null;
  uri: string | null;
  mediaType: string | null;
  metricIds: string[];
  summary: string | null;
  metadata: Record<string, unknown>;
}

export interface StageProductReference {
  id: string;
  label: string;
  kind: 'input_collection' | 'output_collection' | 'analysis_page' | 'report_section';
  stageId: string;
  stageKind: StudioStageKind;
  sourceStageId: string | null;
  collectionId: string | null;
  scenarioId: string | null;
  itemCount: number;
  manifestIds: string[];
  artifactIds: string[];
  summary: string | null;
  metadata: Record<string, unknown>;
}

const METRIC_LABELS: Record<string, { label: string; units: string | null }> = {
  final_validation_loss: { label: 'Final validation loss', units: null },
  within_cell_velocity_rmse_m_per_s: {
    label: 'Within-cell velocity RMSE',
    units: 'm/s',
  },
  peak_velocity_m_per_s: { label: 'Peak velocity', units: 'm/s' },
  hold_drift_mm: { label: 'Hold drift', units: 'mm' },
  target_reach_error: { label: 'Target reach error', units: null },
};

const MANIFEST_METRIC_KEYS = Object.keys(METRIC_LABELS).filter(
  (key) => key !== 'target_reach_error'
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function titleFromId(value: string): string {
  return value
    .replace(/[_:.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricLabel(metricId: string): { label: string; units: string | null } {
  return METRIC_LABELS[metricId] ?? { label: titleFromId(metricId), units: null };
}

function objectiveSpec(value: unknown): StudioObjectiveSpec | null {
  if (!isRecord(value) || !Array.isArray(value.terms)) return null;
  return value as unknown as StudioObjectiveSpec;
}

function analysisPages(scenario: StudioScenarioSpec): AnalysisPageWire[] {
  const spec = scenario.analysis_spec;
  if (!isRecord(spec) || !Array.isArray(spec.pages)) return [];
  return spec.pages.filter((page): page is AnalysisPageWire => isRecord(page));
}

function graphNodes(page: AnalysisPageWire): Array<Record<string, unknown>> {
  const graphSpec = page.graph_spec;
  if (!isRecord(graphSpec) || !isRecord(graphSpec.nodes)) return [];
  return Object.values(graphSpec.nodes).filter(isRecord);
}

function allManifestRefs(workspace: StudioWorkspaceSpec): Array<{
  ref: StudioManifestRef;
  stage: StudioStageSpec | null;
}> {
  const refs = new Map<string, { ref: StudioManifestRef; stage: StudioStageSpec | null }>();
  for (const ref of workspace.manifest_refs) refs.set(ref.id, { ref, stage: null });
  for (const stage of workspace.stages) {
    for (const ref of stage.manifest_refs) refs.set(ref.id, { ref, stage });
    for (const collection of [...stage.input_collections, ...stage.output_collections]) {
      for (const ref of collection.item_refs) refs.set(ref.id, { ref, stage });
    }
  }
  return Array.from(refs.values());
}

function allArtifactRefs(workspace: StudioWorkspaceSpec): Array<{
  ref: StudioArtifactRef;
  stage: StudioStageSpec | null;
}> {
  const refs = new Map<string, { ref: StudioArtifactRef; stage: StudioStageSpec | null }>();
  for (const ref of workspace.artifact_refs ?? []) refs.set(ref.id, { ref, stage: null });
  for (const stage of workspace.stages) {
    for (const ref of stage.artifact_refs ?? []) refs.set(ref.id, { ref, stage });
  }
  return Array.from(refs.values());
}

function pushUniqueMetric(metrics: ScenarioMetricSpec[], metric: ScenarioMetricSpec) {
  const key = `${metric.source}:${metric.id}:${metric.sourceId}`;
  if (metrics.some((existing) => `${existing.source}:${existing.id}:${existing.sourceId}` === key)) {
    return;
  }
  metrics.push(metric);
}

function objectiveMetrics(
  scenario: StudioScenarioSpec,
  stage: StudioStageSpec | null
): ScenarioMetricSpec[] {
  const spec = objectiveSpec(scenario.objective_spec);
  if (!spec) return [];
  return spec.terms
    .filter((term): term is StudioObjectiveTermSpec => term.role === 'metric')
    .map((term) => ({
      id: term.id,
      label: term.label,
      role: term.role,
      source: 'objective' as const,
      selector: term.source_selector?.compact ?? term.target_selector?.compact ?? null,
      units: term.units ?? null,
      stageId: stage?.id ?? scenario.stage_id ?? null,
      scenarioId: scenario.id,
      sourceId: term.id,
      summary: term.source_selector?.compact ?? term.target_selector?.compact ?? null,
      metadata: { ...term.metadata },
    }));
}

function analysisMetrics(
  scenario: StudioScenarioSpec,
  stage: StudioStageSpec | null
): ScenarioMetricSpec[] {
  const metrics: ScenarioMetricSpec[] = [];
  for (const page of analysisPages(scenario)) {
    for (const node of graphNodes(page)) {
      const params = isRecord(node.params) ? node.params : {};
      for (const metricId of arrayOfStrings(params.metrics)) {
        const label = metricLabel(metricId);
        pushUniqueMetric(metrics, {
          id: metricId,
          label: label.label,
          role: 'analysis_metric',
          source: 'analysis',
          selector: stringValue(params.source_figure) ?? page.id,
          units: label.units,
          stageId: stage?.id ?? scenario.stage_id ?? null,
          scenarioId: scenario.id,
          sourceId: `${page.id}:${stringValue(node.id) ?? stringValue(node.label) ?? metricId}`,
          summary: stringValue(node.label) ?? page.name,
          metadata: {
            page_id: page.id,
            node_id: stringValue(node.id),
            node_type: stringValue(node.type),
          },
        });
      }
    }
  }
  return metrics;
}

function manifestMetrics(workspace: StudioWorkspaceSpec): ScenarioMetricSpec[] {
  const metrics: ScenarioMetricSpec[] = [];
  for (const { ref, stage } of allManifestRefs(workspace)) {
    for (const metricId of MANIFEST_METRIC_KEYS) {
      if (!(metricId in ref.metadata)) continue;
      const label = metricLabel(metricId);
      pushUniqueMetric(metrics, {
        id: metricId,
        label: label.label,
        role: 'observed_metric',
        source: 'manifest',
        selector: `manifest:${ref.id}.${metricId}`,
        units: label.units,
        stageId: stage?.id ?? null,
        scenarioId: stage?.scenario_id ?? null,
        sourceId: ref.id,
        summary: ref.role ?? ref.kind,
        metadata: {
          manifest_kind: ref.kind,
          manifest_role: ref.role,
          provider: ref.provider,
        },
      });
    }
  }
  return metrics;
}

function taskDefaultMetrics(
  scenario: StudioScenarioSpec,
  stage: StudioStageSpec | null
): ScenarioMetricSpec[] {
  const taskSpec = scenario.task_spec;
  if (!taskSpec || !isRecord(taskSpec.params) || !('n_targets' in taskSpec.params)) return [];
  const label = metricLabel('target_reach_error');
  return [
    {
      id: 'target_reach_error',
      label: label.label,
      role: 'task_default_metric',
      source: 'task_default',
      selector: `task:${taskSpec.type}.targets`,
      units: label.units,
      stageId: stage?.id ?? scenario.stage_id ?? null,
      scenarioId: scenario.id,
      sourceId: taskSpec.type,
      summary: `${taskSpec.params.n_targets} target task`,
      metadata: { task_type: taskSpec.type },
    },
  ];
}

export function scenarioMetricSpecs(workspace: StudioWorkspaceSpec | null): ScenarioMetricSpec[] {
  if (!workspace) return [];
  const metrics: ScenarioMetricSpec[] = [];
  const stageByScenarioId = new Map(
    workspace.stages.map((stage) => [stage.scenario_id ?? '', stage])
  );

  for (const scenario of Object.values(workspace.scenarios)) {
    const stage = stageByScenarioId.get(scenario.id) ?? null;
    for (const metric of objectiveMetrics(scenario, stage)) pushUniqueMetric(metrics, metric);
    for (const metric of analysisMetrics(scenario, stage)) pushUniqueMetric(metrics, metric);
    for (const metric of taskDefaultMetrics(scenario, stage)) pushUniqueMetric(metrics, metric);
  }
  for (const metric of manifestMetrics(workspace)) pushUniqueMetric(metrics, metric);

  return metrics.sort((a, b) => {
    const sourceOrder =
      ['objective', 'analysis', 'task_default', 'manifest'].indexOf(a.source) -
      ['objective', 'analysis', 'task_default', 'manifest'].indexOf(b.source);
    return sourceOrder || a.label.localeCompare(b.label);
  });
}

export function artifactOverlaysForWorkspace(
  workspace: StudioWorkspaceSpec | null
): ScenarioArtifactOverlay[] {
  if (!workspace) return [];
  const overlays: ScenarioArtifactOverlay[] = [];

  for (const { ref, stage } of allArtifactRefs(workspace)) {
    const overlay = isRecord(ref.metadata.workspace_overlay)
      ? ref.metadata.workspace_overlay
      : null;
    if (!overlay) continue;
    overlays.push({
      id: `${ref.id}:workspace-overlay`,
      label: stringValue(overlay.label) ?? ref.role ?? ref.kind,
      source: 'artifact',
      role: ref.role ?? ref.kind,
      stageId: stage?.id ?? null,
      artifactId: ref.id,
      uri: stringValue(overlay.uri) ?? ref.uri ?? null,
      mediaType: ref.media_type ?? null,
      metricIds: arrayOfStrings(overlay.metric_ids),
      summary: stringValue(overlay.summary),
      metadata: { ...overlay, provider: ref.provider },
    });
  }

  for (const stage of workspace.stages) {
    const scenario = stage.scenario_id ? workspace.scenarios[stage.scenario_id] : null;
    if (!scenario) continue;
    for (const page of analysisPages(scenario)) {
      for (const node of graphNodes(page)) {
        const params = isRecord(node.params) ? node.params : {};
        const overlay = isRecord(params.workspace_overlay) ? params.workspace_overlay : null;
        if (!overlay) continue;
        overlays.push({
          id: `${page.id}:${stringValue(node.id) ?? stringValue(overlay.uri) ?? 'workspace-overlay'}`,
          label: stringValue(overlay.label) ?? stringValue(node.label) ?? 'Workspace overlay',
          source: 'analysis',
          role: stringValue(node.type) ?? 'analysis_output',
          stageId: stage.id,
          artifactId: null,
          uri: stringValue(overlay.uri),
          mediaType: null,
          metricIds: arrayOfStrings(overlay.metric_ids),
          summary: page.name,
          metadata: { page_id: page.id, node_id: stringValue(node.id) },
        });
      }
    }
  }

  return overlays;
}

export function stageProductReferences(
  workspace: StudioWorkspaceSpec | null,
  stageId: string | null | undefined
): StageProductReference[] {
  const stage = workspace?.stages.find((candidate) => candidate.id === stageId);
  if (!workspace || !stage) return [];
  const scenario = stage.scenario_id ? workspace.scenarios[stage.scenario_id] : null;
  const references: StageProductReference[] = [];

  for (const [direction, collections] of [
    ['input_collection', stage.input_collections],
    ['output_collection', stage.output_collections],
  ] as const) {
    for (const collection of collections) {
      references.push({
        id: `${stage.id}:${direction}:${collection.id}`,
        label: `${direction === 'input_collection' ? 'Input' : 'Output'}: ${
          collection.label ?? collection.kind
        }`,
        kind: direction,
        stageId: stage.id,
        stageKind: stage.kind,
        sourceStageId: collection.source_stage_id ?? null,
        collectionId: collection.id,
        scenarioId: scenario?.id ?? null,
        itemCount: collection.item_refs.length,
        manifestIds: collection.item_refs.map((ref) => ref.id),
        artifactIds: [],
        summary: collection.kind,
        metadata: { ...collection.metadata },
      });
    }
  }

  if (scenario) {
    for (const page of analysisPages(scenario)) {
      references.push({
        id: `${stage.id}:analysis-page:${page.id}`,
        label: page.name,
        kind: 'analysis_page',
        stageId: stage.id,
        stageKind: stage.kind,
        sourceStageId: null,
        collectionId: null,
        scenarioId: scenario.id,
        itemCount: graphNodes(page).length,
        manifestIds: page.eval_run_id ? [page.eval_run_id] : [],
        artifactIds: [],
        summary: 'Analysis page',
        metadata: { page_id: page.id, active_eval_run_id: page.eval_run_id },
      });
    }
  }

  const reportSpec = isRecord(scenario?.report_spec) ? scenario.report_spec : null;
  const sections = Array.isArray(reportSpec?.sections) ? reportSpec.sections.filter(isRecord) : [];
  for (const section of sections) {
    references.push({
      id: `${stage.id}:report-section:${stringValue(section.id) ?? stringValue(section.title)}`,
      label: stringValue(section.title) ?? 'Report section',
      kind: 'report_section',
      stageId: stage.id,
      stageKind: stage.kind,
      sourceStageId: stringValue(section.source_stage_id),
      collectionId: stringValue(section.collection_id),
      scenarioId: scenario?.id ?? null,
      itemCount: arrayOfStrings(section.manifest_ids).length,
      manifestIds: arrayOfStrings(section.manifest_ids),
      artifactIds: arrayOfStrings(section.artifact_ids),
      summary: stringValue(section.role),
      metadata: { ...section },
    });
  }

  return references;
}
