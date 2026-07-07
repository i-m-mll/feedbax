import type {
  StudioCollectionRef,
  StudioManifestRef,
  StudioStageKind,
  StudioStageSpec,
  StudioWorkspaceSpec,
} from '@/types/workspace';

export interface TrainingRunSummary {
  id: string;
  label: string;
  status: string;
  variant: string | null;
  rampShape: string | null;
  rampDurationSteps: number | null;
  nnOutputPreGo: number | null;
  finalValidationLoss: number | null;
  velocityRmse: number | null;
  peakVelocityMean: number | null;
  peakVelocitySd: number | null;
  holdDriftMeanMm: number | null;
  holdDriftSdMm: number | null;
  metrics: Record<string, number | null>;
  replicateCount: number | null;
  batchSize: number | null;
  warmupBatches: number | null;
  checkpointAvailable: boolean;
  sourceIssue: string | null;
  provenanceId: string;
  uri: string | null;
  jobId: string | null;
  axisCoordinates: Record<string, unknown>;
  runSetId: string | null;
  planned: boolean;
  supersededBy: string | null;
}

export interface EvaluationRunSummary {
  id: string;
  label: string;
  status: string;
  selectedTrainingRunId: string | null;
  trainingRunIds: string[];
  targets: string | null;
  sisu: number | null;
  perturbation: string | null;
  sourceIssue: string | null;
  provenanceId: string;
  uri: string | null;
}

export type LineageNodeSource = 'collection-input' | 'collection-output' | 'stage-manifest' | 'workspace-manifest' | 'parent-ref';

export interface LineageProjectionNode {
  id: string;
  label: string;
  kind: string;
  role: string | null;
  status: string;
  statusReason: string | null;
  runSetId: string | null;
  stageId: string | null;
  stageKind: StudioStageKind | 'unknown';
  stageLabel: string | null;
  collectionIds: string[];
  focusStageId: string | null;
  focusCollectionId: string | null;
  source: LineageNodeSource;
}

export interface LineageProjectionEdge {
  id: string;
  parentId: string;
  childId: string;
  role: string | null;
  status: string | null;
  reason: string | null;
}

export interface LineageProjectionGroup {
  id: string;
  label: string;
  nodeIds: string[];
}

export interface LineageProjection {
  nodes: LineageProjectionNode[];
  edges: LineageProjectionEdge[];
  groups: LineageProjectionGroup[];
}

export function selectedIds(stage: StudioStageSpec | null | undefined, key: string): string[] {
  const value = stage?.selection_spec[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

export function trainingRunSummaries(stage: StudioStageSpec | null | undefined): TrainingRunSummary[] {
  const refs = uniqueRefs(
    stage?.output_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'training_run' || ref.kind === 'TrainingRun')
    .map(trainingRunSummary)
    .sort((a, b) => {
      if (a.finalValidationLoss === null && b.finalValidationLoss === null) {
        return a.label.localeCompare(b.label);
      }
      if (a.finalValidationLoss === null) return 1;
      if (b.finalValidationLoss === null) return -1;
      return a.finalValidationLoss - b.finalValidationLoss;
    });
}

export function trainingInputSummaries(
  stage: StudioStageSpec | null | undefined
): TrainingRunSummary[] {
  const refs = uniqueRefs(
    stage?.input_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'training_run' || ref.kind === 'TrainingRun')
    .map(trainingRunSummary)
    .sort((a, b) => {
      if (a.finalValidationLoss === null && b.finalValidationLoss === null) {
        return a.label.localeCompare(b.label);
      }
      if (a.finalValidationLoss === null) return 1;
      if (b.finalValidationLoss === null) return -1;
      return a.finalValidationLoss - b.finalValidationLoss;
    });
}

export function evaluationRunSummaries(
  stage: StudioStageSpec | null | undefined
): EvaluationRunSummary[] {
  const refs = uniqueRefs(
    stage?.output_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'evaluation_run' || ref.kind === 'EvaluationRun')
    .map(evaluationRunSummary);
}

export function bestTrainingRun(rows: TrainingRunSummary[]): TrainingRunSummary | null {
  return rows.find((row) => row.finalValidationLoss !== null) ?? rows[0] ?? null;
}

export function formatMetric(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return 'Not recorded';
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(digits);
  return value.toPrecision(digits);
}

export function trainingRunMetricValue(
  row: TrainingRunSummary,
  metricId: string
): number | null {
  return row.metrics[metricId] ?? null;
}

export function runParameterSummary(row: TrainingRunSummary): string {
  const parts = [
    row.rampShape ? `${capitalize(row.rampShape)} ramp` : null,
    row.rampDurationSteps !== null ? `${row.rampDurationSteps} steps` : null,
    row.nnOutputPreGo !== null ? `pre-go ${row.nnOutputPreGo}` : null,
    row.replicateCount !== null ? `${row.replicateCount} reps` : null,
  ];
  return parts.filter(Boolean).join(' - ') || 'Parameters not recorded';
}

export function evaluationProtocolLabel(row: EvaluationRunSummary): string {
  const parts = [
    row.targets,
    row.sisu !== null ? `SISU ${row.sisu}` : null,
    row.perturbation ? perturbationLabel(row.perturbation) : null,
  ];
  return parts.filter(Boolean).join(' - ') || 'Protocol not recorded';
}

export function buildLineageProjection(
  workspace: StudioWorkspaceSpec | null | undefined
): LineageProjection {
  const nodes = new Map<string, LineageProjectionNode>();
  const refsById = new Map<string, StudioManifestRef>();
  const edgeMap = new Map<string, LineageProjectionEdge>();
  const stages = workspace?.stages ?? [];
  const stageById = new Map(stages.map((stage) => [stage.id, stage]));

  const addNode = (
    ref: StudioManifestRef,
    options: {
      source: LineageNodeSource;
      stage: StudioStageSpec | null;
      collection: StudioCollectionRef | null;
    }
  ) => {
    refsById.set(ref.id, ref);
    const existing = nodes.get(ref.id);
    const existingCollections = existing?.collectionIds ?? [];
    const collectionIds = options.collection
      ? uniqueStrings([...existingCollections, options.collection.id])
      : existingCollections;
    const next = nodeFromRef(ref, options, collectionIds);
    if (!existing || lineageSourceRank(next.source) >= lineageSourceRank(existing.source)) {
      nodes.set(ref.id, {
        ...next,
        collectionIds,
      });
      return;
    }
    nodes.set(ref.id, {
      ...existing,
      collectionIds,
    });
  };

  for (const collection of workspace?.collections ?? []) {
    const stage = collection.source_stage_id ? stageById.get(collection.source_stage_id) ?? null : null;
    for (const ref of collection.item_refs) {
      addNode(ref, { source: 'collection-output', stage, collection });
    }
  }

  for (const ref of workspace?.manifest_refs ?? []) {
    addNode(ref, { source: 'workspace-manifest', stage: null, collection: null });
  }

  for (const stage of stages) {
    for (const collection of stage.input_collections) {
      for (const ref of collection.item_refs) {
        addNode(ref, { source: 'collection-input', stage, collection });
      }
    }
    for (const collection of stage.output_collections) {
      for (const ref of collection.item_refs) {
        addNode(ref, { source: 'collection-output', stage, collection });
      }
    }
    for (const ref of stage.manifest_refs) {
      addNode(ref, { source: 'stage-manifest', stage, collection: null });
    }
  }

  for (const [childId, ref] of refsById) {
    for (const parent of parentRefsForManifest(ref)) {
      if (!nodes.has(parent.id)) {
        nodes.set(parent.id, nodeFromParentRef(parent));
      }
      const edge: LineageProjectionEdge = {
        id: `${parent.id}->${childId}:${parent.role ?? parent.kind}`,
        parentId: parent.id,
        childId,
        role: parent.role ?? null,
        status: stringValue(parent.metadata?.status),
        reason: statusReason(parent.metadata ?? {}),
      };
      edgeMap.set(edge.id, edge);
    }
  }

  const orderedNodes = Array.from(nodes.values()).sort(compareLineageNodes);
  const orderedEdges = Array.from(edgeMap.values()).sort((a, b) => {
    const parent = a.parentId.localeCompare(b.parentId);
    if (parent !== 0) return parent;
    return a.childId.localeCompare(b.childId);
  });
  return {
    nodes: orderedNodes,
    edges: orderedEdges,
    groups: lineageGroups(orderedNodes),
  };
}

function uniqueRefs(refs: StudioManifestRef[]): StudioManifestRef[] {
  const byId = new Map<string, StudioManifestRef>();
  for (const ref of refs) byId.set(ref.id, ref);
  return Array.from(byId.values());
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => value.trim().length > 0)));
}

function trainingRunSummary(ref: StudioManifestRef): TrainingRunSummary {
  const hyperparams = objectValue(ref.metadata.hyperparams);
  const typedCheckpointAvailable = booleanValue(ref.metadata.checkpoint_available);
  const axisCoordinates = axisCoordinatesFromRef(ref);
  return {
    id: ref.id,
    label: stringValue(ref.metadata.name) ?? ref.id,
    status: stringValue(ref.metadata.status) ?? 'unknown',
    variant: stringValue(metadataValue(ref, 'run_variant')),
    rampShape: stringValue(metadataValue(ref, 'ramp_shape')),
    rampDurationSteps: numberValue(metadataValue(ref, 'ramp_duration_steps')),
    nnOutputPreGo: numberValue(metadataValue(ref, 'nn_output_pre_go')),
    finalValidationLoss: numberValue(metadataValue(ref, 'final_validation_loss')),
    velocityRmse: numberValue(metadataValue(ref, 'within_cell_velocity_rmse_m_per_s')),
    peakVelocityMean: nestedNumber(metadataValue(ref, 'peak_velocity_m_per_s'), 'mean'),
    peakVelocitySd: nestedNumber(metadataValue(ref, 'peak_velocity_m_per_s'), 'sd'),
    holdDriftMeanMm: nestedNumber(metadataValue(ref, 'hold_drift_mm'), 'mean'),
    holdDriftSdMm: nestedNumber(metadataValue(ref, 'hold_drift_mm'), 'sd'),
    metrics: trainingRunMetrics(ref),
    replicateCount: numberValue(hyperparams?.n_replicates ?? ref.metadata.n_replicates),
    batchSize: numberValue(hyperparams?.batch_size ?? ref.metadata.batch_size),
    warmupBatches: numberValue(
      hyperparams?.n_warmup_batches ?? ref.metadata.n_warmup_batches
    ),
    checkpointAvailable:
      typedCheckpointAvailable ??
      (Boolean(ref.uri) || stringValue(ref.metadata.checkpoint_uri) !== null),
    sourceIssue: stringValue(ref.metadata.source_issue),
    provenanceId: stringValue(ref.metadata.provenance_id) ?? ref.id,
    uri: ref.uri ?? null,
    jobId: stringValue(ref.metadata.job_id),
    axisCoordinates,
    runSetId: stringValue(ref.metadata.run_set_id),
    planned: booleanValue(ref.metadata.planned) ?? false,
    supersededBy: stringValue(ref.metadata.superseded_by),
  };
}

function nodeFromRef(
  ref: StudioManifestRef,
  {
    source,
    stage,
    collection,
  }: {
    source: LineageNodeSource;
    stage: StudioStageSpec | null;
    collection: StudioCollectionRef | null;
  },
  collectionIds: string[]
): LineageProjectionNode {
  const stageKind = stage?.kind ?? stageKindForRef(ref);
  const focusCollectionId = collection?.id ?? collectionIds[0] ?? null;
  return {
    id: ref.id,
    label: stringValue(ref.metadata.name) ?? stringValue(ref.metadata.label) ?? ref.id,
    kind: ref.kind,
    role: ref.role ?? null,
    status: stringValue(ref.metadata.status) ?? 'unknown',
    statusReason: statusReason(ref.metadata),
    runSetId: stringValue(ref.metadata.run_set_id),
    stageId: stage?.id ?? null,
    stageKind,
    stageLabel: stage?.label ?? null,
    collectionIds,
    focusStageId: stage?.id ?? null,
    focusCollectionId,
    source,
  };
}

function nodeFromParentRef(parent: ParentRefLike): LineageProjectionNode {
  return {
    id: parent.id,
    label: stringValue(parent.metadata?.name) ?? parent.id,
    kind: parent.kind,
    role: parent.role ?? null,
    status: stringValue(parent.metadata?.status) ?? 'unknown',
    statusReason: statusReason(parent.metadata ?? {}),
    runSetId: stringValue(parent.metadata?.run_set_id),
    stageId: null,
    stageKind: stageKindForManifestKind(parent.kind),
    stageLabel: null,
    collectionIds: [],
    focusStageId: null,
    focusCollectionId: null,
    source: 'parent-ref',
  };
}

interface ParentRefLike {
  kind: string;
  id: string;
  role?: string | null;
  uri?: string | null;
  metadata?: Record<string, unknown>;
}

function parentRefsForManifest(ref: StudioManifestRef): ParentRefLike[] {
  const refs = [
    ...parentRefArray(ref.metadata.parent_refs),
    ...parentRefArray(ref.metadata.inputs),
    ...parentRefArray(ref.metadata.input_training_runs),
    ...parentRefArray(objectValue(ref.metadata.provenance)?.parents),
  ];
  const trainingRunIds = arrayOfStrings(ref.metadata.training_run_ids);
  const evalRunIds = [
    ...arrayOfStrings(ref.metadata.eval_run_ids),
    ...arrayOfStrings(ref.metadata.evaluation_run_ids),
  ];
  const fallbackIds = [
    ...trainingRunIds.map((id) => ({
      kind: 'TrainingRunManifest',
      id,
      role: 'training_run',
    })),
    ...evalRunIds.map((id) => ({
      kind: 'EvaluationRunManifest',
      id,
      role: 'evaluation_run',
    })),
    ...arrayOfStrings(ref.metadata.analysis_product_ids).map((id) => ({
      kind: 'AnalysisRunManifest',
      id,
      role: 'analysis_run',
    })),
  ];
  const selectedTrainingRunId = stringValue(ref.metadata.selected_training_run_id);
  const selectedEvalRunId = stringValue(ref.metadata.selected_eval_run_id);
  if (selectedTrainingRunId && !trainingRunIds.includes(selectedTrainingRunId)) {
    fallbackIds.push({
      kind: 'TrainingRunManifest',
      id: selectedTrainingRunId,
      role: 'selected_training_run',
    });
  }
  if (selectedEvalRunId && !evalRunIds.includes(selectedEvalRunId)) {
    fallbackIds.push({
      kind: 'EvaluationRunManifest',
      id: selectedEvalRunId,
      role: 'selected_evaluation_run',
    });
  }
  return uniqueParentRefs([...refs, ...fallbackIds]);
}

function parentRefArray(value: unknown): ParentRefLike[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      const ref = objectValue(item);
      const id = stringValue(ref?.id);
      const kind = stringValue(ref?.kind);
      if (!id || !kind) return null;
      return {
        kind,
        id,
        role: stringValue(ref?.role),
        uri: stringValue(ref?.uri),
        metadata: objectValue(ref?.metadata) ?? undefined,
      };
    })
    .filter((item): item is ParentRefLike => item !== null);
}

function uniqueParentRefs(refs: ParentRefLike[]): ParentRefLike[] {
  const byKey = new Map<string, ParentRefLike>();
  for (const ref of refs) {
    const key = `${ref.kind}:${ref.id}:${ref.role ?? ''}`;
    if (!byKey.has(key)) byKey.set(key, ref);
  }
  return Array.from(byKey.values());
}

function lineageGroups(nodes: LineageProjectionNode[]): LineageProjectionGroup[] {
  const groups = new Map<string, LineageProjectionGroup>();
  for (const node of nodes) {
    const id = node.runSetId ? `run-set:${node.runSetId}` : `stage:${node.stageKind}`;
    const label = node.runSetId ? `Run set ${node.runSetId}` : lineageStageLabel(node.stageKind);
    const group = groups.get(id) ?? { id, label, nodeIds: [] };
    group.nodeIds.push(node.id);
    groups.set(id, group);
  }
  return Array.from(groups.values()).sort((a, b) => groupRank(a.id) - groupRank(b.id));
}

function compareLineageNodes(a: LineageProjectionNode, b: LineageProjectionNode): number {
  const stage = stageRank(a.stageKind) - stageRank(b.stageKind);
  if (stage !== 0) return stage;
  const runSet = (a.runSetId ?? '').localeCompare(b.runSetId ?? '');
  if (runSet !== 0) return runSet;
  return a.label.localeCompare(b.label);
}

function groupRank(groupId: string): number {
  if (groupId.startsWith('run-set:')) return 0;
  const stage = groupId.slice('stage:'.length) as StudioStageKind | 'unknown';
  return stageRank(stage) + 1;
}

function stageRank(stage: StudioStageKind | 'unknown'): number {
  return ['train', 'eval', 'analysis', 'report', 'import', 'compare', 'export', 'protocol', 'unknown']
    .indexOf(stage);
}

function lineageSourceRank(source: LineageNodeSource): number {
  return {
    'parent-ref': 0,
    'workspace-manifest': 1,
    'collection-input': 2,
    'collection-output': 3,
    'stage-manifest': 4,
  }[source];
}

function stageKindForRef(ref: StudioManifestRef): StudioStageKind | 'unknown' {
  return stageKindForManifestKind(ref.kind, ref.role);
}

function stageKindForManifestKind(kind: string, role?: string | null): StudioStageKind | 'unknown' {
  const key = `${kind} ${role ?? ''}`.toLowerCase();
  if (key.includes('training')) return 'train';
  if (key.includes('evaluation') || key.includes('eval')) return 'eval';
  if (key.includes('analysis')) return 'analysis';
  if (key.includes('report')) return 'report';
  return 'unknown';
}

function lineageStageLabel(stage: StudioStageKind | 'unknown'): string {
  return stage === 'eval' ? 'Evaluation' : capitalize(stage);
}

function statusReason(metadata: Record<string, unknown>): string | null {
  return (
    stringValue(metadata.status_reason) ??
    stringValue(metadata.staleness_reason) ??
    stringValue(metadata.skip_reason) ??
    stringValue(metadata.not_applicable_reason) ??
    stringValue(metadata.unavailable_reason) ??
    stringValue(metadata.failure_reason) ??
    stringValue(metadata.reason)
  );
}

function trainingRunMetrics(ref: StudioManifestRef): Record<string, number | null> {
  const metrics: Record<string, number | null> = {};
  const typedMetrics = objectValue(ref.metadata.metrics);
  if (typedMetrics) {
    for (const [key, value] of Object.entries(typedMetrics)) {
      const direct = numberValue(value);
      const mean = nestedNumber(value, 'mean');
      if (direct !== null) metrics[key] = direct;
      else if (mean !== null) metrics[key] = mean;
    }
  }
  for (const [key, value] of Object.entries(ref.metadata)) {
    const direct = numberValue(value);
    const mean = nestedNumber(value, 'mean');
    if (direct !== null) metrics[key] = direct;
    else if (mean !== null) metrics[key] = mean;
  }
  return metrics;
}

function metadataValue(ref: StudioManifestRef, key: string): unknown {
  const direct = ref.metadata[key];
  if (direct !== undefined) return direct;
  const metrics = objectValue(ref.metadata.metrics);
  if (metrics && metrics[key] !== undefined) return metrics[key];
  const hyperparams = objectValue(ref.metadata.hyperparams);
  return hyperparams?.[key];
}

function axisCoordinatesFromRef(ref: StudioManifestRef): Record<string, unknown> {
  const studio = objectValue(ref.metadata.studio);
  const direct = objectValue(ref.metadata.axis_coordinates);
  const nested = objectValue(studio?.axis_coordinates);
  const coordinates = nested ?? direct ?? {};
  const hyperparams = objectValue(ref.metadata.hyperparams);
  const axisHyperparams = Object.fromEntries(
    Object.entries(hyperparams ?? ref.metadata)
      .filter(([key]) => key.startsWith('axis_'))
      .map(([key, value]) => [key.slice('axis_'.length), value])
  );
  return {
    ...axisHyperparams,
    ...coordinates,
  };
}

function evaluationRunSummary(ref: StudioManifestRef): EvaluationRunSummary {
  const protocol = objectValue(ref.metadata.eval_protocol);
  const trainingRunIds = arrayOfStrings(ref.metadata.training_run_ids);
  return {
    id: ref.id,
    label: stringValue(ref.metadata.name) ?? ref.id,
    status: stringValue(ref.metadata.status) ?? 'unknown',
    selectedTrainingRunId: stringValue(ref.metadata.selected_training_run_id),
    trainingRunIds,
    targets: stringValue(protocol?.targets),
    sisu: numberValue(protocol?.sisu),
    perturbation: stringValue(protocol?.perturbation),
    sourceIssue: stringValue(ref.metadata.source_issue),
    provenanceId: ref.id,
    uri: ref.uri ?? null,
  };
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function booleanValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function nestedNumber(value: unknown, key: string): number | null {
  return numberValue(objectValue(value)?.[key]);
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function capitalize(value: string): string {
  return value.length === 0 ? value : `${value[0].toUpperCase()}${value.slice(1)}`;
}

function perturbationLabel(value: string): string {
  return value === 'none' ? 'no perturbation' : `${value} perturbation`;
}
