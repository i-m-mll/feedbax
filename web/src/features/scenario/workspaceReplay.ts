import type {
  WorkspaceReplayProduct,
  WorkspaceReplayTrack,
  WorkspaceReplayTrial,
} from '@/generated/studioContracts';
import type {
  StudioArtifactRef,
  StudioCollectionRef,
  StudioManifestRef,
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioStageSpec,
  StudioWorkspaceSpec,
  WorkspaceComparisonSelection,
} from '@/types/workspace';

export interface WorkspaceReplayTimelineBand {
  id: string;
  label: string;
  start: number;
  end: number;
  kind: 'epoch' | 'loss_window';
}

export interface WorkspaceReplayEventTick {
  id: string;
  label: string;
  time: number;
}

export interface WorkspaceReplayModel {
  product: WorkspaceReplayProduct;
  source: 'embedded' | 'fixture';
  message: string;
  warnings: string[];
}

export interface WorkspaceReplaySourceModel extends WorkspaceReplayModel {
  ref: string;
  label: string;
}

export interface WorkspaceReplayComparisonMember {
  role: 'baseline' | 'candidate';
  ref: string;
  label: string;
  color: string;
  product: WorkspaceReplayProduct;
  trial: WorkspaceReplayTrial;
  track: WorkspaceReplayTrack | null;
}

export interface WorkspaceReplayComparisonModel {
  members: WorkspaceReplayComparisonMember[];
  sources: WorkspaceReplaySourceModel[];
  primaryTrial: WorkspaceReplayTrial | null;
  warnings: string[];
}

const REPLAY_COMPARISON_COLORS = {
  baseline: '#0f766e',
  candidate: '#7c3aed',
} as const;

export const WORKSPACE_REPLAY_FIXTURE: WorkspaceReplayProduct = {
  schema_id: 'feedbax.manifest.studio.workspace_replay',
  schema_version: 'feedbax.manifest.studio.workspace_replay.v1',
  product_kind: 'workspace_replay',
  source_mode: 'resolved_scene',
  trials: [
    {
      identity: {
        index: 0,
        stable_id: 'fixture-validation-0',
        source: 'stable_id',
        label: 'Validation fixture trial',
        metadata: { condition: 'validation' },
      },
      time: {
        length: 61,
        units: 's',
        dt: 0.02,
        values: Array.from({ length: 61 }, (_, index) => Number((index * 0.02).toFixed(2))),
      },
      tracks: [
        {
          anchor_id: 'effector',
          selector: {
            namespace: 'graph_output',
            compact: 'graph_output:effector',
            target_id: 'effector',
            role: 'observed',
          },
          samples: Array.from({ length: 61 }, (_, index) => {
            const progress = index / 60;
            return [
              -0.36 + progress * 0.72,
              -0.18 + Math.sin(progress * Math.PI) * 0.28,
            ];
          }),
          dim: 2,
          dtype: 'float32',
          units: 'm',
          frame: 'world',
          label: 'Effector',
        },
      ],
      trial_spec: {
        summary: { condition: 'validation', note: 'typed browser fixture' },
        timeline: {
          epochs: [
            { id: 'prep', label: 'Prep', start: 0, end: 0.28 },
            { id: 'movement', label: 'Movement', start: 0.28, end: 1.2 },
          ],
          events: [{ id: 'go_cue', label: 'Go cue', time: 0.28 }],
        },
        schema_id: 'feedbax.fixture.workspace_replay_trial',
        schema_version: 'v1',
      },
      manifest_refs: {
        checkpoint: {
          kind: 'checkpoint',
          id: 'fixture:checkpoint:browser',
          role: 'validation_checkpoint',
          provider: 'fixture',
          metadata: { checkpoint_step: 8000 },
        },
        spec_snapshot: {
          kind: 'trial_spec_snapshot',
          id: 'fixture:trial-spec:browser',
          role: 'spec_snapshot',
          provider: 'fixture',
          metadata: {},
        },
        seed: 7,
      },
      warnings: [
        {
          code: 'missing_manifest_refs',
          message: 'Using typed browser fixture because no workspace replay product is attached.',
        },
      ],
      metadata: { run_ref: 'fixture-run', checkpoint_step: 8000 },
    },
    {
      identity: {
        index: 1,
        stable_id: 'fixture-validation-1',
        source: 'stable_id',
        label: 'Validation fixture alternate',
        metadata: { condition: 'holdout' },
      },
      time: {
        length: 61,
        units: 's',
        dt: 0.02,
        values: Array.from({ length: 61 }, (_, index) => Number((index * 0.02).toFixed(2))),
      },
      tracks: [
        {
          anchor_id: 'effector',
          selector: {
            namespace: 'graph_output',
            compact: 'graph_output:effector',
            target_id: 'effector',
            role: 'observed',
          },
          samples: Array.from({ length: 61 }, (_, index) => {
            const progress = index / 60;
            return [
              -0.32 + progress * 0.64,
              -0.22 + Math.sin(progress * Math.PI) * 0.2 + 0.05,
            ];
          }),
          dim: 2,
          dtype: 'float32',
          units: 'm',
          frame: 'world',
          label: 'Effector',
        },
      ],
      trial_spec: {
        summary: { condition: 'holdout', note: 'static comparison fixture' },
        timeline: {
          epochs: [
            { id: 'prep', label: 'Prep', start: 0, end: 0.28 },
            { id: 'movement', label: 'Movement', start: 0.28, end: 1.2 },
          ],
          events: [{ id: 'go_cue', label: 'Go cue', time: 0.28 }],
        },
        schema_id: 'feedbax.fixture.workspace_replay_trial',
        schema_version: 'v1',
      },
      manifest_refs: { seed: 8 },
      warnings: [],
      metadata: { run_ref: 'fixture-run', checkpoint_step: 8000 },
    },
  ],
  warnings: [
    {
      code: 'missing_manifest_refs',
      message: 'No attached workspace replay product was found; Studio is showing typed fixture data.',
    },
  ],
  metadata: { source: 'studio_fixture' },
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function asWorkspaceReplayProduct(value: unknown): WorkspaceReplayProduct | null {
  if (!isRecord(value)) return null;
  if (value.product_kind !== 'workspace_replay' && value.schema_id !== 'feedbax.manifest.studio.workspace_replay') {
    return null;
  }
  if (!Array.isArray(value.trials)) return null;
  return value as WorkspaceReplayProduct;
}

function replayProductFromMetadata(metadata: Record<string, unknown> | undefined): WorkspaceReplayProduct | null {
  if (!metadata) return null;
  const candidates = [
    metadata.workspace_replay_product,
    metadata.workspace_replay,
    metadata.replay_product,
    metadata.product,
  ];
  for (const candidate of candidates) {
    const product = asWorkspaceReplayProduct(candidate);
    if (product) return product;
  }
  return null;
}

function replayProductFromArtifactRefs(
  refs: Array<{ metadata: Record<string, unknown> }> | undefined
): WorkspaceReplayProduct | null {
  for (const ref of refs ?? []) {
    const product = replayProductFromMetadata(ref.metadata);
    if (product) return product;
  }
  return null;
}

function replayWarnings(product: WorkspaceReplayProduct): string[] {
  return [
    ...(product.warnings ?? []).map((warning) => warning.message),
    ...(product.trials ?? []).flatMap((trial) =>
      (trial.warnings ?? []).map((warning) => warning.message)
    ),
  ];
}

function refDisplayLabel(
  ref: StudioArtifactRef | StudioManifestRef,
  fallbackPrefix: string
): string {
  const metadataLabel = ref.metadata.label;
  if (typeof metadataLabel === 'string' && metadataLabel.length > 0) return metadataLabel;
  if (ref.role) return `${ref.role}: ${ref.id}`;
  if (ref.kind) return `${ref.kind}: ${ref.id}`;
  return `${fallbackPrefix}: ${ref.id}`;
}

function pushReplaySource(
  sources: WorkspaceReplaySourceModel[],
  seen: Set<string>,
  ref: string,
  label: string,
  product: WorkspaceReplayProduct,
  message: string
) {
  if (seen.has(ref)) return;
  seen.add(ref);
  sources.push({
    ref,
    label,
    product,
    source: 'embedded',
    message,
    warnings: replayWarnings(product),
  });
}

function collectReplaySourcesFromRefs(
  sources: WorkspaceReplaySourceModel[],
  seen: Set<string>,
  refs: Array<StudioArtifactRef | StudioManifestRef> | undefined,
  owner: string
) {
  for (const ref of refs ?? []) {
    const product = replayProductFromMetadata(ref.metadata);
    if (!product) continue;
    pushReplaySource(
      sources,
      seen,
      ref.id,
      refDisplayLabel(ref, owner),
      product,
      `Eval playback uses ${owner} replay product ${ref.id}.`
    );
    if (ref.uri) {
      pushReplaySource(
        sources,
        seen,
        ref.uri,
        refDisplayLabel(ref, owner),
        product,
        `Eval playback uses ${owner} replay product ${ref.id}.`
      );
    }
  }
}

function collectReplaySourcesFromCollections(
  sources: WorkspaceReplaySourceModel[],
  seen: Set<string>,
  collections: StudioCollectionRef[] | undefined,
  owner: string
) {
  for (const collection of collections ?? []) {
    collectReplaySourcesFromRefs(
      sources,
      seen,
      collection.item_refs,
      `${owner} ${collection.label ?? collection.kind}`
    );
  }
}

function collectReplaySourcesFromMetadata(
  sources: WorkspaceReplaySourceModel[],
  seen: Set<string>,
  ref: string,
  label: string,
  metadata: Record<string, unknown> | undefined
) {
  const product = replayProductFromMetadata(metadata);
  if (!product) return;
  pushReplaySource(
    sources,
    seen,
    ref,
    label,
    product,
    'Eval playback uses attached workspace replay product.'
  );
}

export function resolveWorkspaceReplaySources(
  workspace: StudioWorkspaceSpec | null | undefined,
  stage: StudioStageSpec | null | undefined
): WorkspaceReplaySourceModel[] {
  const sources: WorkspaceReplaySourceModel[] = [];
  const seen = new Set<string>();

  collectReplaySourcesFromMetadata(
    sources,
    seen,
    stage ? `stage:${stage.id}:metadata` : 'stage:metadata',
    stage?.label ? `${stage.label} metadata` : 'Stage metadata',
    stage?.metadata
  );
  collectReplaySourcesFromRefs(sources, seen, stage?.artifact_refs, 'stage artifact');
  collectReplaySourcesFromRefs(sources, seen, stage?.manifest_refs, 'stage manifest');
  collectReplaySourcesFromCollections(sources, seen, stage?.input_collections, 'stage input');
  collectReplaySourcesFromCollections(sources, seen, stage?.output_collections, 'stage output');

  collectReplaySourcesFromMetadata(
    sources,
    seen,
    'workspace:metadata',
    workspace?.label ? `${workspace.label} metadata` : 'Workspace metadata',
    workspace?.metadata
  );
  collectReplaySourcesFromRefs(sources, seen, workspace?.artifact_refs, 'workspace artifact');
  collectReplaySourcesFromRefs(sources, seen, workspace?.manifest_refs, 'workspace manifest');
  collectReplaySourcesFromCollections(sources, seen, workspace?.collections, 'workspace');

  return sources;
}

export function resolveWorkspaceReplayModel(
  workspace: StudioWorkspaceSpec | null | undefined,
  stage: StudioStageSpec | null | undefined
): WorkspaceReplayModel {
  const source = resolveWorkspaceReplaySources(workspace, stage)[0];
  if (source) return source;

  const product =
    replayProductFromMetadata(stage?.metadata) ??
    replayProductFromArtifactRefs(stage?.artifact_refs) ??
    replayProductFromMetadata(workspace?.metadata) ??
    replayProductFromArtifactRefs(workspace?.artifact_refs) ??
    null;

  if (product) {
    return {
      product,
      source: 'embedded',
      message: 'Eval playback uses attached workspace replay product.',
      warnings: replayWarnings(product),
    };
  }

  return {
    product: WORKSPACE_REPLAY_FIXTURE,
    source: 'fixture',
    message: 'No replay artifact is attached; showing typed fixture playback.',
    warnings: (WORKSPACE_REPLAY_FIXTURE.warnings ?? []).map((warning) => warning.message),
  };
}

export function workspaceReplayTrialRef(trial: WorkspaceReplayTrial): string {
  if (trial.identity.stable_id) return `trial:${trial.identity.stable_id}`;
  return `trial:index:${trial.identity.index}`;
}

export function workspaceReplayTrialLabel(trial: WorkspaceReplayTrial): string {
  return trial.identity.label ?? trial.identity.stable_id ?? `Trial ${trial.identity.index + 1}`;
}

export function selectWorkspaceReplayTrial(
  product: WorkspaceReplayProduct,
  selectedRef: string | null | undefined
): WorkspaceReplayTrial | null {
  const trials = product.trials ?? [];
  if (trials.length === 0) return null;
  return (
    trials.find((trial) => workspaceReplayTrialRef(trial) === selectedRef) ??
    trials[0] ??
    null
  );
}

function sameTrialIdentity(left: WorkspaceReplayTrial, right: WorkspaceReplayTrial): boolean {
  return (
    workspaceReplayTrialRef(left) === workspaceReplayTrialRef(right) ||
    Boolean(left.identity.stable_id && left.identity.stable_id === right.identity.stable_id) ||
    left.identity.index === right.identity.index
  );
}

function selectComparableWorkspaceReplayTrial(
  product: WorkspaceReplayProduct,
  selectedRef: string | null | undefined,
  baselineTrial: WorkspaceReplayTrial | null
): WorkspaceReplayTrial | null {
  const selected = selectWorkspaceReplayTrial(product, selectedRef);
  const trials = product.trials ?? [];
  if (!baselineTrial) return selected;
  return (
    trials.find((trial) => sameTrialIdentity(trial, baselineTrial)) ??
    selected ??
    null
  );
}

export function resolveWorkspaceReplayComparison(
  sources: WorkspaceReplaySourceModel[],
  selection: WorkspaceComparisonSelection,
  selectedTrialRef: string | null | undefined
): WorkspaceReplayComparisonModel {
  const sourceByRef = new Map(sources.map((source) => [source.ref, source]));
  const baselineSource =
    (selection.baseline_ref ? sourceByRef.get(selection.baseline_ref) : undefined) ??
    null;
  const candidateSource =
    (selection.candidate_ref ? sourceByRef.get(selection.candidate_ref) : undefined) ??
    null;
  const baselineTrial = baselineSource
    ? selectWorkspaceReplayTrial(baselineSource.product, selectedTrialRef)
    : null;
  const candidateTrial = candidateSource
    ? selectComparableWorkspaceReplayTrial(
        candidateSource.product,
        selectedTrialRef,
        baselineTrial
      )
    : null;

  const members: WorkspaceReplayComparisonMember[] = [];
  if (baselineSource && baselineTrial) {
    members.push({
      role: 'baseline',
      ref: baselineSource.ref,
      label: baselineSource.label,
      color: REPLAY_COMPARISON_COLORS.baseline,
      product: baselineSource.product,
      trial: baselineTrial,
      track: primaryWorkspaceReplayTrack(baselineTrial),
    });
  }
  if (candidateSource && candidateTrial && candidateSource.ref !== baselineSource?.ref) {
    members.push({
      role: 'candidate',
      ref: candidateSource.ref,
      label: candidateSource.label,
      color: REPLAY_COMPARISON_COLORS.candidate,
      product: candidateSource.product,
      trial: candidateTrial,
      track: primaryWorkspaceReplayTrack(candidateTrial),
    });
  }

  const warnings = [
    ...(baselineSource?.warnings ?? []),
    ...(candidateSource?.warnings ?? []),
  ];

  return {
    members,
    sources,
    primaryTrial: baselineTrial ?? candidateTrial,
    warnings,
  };
}

export function primaryWorkspaceReplayTrack(
  trial: WorkspaceReplayTrial | null | undefined
): WorkspaceReplayTrack | null {
  return (
    trial?.tracks?.find((track) => track.dim >= 2 && track.samples.length > 0) ??
    trial?.tracks?.[0] ??
    null
  );
}

export function workspaceReplayFrameTimes(trial: WorkspaceReplayTrial | null | undefined): number[] {
  if (!trial) return [0];
  const explicit = trial.time.values?.filter(Number.isFinite);
  if (explicit && explicit.length === trial.time.length) return explicit;
  const dt = typeof trial.time.dt === 'number' && Number.isFinite(trial.time.dt)
    ? trial.time.dt
    : 1;
  return Array.from({ length: Math.max(1, trial.time.length) }, (_, index) => index * dt);
}

export function workspaceReplayDuration(trial: WorkspaceReplayTrial | null | undefined): number {
  const times = workspaceReplayFrameTimes(trial);
  return times[times.length - 1] ?? 0;
}

export function workspaceReplayFrameIndex(trial: WorkspaceReplayTrial, position: number): number {
  const times = workspaceReplayFrameTimes(trial);
  const bounded = Math.max(0, Math.min(workspaceReplayDuration(trial), position));
  let best = 0;
  for (let index = 1; index < times.length; index += 1) {
    if (Math.abs(times[index] - bounded) < Math.abs(times[best] - bounded)) best = index;
  }
  return best;
}

export function workspaceReplaySampleAt(
  track: WorkspaceReplayTrack | null | undefined,
  frameIndex: number
): [number, number] | null {
  const sample = track?.samples[Math.max(0, Math.min(track.samples.length - 1, frameIndex))];
  if (!sample || sample.length < 2) return null;
  const x = Number(sample[0]);
  const y = Number(sample[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

export function workspaceReplayPolyline(track: WorkspaceReplayTrack | null | undefined): Array<[number, number]> {
  if (!track) return [];
  return track.samples
    .map((sample) => {
      const x = Number(sample[0]);
      const y = Number(sample[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? ([x, y] as [number, number]) : null;
    })
    .filter((point): point is [number, number] => point !== null);
}

function numberFromRecord(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = Number(record[key]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

export function workspaceReplayTimelineBands(trial: WorkspaceReplayTrial | null | undefined): WorkspaceReplayTimelineBand[] {
  const timeline = trial?.trial_spec?.timeline;
  if (!isRecord(timeline) || !Array.isArray(timeline.epochs)) return [];
  let cursor = 0;
  return timeline.epochs.flatMap((epoch, index) => {
    if (!isRecord(epoch)) return [];
    const start = numberFromRecord(epoch, ['start', 'start_time', 'start_s']) ?? cursor;
    const explicitEnd = numberFromRecord(epoch, ['end', 'end_time', 'end_s']);
    const length = numberFromRecord(epoch, ['length', 'duration', 'duration_s']);
    const end = explicitEnd ?? start + (length ?? 0);
    cursor = end;
    if (end <= start) return [];
    return [{
      id: String(epoch.id ?? `epoch:${index}`),
      label: String(epoch.label ?? epoch.id ?? `Epoch ${index + 1}`),
      start,
      end,
      kind: 'epoch' as const,
    }];
  });
}

export function workspaceReplayEventTicks(trial: WorkspaceReplayTrial | null | undefined): WorkspaceReplayEventTick[] {
  const timeline = trial?.trial_spec?.timeline;
  if (!isRecord(timeline) || !Array.isArray(timeline.events)) return [];
  return timeline.events.flatMap((event, index) => {
    if (!isRecord(event)) return [];
    const time = numberFromRecord(event, ['time', 'at', 'start', 'step']);
    if (time === null) return [];
    return [{
      id: String(event.id ?? `event:${index}`),
      label: String(event.label ?? event.id ?? `Event ${index + 1}`),
      time,
    }];
  });
}

export function objectiveLossWindowBands(
  objectiveSpec: StudioObjectiveSpec,
  duration: number
): WorkspaceReplayTimelineBand[] {
  return objectiveSpec.terms.flatMap((term) => lossWindowBandsForTerm(term, duration));
}

function lossWindowBandsForTerm(
  term: StudioObjectiveTermSpec,
  duration: number
): WorkspaceReplayTimelineBand[] {
  const selector = term.temporal_selector;
  if (selector && typeof selector === 'object') {
    if (selector.mode === 'range' && typeof selector.start === 'number' && typeof selector.end === 'number') {
      const band: WorkspaceReplayTimelineBand = {
        id: `loss:${term.id}:range`,
        label: term.label,
        start: Math.max(0, selector.start),
        end: Math.min(duration, selector.end),
        kind: 'loss_window',
      };
      return band.end > band.start ? [band] : [];
    }
    if (selector.mode === 'final') {
      return [{
        id: `loss:${term.id}:final`,
        label: term.label,
        start: Math.max(0, duration * 0.92),
        end: duration,
        kind: 'loss_window',
      }];
    }
  }

  const timing = term.metadata.target_timing;
  if (!isRecord(timing)) return [];
  const targetSpec = timing.target_spec;
  const source = isRecord(targetSpec) ? targetSpec : timing;
  const start = numberFromRecord(source, ['start', 'start_time', 'start_s']);
  const end = numberFromRecord(source, ['end', 'end_time', 'end_s']);
  if (start === null || end === null || end <= start) return [];
  return [{
    id: `loss:${term.id}:metadata`,
    label: term.label,
    start: Math.max(0, start),
    end: Math.min(duration, end),
    kind: 'loss_window',
  }];
}

export function workspaceReplayProvenance(trial: WorkspaceReplayTrial | null | undefined): string {
  if (!trial) return 'No trial selected';
  const refs = trial.manifest_refs;
  const run =
    typeof trial.metadata?.run_ref === 'string'
      ? trial.metadata.run_ref
      : typeof refs?.producer_manifest?.id === 'string'
        ? refs.producer_manifest.id
        : null;
  const checkpoint =
    typeof trial.metadata?.checkpoint_step === 'number'
      ? `checkpoint ${trial.metadata.checkpoint_step}`
      : typeof refs?.checkpoint?.id === 'string'
        ? refs.checkpoint.id
        : null;
  const trialLabel = workspaceReplayTrialLabel(trial);
  return ['Eval playback', run, trialLabel, checkpoint].filter(Boolean).join(' - ');
}
