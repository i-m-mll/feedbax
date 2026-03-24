/**
 * RLRMP Part 1 project template for Feedbax Studio.
 *
 * Generates a complete analysis project with 4 pages mirroring the
 * Part 1 RLRMP analysis modules:
 *   1. plant_perts   — Plant perturbation analysis (curl/gusts)
 *   2. feedback_perts — Feedback impulse perturbation analysis
 *   3. freq_response  — Frequency response analysis (FFT-based)
 *   4. unit_prefs     — Unit preference tuning analysis
 *
 * Each page declares analysis nodes, dependency nodes, transforms,
 * wires (with state field paths), and eval parametrization matching
 * the Python analysis module declarations.
 */

import type {
  AnalysisNodeSpec,
  AnalysisWire,
  AnalysisGraphSpec,
  AnalysisPageSpec,
  AnalysisSnapshot,
  EvalParametrization,
  TransformSpec,
} from '@/types/analysis';

// ---------------------------------------------------------------------------
// ID generation helpers — deterministic, prefixed per page for uniqueness
// ---------------------------------------------------------------------------

function nodeId(page: string, name: string): string {
  return `${page}__${name}`;
}

function wireId(page: string, index: number): string {
  return `${page}__wire_${index}`;
}

function transformId(page: string, name: string): string {
  return `${page}__transform_${name}`;
}

// ---------------------------------------------------------------------------
// Data source ID — shared constant
// ---------------------------------------------------------------------------

const DATA_SOURCE_ID = '__data_source__';

// ---------------------------------------------------------------------------
// Page 1: plant_perts
//
// Source: rlrmp/modules/analysis/part1/plant_perts.py
//
// Pipeline:
//   DataSource → GetBestReplicate → AlignedVars → ApplyFns(measures) → Violins
//   DataSource → GetBestReplicate → AlignedTrajectories (by pert_amp, aligned)
//   DataSource → GetBestReplicate → AlignedTrajectories (by pert_amp, no align)
//   DataSource → GetBestReplicate → Profiles (PCA of hidden states)
//
// Eval: Plant perturbation (curl or gusts), multiple amplitudes
// ---------------------------------------------------------------------------

function buildPlantPertsPage(): AnalysisPageSpec {
  const P = 'plant_perts';

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Preprocessing nodes
    [nodeId(P, 'get_best_replicate')]: {
      id: nodeId(P, 'get_best_replicate'),
      type: 'GetBestReplicate',
      label: 'GetBestReplicate',
      category: 'Preprocessing',
      inputPorts: ['data'],
      outputPorts: ['data'],
      params: { axis: 0 },
      role: 'dependency',
    },

    // Computation: AlignedVars for measure extraction
    [nodeId(P, 'aligned_vars')]: {
      id: nodeId(P, 'aligned_vars'),
      type: 'AlignedVars',
      label: 'AlignedVars',
      category: 'Computation',
      inputPorts: ['data'],
      outputPorts: ['aligned_vars'],
      params: {
        varset: ['pos', 'vel', 'command', 'force'],
        directions_fn: 'reach_directions',
        align_epoch: null,
      },
      role: 'dependency',
    },

    // Computation: ApplyFns to extract measures from aligned vars
    [nodeId(P, 'apply_fns_measures')]: {
      id: nodeId(P, 'apply_fns_measures'),
      type: 'ApplyFns',
      label: 'ApplyFns (measures)',
      category: 'Computation',
      inputPorts: ['input'],
      outputPorts: ['measures'],
      params: {
        measure_names: [
          'initial_command',
          'max_net_command',
          'sum_net_command',
          'initial_force',
          'max_net_force',
          'sum_net_force',
          'max_parallel_vel_forward',
          'max_lateral_vel_left',
          'max_lateral_vel_right',
          'max_lateral_distance_left',
          'sum_lateral_distance',
          'end_position_error',
          'max_parallel_force_forward',
          'sum_parallel_force',
          'max_lateral_force_right',
          'sum_lateral_force_abs',
        ],
      },
      role: 'dependency',
    },

    // Visualization: Violins for measure distributions
    [nodeId(P, 'violins_measures')]: {
      id: nodeId(P, 'violins_measures'),
      type: 'Violins',
      label: 'Violins (measures)',
      category: 'Visualization',
      inputPorts: ['input', 'input_split'],
      outputPorts: ['figure'],
      params: {
        violinmode: 'overlay',
        zero_hline: false,
        map_figs_at_level: 'measure',
      },
      role: 'analysis',
    },

    // Visualization: Aligned trajectories colored by perturbation amplitude (aligned)
    [nodeId(P, 'aligned_traj_by_amp')]: {
      id: nodeId(P, 'aligned_traj_by_amp'),
      type: 'EffectorTrajectories',
      label: 'Aligned Trajectories (by amp)',
      category: 'Visualization',
      inputPorts: ['data'],
      outputPorts: ['figure'],
      params: {
        pos_endpoints: true,
        straight_guides: true,
        colorscale_key: 'pert__amp',
        align_epoch: 2,
        task_variant: 'small',
      },
      role: 'analysis',
    },

    // Visualization: Aligned trajectories, no alignment
    [nodeId(P, 'aligned_traj_noalign')]: {
      id: nodeId(P, 'aligned_traj_noalign'),
      type: 'EffectorTrajectories',
      label: 'Aligned Trajectories (no align)',
      category: 'Visualization',
      inputPorts: ['data'],
      outputPorts: ['figure'],
      params: {
        pos_endpoints: true,
        straight_guides: true,
        colorscale_key: 'pert__amp',
        align_epoch: null,
        task_variant: 'small',
      },
      role: 'analysis',
    },

    // Visualization: Profiles (time-series with PCA)
    [nodeId(P, 'profiles')]: {
      id: nodeId(P, 'profiles'),
      type: 'Profiles',
      label: 'Profiles',
      category: 'Visualization',
      inputPorts: ['vars'],
      outputPorts: ['figure'],
      params: {
        varset: ['pos', 'vel', 'command', 'force'],
        mode: 'std',
        n_std_plot: 1,
        layout_height: 300,
        layout_width: 450,
        mvt_epoch_idx: 2,
      },
      role: 'analysis',
    },
  };

  let wireIdx = 0;
  const wires: AnalysisWire[] = [
    // DataSource → GetBestReplicate (states)
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'get_best_replicate'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'states',
    },

    // GetBestReplicate → AlignedVars
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'get_best_replicate'),
      sourcePort: 'data',
      targetId: nodeId(P, 'aligned_vars'),
      targetPort: 'data',
      implicit: false,
    },

    // AlignedVars → ApplyFns(measures)
    // With transform: discard varset, keep only "full" aligned vars
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'aligned_vars'),
      sourcePort: 'aligned_vars',
      targetId: nodeId(P, 'apply_fns_measures'),
      targetPort: 'input',
      implicit: false,
      transform: {
        id: transformId(P, 'discard_varset'),
        type: 'GetitemAtLevel',
        label: 'Keep "full" varset',
        params: { level: 'varset', key: 'full' },
      },
    },

    // ApplyFns(measures) → Violins
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'apply_fns_measures'),
      sourcePort: 'measures',
      targetId: nodeId(P, 'violins_measures'),
      targetPort: 'input',
      implicit: false,
    },

    // GetBestReplicate → Aligned Trajectories (aligned)
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'get_best_replicate'),
      sourcePort: 'data',
      targetId: nodeId(P, 'aligned_traj_by_amp'),
      targetPort: 'data',
      implicit: false,
    },

    // GetBestReplicate → Aligned Trajectories (no align)
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'get_best_replicate'),
      sourcePort: 'data',
      targetId: nodeId(P, 'aligned_traj_noalign'),
      targetPort: 'data',
      implicit: false,
    },

    // DataSource (states.net.hidden) → Profiles
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'profiles'),
      targetPort: 'vars',
      implicit: true,
      fieldPath: 'states.net.hidden',
    },
  ];

  const evalParams: EvalParametrization = {
    perturbation_type: 'curl_field',
    perturbation_amplitudes: [0.0, 0.5, 1.0, 2.0, 4.0],
    task_variants: { small: 'small' },
  };

  return {
    id: 'rlrmp-p1-plant-perts',
    name: 'plant_perts',
    graphSpec: {
      nodes,
      wires,
      dataSourceId: DATA_SOURCE_ID,
    },
    evalParams,
    viewport: { x: 0, y: 0, zoom: 1 },
    evalRunId: null,
  };
}

// ---------------------------------------------------------------------------
// Page 2: feedback_perts
//
// Source: rlrmp/modules/analysis/part1/feedback_perts.py
//
// Pipeline:
//   DataSource → AlignedVars (with impulse directions) → ApplyFns(measures) → Violins
//
// Eval: Feedback impulses, varying amplitudes and directions
//   - Perturbation variables: fb_pos, fb_vel
//   - Directions: rand or xy
//   - Amplitudes: linspace(0, amp_max, n_amps)
// ---------------------------------------------------------------------------

function buildFeedbackPertsPage(): AnalysisPageSpec {
  const P = 'feedback_perts';

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Computation: AlignedVars with impulse direction alignment
    [nodeId(P, 'aligned_vars')]: {
      id: nodeId(P, 'aligned_vars'),
      type: 'AlignedVars',
      label: 'AlignedVars (impulse dirs)',
      category: 'Computation',
      inputPorts: ['data'],
      outputPorts: ['aligned_vars'],
      params: {
        varset: ['pos', 'vel', 'command', 'force'],
        directions_fn: 'impulse_directions',
        align_epoch: null,
      },
      role: 'dependency',
    },

    // Computation: ApplyFns for feedback perturbation measures
    [nodeId(P, 'apply_fns_measures')]: {
      id: nodeId(P, 'apply_fns_measures'),
      type: 'ApplyFns',
      label: 'ApplyFns (measures)',
      category: 'Computation',
      inputPorts: ['input'],
      outputPorts: ['measures'],
      params: {
        measure_names: [
          'max_net_force',
          'max_parallel_force_reverse',
          'sum_net_force',
          'max_parallel_vel_forward',
          'max_parallel_vel_reverse',
          'max_lateral_vel_left',
          'max_lateral_vel_right',
          'max_deviation',
          'sum_deviation',
        ],
      },
      role: 'dependency',
    },

    // Visualization: Violins for measures
    [nodeId(P, 'violins_measures')]: {
      id: nodeId(P, 'violins_measures'),
      type: 'Violins',
      label: 'Violins (measures)',
      category: 'Visualization',
      inputPorts: ['input', 'input_split'],
      outputPorts: ['figure'],
      params: {
        violinmode: 'overlay',
        zero_hline: false,
        map_figs_at_level: 'measure',
      },
      role: 'analysis',
    },
  };

  let wireIdx = 0;
  const wires: AnalysisWire[] = [
    // DataSource → AlignedVars (states)
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'aligned_vars'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'states',
    },

    // AlignedVars → ApplyFns(measures)
    // With transform: discard varset, keep only "full"
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'aligned_vars'),
      sourcePort: 'aligned_vars',
      targetId: nodeId(P, 'apply_fns_measures'),
      targetPort: 'input',
      implicit: false,
      transform: {
        id: transformId(P, 'discard_varset'),
        type: 'GetitemAtLevel',
        label: 'Keep "full" varset',
        params: { level: 'varset', key: 'full' },
      },
    },

    // ApplyFns(measures) → Violins
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'apply_fns_measures'),
      sourcePort: 'measures',
      targetId: nodeId(P, 'violins_measures'),
      targetPort: 'input',
      implicit: false,
    },
  ];

  const evalParams: EvalParametrization = {
    perturbation_type: 'feedback_impulse',
    perturbation_amplitudes: [0.5, 1.0, 2.0, 4.0],
    perturbation_variables: ['fb_pos', 'fb_vel'],
    perturbation_direction: 'rand',
    perturbation_duration: 5,
    perturbation_start_step: 30,
    perturbation_n_amps: 4,
  };

  return {
    id: 'rlrmp-p1-feedback-perts',
    name: 'feedback_perts',
    graphSpec: {
      nodes,
      wires,
      dataSourceId: DATA_SOURCE_ID,
    },
    evalParams,
    viewport: { x: 0, y: 0, zoom: 1 },
    evalRunId: null,
  };
}

// ---------------------------------------------------------------------------
// Page 3: freq_response
//
// Source: rlrmp/modules/analysis/part1/freq_response.py
//
// Pipeline:
//   DataSource → FrequencyResponse
//
// Eval: Standard task — Gaussian noise in feedback channels suffices.
//   INPUT_WHERE = state.feedback.noise[idx]
//   OUTPUT_WHERE = state.net.output
// ---------------------------------------------------------------------------

function buildFreqResponsePage(): AnalysisPageSpec {
  const P = 'freq_response';

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Dynamics: FrequencyResponse — FFT-based transfer function analysis
    [nodeId(P, 'frequency_response')]: {
      id: nodeId(P, 'frequency_response'),
      type: 'FrequencyResponse',
      label: 'FrequencyResponse',
      category: 'Dynamics',
      inputPorts: ['data'],
      outputPorts: ['freq_response'],
      params: {
        input_field: 'states.feedback.noise',
        output_field: 'states.net.output',
        variant: 'full',
        fb_var_names: ['fb_pos', 'fb_vel'],
      },
      role: 'analysis',
    },
  };

  let wireIdx = 0;
  const wires: AnalysisWire[] = [
    // DataSource (states.feedback.noise) → FrequencyResponse
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'frequency_response'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'states',
    },
  ];

  const evalParams: EvalParametrization = {
    // Standard evaluation — no special perturbation needed.
    // The existing Gaussian noise in feedback channels is the input signal.
    perturbation_type: '',
  };

  return {
    id: 'rlrmp-p1-freq-response',
    name: 'freq_response',
    graphSpec: {
      nodes,
      wires,
      dataSourceId: DATA_SOURCE_ID,
    },
    evalParams,
    viewport: { x: 0, y: 0, zoom: 1 },
    evalRunId: null,
  };
}

// ---------------------------------------------------------------------------
// Page 4: unit_prefs
//
// Source: rlrmp/modules/analysis/part1/unit_prefs.py
//
// Pipeline:
//   DataSource → GetBestReplicate → SegmentEpochs → UnitPreferences (x2)
//
// Two UnitPreferences instances:
//   1. Control forces: feature_fn extracts states.efferent.output
//   2. Goal positions: feature_fn extracts task.validation_trials.targets
//
// Eval: Plant perturbations (same as plant_perts page)
// ---------------------------------------------------------------------------

function buildUnitPrefsPage(): AnalysisPageSpec {
  const P = 'unit_prefs';

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Preprocessing: GetBestReplicate
    [nodeId(P, 'get_best_replicate')]: {
      id: nodeId(P, 'get_best_replicate'),
      type: 'GetBestReplicate',
      label: 'GetBestReplicate',
      category: 'Preprocessing',
      inputPorts: ['data'],
      outputPorts: ['data'],
      params: { axis: 0 },
      role: 'dependency',
    },

    // Preprocessing: SegmentEpochs — split into accel/decel phases
    [nodeId(P, 'segment_epochs')]: {
      id: nodeId(P, 'segment_epochs'),
      type: 'SegmentEpochs',
      label: 'SegmentEpochs (accel/decel)',
      category: 'Preprocessing',
      inputPorts: ['data'],
      outputPorts: ['data'],
      params: {
        epoch_fn: 'symmetric_accel_decel',
        epoch_boundaries: [],
      },
      role: 'dependency',
    },

    // Dynamics: UnitPreferences — control forces
    [nodeId(P, 'unit_prefs_forces')]: {
      id: nodeId(P, 'unit_prefs_forces'),
      type: 'UnitPreferences',
      label: 'UnitPreferences (control forces)',
      category: 'Dynamics',
      inputPorts: ['data'],
      outputPorts: ['unit_prefs'],
      params: {
        feature_fn: 'control_forces',
        feature_field: 'states.efferent.output',
      },
      role: 'analysis',
    },

    // Dynamics: UnitPreferences — goal positions
    [nodeId(P, 'unit_prefs_goals')]: {
      id: nodeId(P, 'unit_prefs_goals'),
      type: 'UnitPreferences',
      label: 'UnitPreferences (goal positions)',
      category: 'Dynamics',
      inputPorts: ['data'],
      outputPorts: ['unit_prefs'],
      params: {
        feature_fn: 'goal_positions',
        feature_field: 'task.validation_trials.targets',
      },
      role: 'analysis',
    },
  };

  let wireIdx = 0;
  const wires: AnalysisWire[] = [
    // DataSource → GetBestReplicate (states)
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'get_best_replicate'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'states',
    },

    // GetBestReplicate → SegmentEpochs
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'get_best_replicate'),
      sourcePort: 'data',
      targetId: nodeId(P, 'segment_epochs'),
      targetPort: 'data',
      implicit: false,
    },

    // SegmentEpochs → UnitPreferences (control forces)
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'segment_epochs'),
      sourcePort: 'data',
      targetId: nodeId(P, 'unit_prefs_forces'),
      targetPort: 'data',
      implicit: false,
    },

    // SegmentEpochs → UnitPreferences (goal positions)
    {
      id: wireId(P, wireIdx++),
      sourceId: nodeId(P, 'segment_epochs'),
      sourcePort: 'data',
      targetId: nodeId(P, 'unit_prefs_goals'),
      targetPort: 'data',
      implicit: false,
    },

    // DataSource → UnitPreferences (goal positions): task targets
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'task',
      targetId: nodeId(P, 'unit_prefs_goals'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'task.validation_trials.targets',
    },

    // DataSource → UnitPreferences (control forces): efferent output
    {
      id: wireId(P, wireIdx++),
      sourceId: DATA_SOURCE_ID,
      sourcePort: 'states',
      targetId: nodeId(P, 'unit_prefs_forces'),
      targetPort: 'data',
      implicit: true,
      fieldPath: 'states.efferent.output',
    },
  ];

  const evalParams: EvalParametrization = {
    perturbation_type: 'curl_field',
    perturbation_amplitudes: [0.0, 0.5, 1.0, 2.0, 4.0],
  };

  return {
    id: 'rlrmp-p1-unit-prefs',
    name: 'unit_prefs',
    graphSpec: {
      nodes,
      wires,
      dataSourceId: DATA_SOURCE_ID,
    },
    evalParams,
    viewport: { x: 0, y: 0, zoom: 1 },
    evalRunId: null,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create the RLRMP Part 1 project analysis snapshot.
 *
 * Returns a complete AnalysisSnapshot with 4 pages ready to load
 * into the analysis store. The first page (plant_perts) is set as
 * the active page.
 */
export function createRlrmpPart1Analysis(): AnalysisSnapshot {
  const pages = [
    buildPlantPertsPage(),
    buildFeedbackPertsPage(),
    buildFreqResponsePage(),
    buildUnitPrefsPage(),
  ];

  return {
    pages,
    activePageId: pages[0].id,
  };
}

/**
 * Metadata for the template project.
 */
export const RLRMP_PART1_TEMPLATE = {
  id: 'template-rlrmp-part1',
  name: 'RLRMP: Part 1',
  description: 'Plant perturbations, feedback perturbations, frequency response, and unit preferences analysis.',
  pageNames: ['plant_perts', 'feedback_perts', 'freq_response', 'unit_prefs'],
} as const;
