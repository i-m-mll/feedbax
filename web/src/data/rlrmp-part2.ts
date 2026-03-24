/**
 * Project template for "rlrmp: Part 2" — all 7 analysis modules from the
 * RLRMP Part 2 analysis pipeline, faithfully reproduced as Feedbax Studio
 * analysis pages with full node/wire/transform specification.
 *
 * Each page corresponds to a Python module in
 *   rlrmp.modules.analysis.part2.*
 *
 * Modules:
 *   1. plant_perts    — Plant perturbations with SISU variation, PCA, aligned vars
 *   2. feedback_perts — Feedback impulse perturbations with alignment + custom measures
 *   3. fps_steady     — Fixed points at steady state: PCA -> FP -> Jacobians -> Eig/SVD
 *   4. fps_reach      — Fixed points during reaching trajectories
 *   5. sisu_pert      — SISU step perturbation: effector + network activity + profiles
 *   6. tangling       — Tangling analysis with PCA projection
 *   7. unit_perts     — Unit stimulation: triple-vmap, profiles, response distributions
 */

import type {
  AnalysisNodeSpec,
  AnalysisWire,
  AnalysisGraphSpec,
  AnalysisPageSpec,
  AnalysisSnapshot,
  EvalParametrization,
  AnalysisViewport,
  TransformSpec,
} from '@/types/analysis';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DATA_SOURCE_ID = '__data_source__';
const DEFAULT_VIEWPORT: AnalysisViewport = { x: 0, y: 0, zoom: 1 };

let _nodeCounter = 0;
let _wireCounter = 0;

function nodeId(prefix: string): string {
  return `${prefix}_${++_nodeCounter}`;
}

function wireId(): string {
  return `p2_wire_${++_wireCounter}`;
}

function resetCounters(): void {
  _nodeCounter = 0;
  _wireCounter = 0;
}

function makeNode(
  id: string,
  type: string,
  label: string,
  category: string,
  opts: {
    inputPorts?: string[];
    outputPorts?: string[];
    params?: Record<string, unknown>;
    role?: 'analysis' | 'dependency';
  } = {},
): AnalysisNodeSpec {
  return {
    id,
    type,
    label,
    category,
    inputPorts: opts.inputPorts ?? ['input'],
    outputPorts: opts.outputPorts ?? ['output'],
    params: (opts.params ?? {}) as Record<string, number | string | boolean | null>,
    role: opts.role ?? 'analysis',
  };
}

function makeWire(
  sourceId: string,
  sourcePort: string,
  targetId: string,
  targetPort: string,
  opts: {
    implicit?: boolean;
    fieldPath?: string;
    transform?: TransformSpec;
  } = {},
): AnalysisWire {
  return {
    id: wireId(),
    sourceId,
    sourcePort,
    targetId,
    targetPort,
    implicit: opts.implicit ?? false,
    fieldPath: opts.fieldPath,
    transform: opts.transform,
  };
}

function makePage(
  id: string,
  name: string,
  nodes: Record<string, AnalysisNodeSpec>,
  wires: AnalysisWire[],
  evalParams: EvalParametrization = {},
): AnalysisPageSpec {
  return {
    id,
    name,
    graphSpec: { nodes, wires, dataSourceId: DATA_SOURCE_ID },
    evalParams,
    viewport: { ...DEFAULT_VIEWPORT },
    evalRunId: null,
  };
}

// ---------------------------------------------------------------------------
// Page 1: plant_perts
// ---------------------------------------------------------------------------

function buildPlantPertsPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const hiddenPcaId = nodeId('dep_states_pca');
  const measuresDepId = nodeId('dep_measures');
  const alignedVarsId = nodeId('dep_aligned_vars');

  // --- Analyses ---
  const alignedTrajBySisuId = nodeId('aligned_traj_by_sisu');
  const alignedTrajByStdId = nodeId('aligned_traj_by_std');
  const profilesBySisuId = nodeId('profiles_by_sisu');
  const measureViolinsId = nodeId('measures_by_pert_amp');

  // --- Draft (commented-out in Python) ---
  const tanglingDraftId = nodeId('tangling_draft');
  const jacobiansDraftId = nodeId('jacobians_draft');
  const tanglingViolinsByStdId = nodeId('tangling_violins_by_std');
  const tanglingViolinsByAmpId = nodeId('tangling_violins_by_amp');
  const profilesByStdId = nodeId('profiles_by_std_draft');
  const measuresByStdId = nodeId('measures_by_std_draft');
  const measuresStdByAmpId = nodeId('measures_std_by_amp_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Dependencies
    [hiddenPcaId]: makeNode(hiddenPcaId, 'StatesPCA', 'Hidden States PCA', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['pca_results'],
      params: {
        n_components: 10,
        where_states: 'states.net.hidden',
        aggregate_over_labels: ['pert__amp', 'sisu'],
        start_step: 0,
        end_step: 100,
      },
      role: 'dependency',
    }),
    [alignedVarsId]: makeNode(alignedVarsId, 'AlignedVars', 'AlignedVars', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['aligned_vars', 'measures_input'],
      params: {
        task_variant: 'full',
      },
      role: 'dependency',
    }),
    [measuresDepId]: makeNode(measuresDepId, 'ApplyFns', 'Measures', 'Preprocessing', {
      inputPorts: ['input'],
      outputPorts: ['measures'],
      params: {
        measure_keys: [
          'initial_command', 'max_net_command', 'sum_net_command',
          'initial_force', 'max_net_force', 'sum_net_force',
          'max_parallel_vel_forward', 'largest_lateral_distance',
          'sum_lateral_distance_abs', 'end_position_error',
          'max_parallel_force_forward', 'sum_lateral_force_abs',
        ],
        getitem_level: 'task_variant',
        getitem_key: 'full',
      },
      role: 'dependency',
    }),

    // Active analyses
    [alignedTrajBySisuId]: makeNode(alignedTrajBySisuId, 'AlignedTrajectories', 'Aligned Trajectories (by SISU)', 'Trajectory Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        colorscale_key: 'sisu',
        getitem_level: 'task_variant',
        getitem_key: 'small',
        map_figs_level: 'train__pert__std',
      },
    }),
    [alignedTrajByStdId]: makeNode(alignedTrajByStdId, 'AlignedTrajectories', 'Aligned Trajectories (by train std)', 'Trajectory Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        colorscale_key: 'train__pert__std',
        getitem_level: 'task_variant',
        getitem_key: 'small',
        map_figs_level: 'sisu',
      },
    }),
    [profilesBySisuId]: makeNode(profilesBySisuId, 'Profiles', 'Profiles (by SISU)', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        varset: 'default',
        level_to_bottom: 'sisu',
        transform: 'get_best_replicate',
        set_axis_bounds_equal_y: true,
        equal_axes_levels: ['var', 'coord'],
        invert_levels: true,
      },
    }),
    [measureViolinsId]: makeNode(measureViolinsId, 'Violins', 'Measures (by pert amp)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        group_var: 'sisu',
        x_var: 'train__pert__std',
        rearrange_levels: ['...', 'sisu', 'train__pert__std'],
        map_figs_level: 'measure',
        set_axis_bounds_equal_y: true,
      },
    }),

    // Draft / commented-out analyses (non-vestigial — kept for future use)
    [tanglingDraftId]: makeNode(tanglingDraftId, 'Tangling', 'Tangling (draft)', 'Dynamics', {
      inputPorts: ['state', 'pca_results'],
      outputPorts: ['tangling'],
      params: {
        variant: 'small',
        where_state: 'states.net.hidden',
        transform: 'get_best_replicate',
        pca_projection: true,
        _draft: true,
      },
    }),
    [jacobiansDraftId]: makeNode(jacobiansDraftId, 'Jacobians', 'Jacobians (draft)', 'Dynamics', {
      inputPorts: ['fns', 'fn_args'],
      outputPorts: ['jacobians'],
      params: {
        timestep_stride: 5,
        sisu_subset: [-3, 0, 1, 3],
        train_std_subset: [0, 1.5],
        _draft: true,
      },
    }),
    [tanglingViolinsByStdId]: makeNode(tanglingViolinsByStdId, 'Violins', 'Tangling Violins (by std, draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        yaxis_title: 'RMS tangling',
        violinmode: 'group',
        rearrange_levels: ['...', 'sisu', 'pert__amp'],
        rms_agg_t_slice_start: 1,
        _draft: true,
      },
    }),
    [tanglingViolinsByAmpId]: makeNode(tanglingViolinsByAmpId, 'Violins', 'Tangling Violins (by amp, draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        yaxis_title: 'RMS tangling',
        violinmode: 'group',
        rearrange_levels: ['...', 'sisu', 'train__pert__std'],
        rms_agg_t_slice_start: 1,
        _draft: true,
      },
    }),
    [profilesByStdId]: makeNode(profilesByStdId, 'Profiles', 'Profiles (by train std, draft)', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        varset: 'default',
        level_to_bottom: 'train__pert__std',
        transform: 'get_best_replicate',
        _draft: true,
      },
    }),
    [measuresByStdId]: makeNode(measuresByStdId, 'Violins', 'Measures (by train std, draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        group_var: 'sisu',
        x_var: 'pert__amp',
        _draft: true,
      },
    }),
    [measuresStdByAmpId]: makeNode(measuresStdByAmpId, 'Violins', 'Measures (std by amp, draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        group_var: 'train__pert__std',
        x_var: 'sisu',
        _draft: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> Dependencies
    makeWire(DATA_SOURCE_ID, 'states', hiddenPcaId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    makeWire(DATA_SOURCE_ID, 'states', alignedVarsId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),

    // AlignedVars -> Measures
    makeWire(alignedVarsId, 'aligned_vars', measuresDepId, 'input'),

    // DataSource -> Aligned Trajectories
    makeWire(DATA_SOURCE_ID, 'states', alignedTrajBySisuId, 'input', {
      implicit: true,
      fieldPath: 'states',
    }),
    makeWire(DATA_SOURCE_ID, 'states', alignedTrajByStdId, 'input', {
      implicit: true,
      fieldPath: 'states',
    }),

    // DataSource -> Profiles
    makeWire(DATA_SOURCE_ID, 'states', profilesBySisuId, 'vars', {
      implicit: true,
      fieldPath: 'states',
    }),

    // Measures -> Violins
    makeWire(measuresDepId, 'measures', measureViolinsId, 'input'),

    // Draft wiring
    makeWire(DATA_SOURCE_ID, 'states', tanglingDraftId, 'state', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    makeWire(hiddenPcaId, 'pca_results', tanglingDraftId, 'pca_results'),

    // Tangling -> Violins (draft)
    makeWire(tanglingDraftId, 'tangling', tanglingViolinsByStdId, 'input'),
    makeWire(tanglingDraftId, 'tangling', tanglingViolinsByAmpId, 'input'),

    // DataSource -> Jacobians (draft)
    makeWire(DATA_SOURCE_ID, 'states', jacobiansDraftId, 'fn_args', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // DataSource -> Profiles (draft)
    makeWire(DATA_SOURCE_ID, 'states', profilesByStdId, 'vars', {
      implicit: true,
      fieldPath: 'states',
    }),

    // Measures -> draft violins
    makeWire(measuresDepId, 'measures', measuresByStdId, 'input'),
    makeWire(measuresDepId, 'measures', measuresStdByAmpId, 'input'),
  ];

  return makePage('p2_plant_perts', 'Plant Perturbations', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    pert_type: 'curl_field',
    pert_amp: [0, 1],
    pca_n_components: 10,
    pca_start_step: 0,
    pca_end_step: 100,
  });
}

// ---------------------------------------------------------------------------
// Page 2: feedback_perts
// ---------------------------------------------------------------------------

function buildFeedbackPertsPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const alignedVarsId = nodeId('dep_aligned_vars');
  const measuresId = nodeId('dep_measures');

  // --- Analyses ---
  const profilesId = nodeId('profiles');
  const measureViolinsId = nodeId('measure_violins');

  // --- Draft ---
  const effectorTrajId = nodeId('effector_traj_draft');
  const alignedTrajId = nodeId('aligned_traj_draft');
  const alignedTrajByStdId = nodeId('aligned_traj_by_std_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    [alignedVarsId]: makeNode(alignedVarsId, 'AlignedVars', 'AlignedVars (impulse directions)', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['aligned_vars', 'measures_input'],
      params: {
        directions_fn: 'get_impulse_directions',
        varset: 'default',
      },
      role: 'dependency',
    }),
    [measuresId]: makeNode(measuresId, 'ApplyFns', 'Measures', 'Preprocessing', {
      inputPorts: ['input'],
      outputPorts: ['measures'],
      params: {
        measure_keys: [
          'max_net_command', 'max_net_force', 'max_parallel_force_reverse',
          'sum_net_force', 'max_parallel_vel_forward', 'max_parallel_vel_reverse',
          'max_lateral_vel_left', 'max_lateral_vel_right',
          'max_deviation', 'sum_deviation',
        ],
        custom_measures: {
          early_command_mean: {
            response_var: 'command',
            timesteps: [1, 3],
            agg_fn: 'mean',
            transform_fn: 'norm',
          },
          early_force_mean: {
            response_var: 'force',
            timesteps: [1, 3],
            agg_fn: 'mean',
            transform_fn: 'norm',
          },
        },
        getitem_level: 'task_variant',
        getitem_key: 'full',
      },
      role: 'dependency',
    }),
    [profilesId]: makeNode(profilesId, 'Profiles', 'Profiles', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        varset: 'default',
        vrect_kws_fn: 'get_impulse_vrect_kws',
        transform: 'get_best_replicate',
        level_to_bottom: 'sisu',
        index_axis_label: 'pert__amp',
        index_value: -2,
        layout_width: 500,
        layout_height: 300,
        set_axis_bounds_equal_y: true,
        equal_axes_levels: ['var'],
        invert_levels: true,
      },
    }),
    [measureViolinsId]: makeNode(measureViolinsId, 'Violins', 'Measures', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        legend_title: 'SISU',
        xaxis_title: 'Feedback impulse amplitude',
        violinmode: 'group',
        transform: 'get_best_replicate',
        unstack_axis: 1,
        unstack_label: 'pert__amp',
        lohi_level: 'train__pert__std',
        map_figs_levels: ['measure', 'train__pert__std', 'pert__var'],
        set_axis_bounds_equal_y: true,
        equal_axes_levels: ['measure', 'pert__var'],
        invert_levels: true,
      },
    }),

    // Draft (commented-out)
    [effectorTrajId]: makeNode(effectorTrajId, 'EffectorTrajectories', 'Effector Trajectories (draft)', 'Trajectory Plots', {
      inputPorts: ['states'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        colorscale_axis: 1,
        colorscale_key: 'pert__amp',
        transform: 'get_best_replicate',
        _draft: true,
      },
    }),
    [alignedTrajId]: makeNode(alignedTrajId, 'AlignedTrajectories', 'Aligned Trajectories (draft)', 'Trajectory Plots', {
      inputPorts: ['aligned_vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        colorscale_axis: 1,
        colorscale_key: 'pert__amp',
        transform: 'get_best_replicate',
        _draft: true,
      },
    }),
    [alignedTrajByStdId]: makeNode(alignedTrajByStdId, 'AlignedTrajectories', 'Aligned Traj by Std (draft)', 'Trajectory Plots', {
      inputPorts: ['aligned_vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        colorscale_key: 'train__pert__std',
        varset: 'default',
        transform: 'get_best_replicate',
        stacking_level: 'train__pert__std',
        _draft: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> AlignedVars
    makeWire(DATA_SOURCE_ID, 'states', alignedVarsId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),

    // AlignedVars -> Measures
    makeWire(alignedVarsId, 'aligned_vars', measuresId, 'input'),

    // AlignedVars -> Profiles
    makeWire(alignedVarsId, 'aligned_vars', profilesId, 'vars'),

    // Measures -> Violins
    makeWire(measuresId, 'measures', measureViolinsId, 'input'),

    // Draft wiring
    makeWire(DATA_SOURCE_ID, 'states', effectorTrajId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),
    makeWire(alignedVarsId, 'aligned_vars', alignedTrajId, 'aligned_vars'),
    makeWire(alignedVarsId, 'aligned_vars', alignedTrajByStdId, 'aligned_vars'),
  ];

  return makePage('p2_feedback_perts', 'Feedback Perturbations', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    pert_start_step: 30,
    pert_duration: 5,
    pert_n_amps: 8,
    pert_amp_max_fb_pos: 0.5,
    pert_amp_max_fb_vel: 0.1,
    pert_vars: ['fb_pos', 'fb_vel'],
  });
}

// ---------------------------------------------------------------------------
// Page 3: fps_steady  (most complex DAG)
// ---------------------------------------------------------------------------

function buildFpsSteadyPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const statesPcaId = nodeId('dep_states_pca');
  const fpResultsId = nodeId('dep_fp_results');

  // --- Analyses ---
  const fpsInPcId = nodeId('fps_pc');
  const scatterPlotsId = nodeId('scatter_plots');
  const jacobiansId = nodeId('jacobians');
  const jacXEigId = nodeId('jac_x_eigs');
  const jacUSvdId = nodeId('jac_u_svd');
  const eigvalsPlotId = nodeId('eigvals_plot');
  const jacXEigViolinsId = nodeId('jac_x_eigval_violins');
  const jacUSingvalViolinsId = nodeId('jac_u_singval_violins');

  // Draft
  const hessiansId = nodeId('hessians_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Dependencies
    [statesPcaId]: makeNode(statesPcaId, 'StatesPCA', 'States PCA', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['pca_results'],
      params: {
        n_components: 50,
        where_states: 'states.net.hidden',
        aggregate_over_labels: ['sisu'],
        start_step: 0,
        end_step: 100,
      },
      role: 'dependency',
    }),
    [fpResultsId]: makeNode(fpResultsId, 'FixedPoints', 'Steady-State Fixed Points', 'Dynamics', {
      inputPorts: ['fns', 'fn_args', 'candidates'],
      outputPorts: ['fp_results', 'fps'],
      params: {
        stride_candidates: 16,
        fn_type: 'ss_rnn_fn',
        fn_args_type: 'sisu_pos',
        vmap_over: 'positions',
        post_process: 'process_fps',
        n_keep: 6,
      },
      role: 'dependency',
    }),

    // Analysis DAG
    [fpsInPcId]: makeNode(fpsInPcId, 'PlotInPCSpace', 'FPs in PC Space', 'Visualization', {
      inputPorts: ['pca_results', 'plot_data'],
      outputPorts: ['figures'],
      params: {
        spread_label: 'sisu',
        transform_plot_data: 'extract_fps',
      },
    }),
    [scatterPlotsId]: makeNode(scatterPlotsId, 'ScatterPlots', 'FP Scatter Plots', 'Visualization', {
      inputPorts: ['pca_results', 'plot_data'],
      outputPorts: ['figures'],
      params: {
        spread_label: 'sisu',
        transform_plot_data: 'extract_fps',
      },
    }),
    [jacobiansId]: makeNode(jacobiansId, 'Jacobians', 'Steady-State Jacobians', 'Dynamics', {
      inputPorts: ['fns', 'fn_args'],
      outputPorts: ['jacobians'],
      params: {
        fn_type: 'ss_rnn_fn',
        fn_args_type: 'sisu_pos_h',
        fn_args_sources: ['sisu', 'positions', 'steady_state_fps'],
        vmap_over: 'positions',
      },
    }),
    [jacXEigId]: makeNode(jacXEigId, 'Eig', 'Jac-x Eigendecomposition', 'Dynamics', {
      inputPorts: ['matrices'],
      outputPorts: ['eigvals', 'eigvecs'],
      params: {
        source_transform: 'jacobians.h',
      },
    }),
    [jacUSvdId]: makeNode(jacUSvdId, 'SVD', 'Jac-u SVD', 'Dynamics', {
      inputPorts: ['matrices'],
      outputPorts: ['singvals', 'U', 'V'],
      params: {
        source_transform: 'jacobians.pos',
      },
    }),
    [eigvalsPlotId]: makeNode(eigvalsPlotId, 'EigvalsPlot', 'Jac-x Eigenvalues Plot', 'Visualization', {
      inputPorts: ['eigvals'],
      outputPorts: ['figures'],
      params: {
        hide_histograms: false,
        level_to_bottom: 'sisu',
        marginals: 'box',
      },
    }),
    [jacXEigViolinsId]: makeNode(jacXEigViolinsId, 'Violins', 'Jac-x Eigenvalue Violins', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        input_transform: 'complex_to_polar_abs_angle',
        rearrange_levels: ['...', 'component', 'sisu', 'train__pert__std'],
        map_figs_level: 'component',
        component_params: {
          angle: { yaxis_title: 'Eigenvalue angle (rad)', yaxis_range: [0, 3.14159] },
          magnitude: { yaxis_title: 'Eigenvalue magnitude', yaxis_range: [0, 1.1] },
        },
      },
    }),
    [jacUSingvalViolinsId]: makeNode(jacUSingvalViolinsId, 'Violins', 'Jac-u Singular Value Violins', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        rearrange_levels: ['...', 'sisu', 'train__pert__std'],
      },
    }),

    // Draft (Hessians commented out)
    [hessiansId]: makeNode(hessiansId, 'Hessians', 'Hessians (draft)', 'Dynamics', {
      inputPorts: ['fns', 'fn_args'],
      outputPorts: ['hessians'],
      params: {
        fn_type: 'ss_rnn_fn',
        fn_args_type: 'sisu_pos_h',
        vmap_over: 'positions',
        _draft: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> StatesPCA
    makeWire(DATA_SOURCE_ID, 'states', statesPcaId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // DataSource -> FixedPoints (candidates = hidden states)
    makeWire(DATA_SOURCE_ID, 'states', fpResultsId, 'candidates', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    // DataSource -> FixedPoints (fns = model RNN cell)
    makeWire(DATA_SOURCE_ID, 'model', fpResultsId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),

    // StatesPCA -> PlotInPCSpace
    makeWire(statesPcaId, 'pca_results', fpsInPcId, 'pca_results'),
    // FP results -> PlotInPCSpace
    makeWire(fpResultsId, 'fps', fpsInPcId, 'plot_data'),

    // StatesPCA -> ScatterPlots
    makeWire(statesPcaId, 'pca_results', scatterPlotsId, 'pca_results'),
    // FP results -> ScatterPlots
    makeWire(fpResultsId, 'fps', scatterPlotsId, 'plot_data'),

    // DataSource -> Jacobians (fns)
    makeWire(DATA_SOURCE_ID, 'model', jacobiansId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),
    // FP results -> Jacobians (fn_args: sisu, positions, steady state fps)
    makeWire(fpResultsId, 'fps', jacobiansId, 'fn_args'),

    // Jacobians -> Eig (state Jacobians)
    makeWire(jacobiansId, 'jacobians', jacXEigId, 'matrices'),

    // Jacobians -> SVD (input Jacobians)
    makeWire(jacobiansId, 'jacobians', jacUSvdId, 'matrices'),

    // Eig -> EigvalsPlot
    makeWire(jacXEigId, 'eigvals', eigvalsPlotId, 'eigvals'),

    // Eig -> Jac-x Violins
    makeWire(jacXEigId, 'eigvals', jacXEigViolinsId, 'input'),

    // SVD -> Jac-u Violins
    makeWire(jacUSvdId, 'singvals', jacUSingvalViolinsId, 'input'),

    // Draft wiring
    makeWire(DATA_SOURCE_ID, 'model', hessiansId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),
    makeWire(fpResultsId, 'fps', hessiansId, 'fn_args'),
  ];

  return makePage('p2_fps_steady', 'Fixed Points (Steady State)', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    n_pca: 50,
    pca_start_step: 0,
    pca_end_step: 100,
    stride_fp_candidates: 16,
    fp_n_keep: 6,
  });
}

// ---------------------------------------------------------------------------
// Page 4: fps_reach
// ---------------------------------------------------------------------------

function buildFpsReachPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const statesPcaId = nodeId('dep_states_pca');

  // --- Analyses ---
  const reachFpId = nodeId('reach_fp_results');

  // Draft nodes (all commented out in Python)
  const fpsPcDraftId = nodeId('fps_pc_draft');
  const hiddenFpTrajDraftId = nodeId('hidden_fp_traj_draft');
  const compareSisuDraftId = nodeId('compare_sisu_draft');
  const tanglingDraftId = nodeId('tangling_draft');
  const jacsDraftId = nodeId('jacs_draft');
  const jacsEigDraftId = nodeId('jacs_eig_draft');
  const jacsSvdDraftId = nodeId('jacs_svd_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    [statesPcaId]: makeNode(statesPcaId, 'StatesPCA', 'States PCA', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['pca_results'],
      params: {
        n_components: 30,
        where_states: 'states.net.hidden',
        transform: 'get_best_replicate',
        start_step: 50,
        end_step: 100,
      },
      role: 'dependency',
    }),
    [reachFpId]: makeNode(reachFpId, 'FixedPoints', 'Reach Fixed Points', 'Dynamics', {
      inputPorts: ['fns', 'fn_args', 'candidates'],
      outputPorts: ['fp_results'],
      params: {
        fn_type: 'call_rnn_fn',
        fn_args_sources: ['rnn_fns', 'rnn_states'],
        candidates_source: 'rnn_states',
        candidates_transform: 'prepare_candidates_simple',
        candidate_timestep_radius: 1,
        vmap_in_axes: {
          fn_args: ['None_0_None_None', '0_1_2_3'],
          candidates: 'None_1_2_None',
        },
      },
    }),

    // Draft analysis placeholders from commented-out Python
    [fpsPcDraftId]: makeNode(fpsPcDraftId, 'PlotInPCSpace', 'FPs in PC Space (draft)', 'Visualization', {
      inputPorts: ['pca_results', 'plot_data'],
      outputPorts: ['figures'],
      params: { _draft: true },
    }),
    [hiddenFpTrajDraftId]: makeNode(hiddenFpTrajDraftId, 'ScatterPlots', 'Hidden+FP Trajectories PC (draft)', 'Visualization', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: { _draft: true },
    }),
    [compareSisuDraftId]: makeNode(compareSisuDraftId, 'ScatterPlots', 'Compare SISU Trajectories PC (draft)', 'Visualization', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: { _draft: true },
    }),
    [tanglingDraftId]: makeNode(tanglingDraftId, 'Tangling', 'Tangling (draft)', 'Dynamics', {
      inputPorts: ['state'],
      outputPorts: ['tangling'],
      params: { _draft: true },
    }),
    [jacsDraftId]: makeNode(jacsDraftId, 'Jacobians', 'Jacobians (draft)', 'Dynamics', {
      inputPorts: ['fns', 'fn_args'],
      outputPorts: ['jacobians'],
      params: { _draft: true },
    }),
    [jacsEigDraftId]: makeNode(jacsEigDraftId, 'Eig', 'Jacs Eig (draft)', 'Dynamics', {
      inputPorts: ['matrices'],
      outputPorts: ['eigvals'],
      params: { _draft: true },
    }),
    [jacsSvdDraftId]: makeNode(jacsSvdDraftId, 'SVD', 'Jacs SVD (draft)', 'Dynamics', {
      inputPorts: ['matrices'],
      outputPorts: ['singvals'],
      params: { _draft: true },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> StatesPCA
    makeWire(DATA_SOURCE_ID, 'states', statesPcaId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // DataSource -> FixedPoints (fn_args: rnn cell + hidden states)
    makeWire(DATA_SOURCE_ID, 'model', reachFpId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),
    makeWire(DATA_SOURCE_ID, 'states', reachFpId, 'fn_args', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    makeWire(DATA_SOURCE_ID, 'states', reachFpId, 'candidates', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // Draft wiring (placeholder connections)
    makeWire(statesPcaId, 'pca_results', fpsPcDraftId, 'pca_results'),
    makeWire(reachFpId, 'fp_results', fpsPcDraftId, 'plot_data'),
    makeWire(DATA_SOURCE_ID, 'states', tanglingDraftId, 'state', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    makeWire(DATA_SOURCE_ID, 'model', jacsDraftId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),
    makeWire(jacsDraftId, 'jacobians', jacsEigDraftId, 'matrices'),
    makeWire(jacsDraftId, 'jacobians', jacsSvdDraftId, 'matrices'),
  ];

  return makePage('p2_fps_reach', 'Fixed Points (Reach)', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    pert_type: 'curl_field',
    pert_amp: [0, 1],
    n_pca: 30,
    pca_start_step: 50,
    pca_end_step: 100,
    candidate_timestep_radius: 1,
  });
}

// ---------------------------------------------------------------------------
// Page 5: sisu_pert
// ---------------------------------------------------------------------------

function buildSisuPertPage(): AnalysisPageSpec {
  // --- Analyses ---
  const effectorSteadyId = nodeId('effector_traj_steady');
  const networkActSteadyId = nodeId('network_activity_steady');
  const alignedTrajReachId = nodeId('aligned_traj_reach');
  const profilesReachId = nodeId('profiles_reach');

  // Draft
  const profilesSteadyDraftId = nodeId('profiles_steady_draft');
  const networkActPcaDraftId = nodeId('network_act_pca_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    [effectorSteadyId]: makeNode(effectorSteadyId, 'EffectorTrajectories', 'Effector Trajectories (steady)', 'Trajectory Plots', {
      inputPorts: ['states'],
      outputPorts: ['figures'],
      params: {
        variant: 'steady',
        pos_endpoints: false,
        straight_guides: false,
        colorscale_axis: 1,
        colorscale_key: 'reach_condition',
        transform: 'get_best_replicate',
        mean_exclude_axes: [-3],
        legend_title: 'SISU\nbr>pert. amp.',
      },
    }),
    [networkActSteadyId]: makeNode(networkActSteadyId, 'NetworkActivity_SampleUnits', 'Network Activity (steady)', 'Activity Plots', {
      inputPorts: ['states'],
      outputPorts: ['figures'],
      params: {
        variant: 'steady',
        transform: 'get_best_replicate',
        level_to_top: 'train__pert__std',
        legend_title: 'SISU pert. amp.',
      },
    }),
    [alignedTrajReachId]: makeNode(alignedTrajReachId, 'AlignedTrajectories', 'Aligned Trajectories (reach)', 'Trajectory Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        colorscale_key: 'pert__sisu__amp',
        getitem_level: 'task_variant',
        getitem_key: 'reach',
        legend_title: 'Final SISU',
        hide_individual_trials: true,
        combine_figs_by_axis: 3,
        plant_pert_labels: { 0: 'no curl', 1: 'curl' },
        plant_pert_line_dash: { 0: 'dot', 1: 'solid' },
        layout_width: 900,
        layout_height: 300,
      },
    }),
    [profilesReachId]: makeNode(profilesReachId, 'Profiles', 'Profiles (reach)', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'reach',
        level_to_top: 'train__pert__std',
        combine_figs_by_axis: 2,
        plant_pert_labels: { 0: 'no curl', 1: 'curl' },
        plant_pert_line_dash: { 0: 'dot', 1: 'solid' },
      },
    }),

    // Draft
    [profilesSteadyDraftId]: makeNode(profilesSteadyDraftId, 'Profiles', 'Profiles (steady, draft)', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'steady',
        level_to_top: 'train__pert__std',
        _draft: true,
      },
    }),
    [networkActPcaDraftId]: makeNode(networkActPcaDraftId, 'NetworkActivity_ProjectPCA', 'Network Activity PCA (draft)', 'Activity Plots', {
      inputPorts: ['states'],
      outputPorts: ['figures'],
      params: {
        variant: 'steady',
        variant_pca: 'reach_pca',
        _draft: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> EffectorTrajectories
    makeWire(DATA_SOURCE_ID, 'states', effectorSteadyId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),

    // DataSource -> NetworkActivity
    makeWire(DATA_SOURCE_ID, 'states', networkActSteadyId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // DataSource -> AlignedTrajectories
    makeWire(DATA_SOURCE_ID, 'states', alignedTrajReachId, 'input', {
      implicit: true,
      fieldPath: 'states',
    }),

    // DataSource -> Profiles (reach)
    makeWire(DATA_SOURCE_ID, 'states', profilesReachId, 'vars', {
      implicit: true,
      fieldPath: 'states',
    }),

    // Draft wiring
    makeWire(DATA_SOURCE_ID, 'states', profilesSteadyDraftId, 'vars', {
      implicit: true,
      fieldPath: 'states',
    }),
    makeWire(DATA_SOURCE_ID, 'states', networkActPcaDraftId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
  ];

  return makePage('p2_sisu_pert', 'SISU Perturbation', nodes, wires, {
    sisu_init: 0,
    sisu_final: [-3, -2, -1, 0, 1, 2, 3],
    sisu_step: 30,
    plant_pert_type: 'curl_field',
    plant_pert_amp: [0, 1],
  });
}

// ---------------------------------------------------------------------------
// Page 6: tangling
// ---------------------------------------------------------------------------

function buildTanglingPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const hiddenPcaId = nodeId('dep_hidden_pca');

  // --- Analyses ---
  const tanglingId = nodeId('tangling');

  const nodes: Record<string, AnalysisNodeSpec> = {
    [hiddenPcaId]: makeNode(hiddenPcaId, 'StatesPCA', 'Hidden States PCA', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['pca_results'],
      params: {
        n_components: 10,
        where_states: 'states.net.hidden',
        aggregate_over_labels: ['pert__amp', 'sisu'],
        transform: 'get_best_replicate',
      },
      role: 'dependency',
    }),
    [tanglingId]: makeNode(tanglingId, 'Tangling', 'Tangling', 'Dynamics', {
      inputPorts: ['state', 'pca_results'],
      outputPorts: ['tangling'],
      params: {
        variant: 'small',
        where_state: 'states.net.hidden',
        transform: 'get_best_replicate',
        pca_projection: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> StatesPCA
    makeWire(DATA_SOURCE_ID, 'states', hiddenPcaId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // DataSource -> Tangling (hidden state input)
    makeWire(DATA_SOURCE_ID, 'states', tanglingId, 'state', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // StatesPCA -> Tangling (PCA results for projection)
    makeWire(hiddenPcaId, 'pca_results', tanglingId, 'pca_results'),
  ];

  return makePage('p2_tangling', 'Tangling', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    pert_type: 'curl_field',
    pert_amp: [0, 1],
    n_pca: 10,
  });
}

// ---------------------------------------------------------------------------
// Page 7: unit_perts  (most complex)
// ---------------------------------------------------------------------------

function buildUnitPertsPage(): AnalysisPageSpec {
  // --- Dependencies ---
  const alignedVarsTrivialId = nodeId('dep_aligned_vars_trivial');
  const hiddenPcaId = nodeId('dep_hidden_pca');
  const responseVarsId = nodeId('dep_response_vars');

  // --- Analyses ---
  const profilesId = nodeId('unit_stim_profiles');
  const stimResponseDistsId = nodeId('stim_response_dists');

  // Draft / commented-out
  const jacUSsDraftId = nodeId('jac_u_ss_draft');
  const regressionDraftId = nodeId('regression_draft');
  const jacRegressionPrepDraftId = nodeId('jac_regression_prep_draft');
  const jacRegressionDraftId = nodeId('jac_regression_draft');
  const fbGainsDraftId = nodeId('fb_gains_draft');
  const regressionFigsDraftId = nodeId('regression_figs_draft');
  const hiddenPcPlotDraftId = nodeId('hidden_pc_plot_draft');
  const jacViolinsDraftId = nodeId('jac_violins_draft');
  const fbGainViolinsDraftId = nodeId('fb_gain_violins_draft');
  const alignedEffTrajDraftId = nodeId('aligned_eff_traj_draft');
  const unitPrefsDraftId = nodeId('unit_prefs_draft');

  const nodes: Record<string, AnalysisNodeSpec> = {
    // Dependencies
    [alignedVarsTrivialId]: makeNode(alignedVarsTrivialId, 'AlignedVars', 'AlignedVars (trivial directions)', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['aligned_vars'],
      params: {
        varset: ['pos', 'vel'],
        directions_fn: 'get_trivial_reach_directions',
      },
      role: 'dependency',
    }),
    [hiddenPcaId]: makeNode(hiddenPcaId, 'StatesPCA', 'Hidden States PCA', 'Preprocessing', {
      inputPorts: ['states'],
      outputPorts: ['pca_results'],
      params: {
        n_components: 10,
        where_states: 'states.net.hidden',
        aggregate_over_labels: ['pert__amp', 'sisu'],
        replicate_axis: 3,
        getitem_level: 'task_variant',
        getitem_key: 'full',
        start_step: 0,
        end_step: 60,
      },
      role: 'dependency',
    }),
    [responseVarsId]: makeNode(responseVarsId, 'IdentityNode', 'Response Variables', 'Preprocessing', {
      inputPorts: ['input'],
      outputPorts: ['output'],
      params: {
        replicate_axis: 3,
        getitem_level: 'task_variant',
        getitem_key: 'full',
        index_axis: 0,
        index_value: 1,
        index_label: 'stim_amp',
        transform: 'get_response_vars',
        rearrange_levels: ['response_var', 'train__pert__std', 'sisu', 'pert__amp'],
      },
      role: 'dependency',
    }),

    // Active analyses
    [profilesId]: makeNode(profilesId, 'Profiles', 'Unit Stim Profiles', 'Trajectory Plots', {
      inputPorts: ['vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        vrect_kws_fn: 'get_impulse_vrect_kws',
        coord_labels: null,
        agg_mode: { deviation: 'standard', angle: 'circular', speed: 'standard' },
        replicate_axis: 3,
        index_unit_stim: [2, 83, 95, 97, 43, 48],
        transform_vars: 'transform_profile_vars',
        unstack_axes: [
          { axis: 1, label: 'unit_stim_idx', above_level: 'pert__amp' },
          { axis: 0, label: 'stim_amp', above_level: 'pert__amp' },
        ],
        rearrange_levels: ['...', 'train__pert__std', 'var', 'pert__amp'],
        sisu_subset: [-2, 0, 2],
        combine_figs_level: 'sisu',
        sisu_labels: { 0: -2, 1: 0, 2: 2 },
        sisu_line_dash: { 0: 'dot', 1: 'dash', 2: 'solid' },
        layout_width: 500,
        layout_height: 350,
      },
    }),
    [stimResponseDistsId]: makeNode(stimResponseDistsId, 'Violins', 'Stim Response Distributions', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        violinmode: 'group',
        rearrange_levels: ['...', 'sisu', 'train__pert__std'],
        map_figs_level: 'response_var',
      },
    }),

    // Draft analyses (from commented-out Python code)
    [jacUSsDraftId]: makeNode(jacUSsDraftId, 'Jacobians', 'Jac-u Steady State (draft)', 'Dynamics', {
      inputPorts: ['fns', 'fn_args'],
      outputPorts: ['jacobians'],
      params: {
        argnums: 0,
        getitem_level: 'task_variant',
        getitem_key: 'full',
        _draft: true,
      },
    }),
    [regressionDraftId]: makeNode(regressionDraftId, 'Regression', 'Unit Stim Regression (draft)', 'Statistics', {
      inputPorts: ['regressor_tree'],
      outputPorts: ['regression_results'],
      params: {
        map_compute_level: 'sisu',
        vmap_over_stim_units: true,
        _draft: true,
      },
    }),
    [jacRegressionPrepDraftId]: makeNode(jacRegressionPrepDraftId, 'IdentityNode', 'Jac Regression Prep (draft)', 'Preprocessing', {
      inputPorts: ['input'],
      outputPorts: ['output'],
      params: {
        transform: 'get_best_replicate',
        rearrange_levels: ['train__pert__std', '...'],
        _draft: true,
      },
    }),
    [jacRegressionDraftId]: makeNode(jacRegressionDraftId, 'Regression', 'Jac-u Regression (draft)', 'Statistics', {
      inputPorts: ['regressor_tree'],
      outputPorts: ['regression_results'],
      params: {
        map_compute_level: 'sisu',
        vmap_over_stim_units: true,
        _draft: true,
      },
    }),
    [fbGainsDraftId]: makeNode(fbGainsDraftId, 'InstantFBResponse', 'FB Gains (draft)', 'Dynamics', {
      inputPorts: ['states'],
      outputPorts: ['gains'],
      params: {
        n_directions: 24,
        fb_pert_amp: 0.5,
        _draft: true,
      },
    }),
    [regressionFigsDraftId]: makeNode(regressionFigsDraftId, 'UnitStimRegressionFigures', 'Regression Figures (draft)', 'Visualization', {
      inputPorts: ['regression_results', 'unit_fb_gains'],
      outputPorts: ['figures'],
      params: { _draft: true },
    }),
    [hiddenPcPlotDraftId]: makeNode(hiddenPcPlotDraftId, 'ScatterPlots', 'Hidden PC Plot (draft)', 'Visualization', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        subplot_level: 'train__pert__std',
        colorscale_key: 'sisu',
        _draft: true,
      },
    }),
    [jacViolinsDraftId]: makeNode(jacViolinsDraftId, 'Violins', 'Jac-u Violins (draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: { _draft: true },
    }),
    [fbGainViolinsDraftId]: makeNode(fbGainViolinsDraftId, 'Violins', 'FB Gain Violins (draft)', 'Statistical Plots', {
      inputPorts: ['input'],
      outputPorts: ['figures'],
      params: {
        transform: 'get_best_replicate',
        rearrange_levels: ['...', 'sisu', 'train__pert__std'],
        _draft: true,
      },
    }),
    [alignedEffTrajDraftId]: makeNode(alignedEffTrajDraftId, 'AlignedTrajectories', 'Aligned Eff. Traj (draft)', 'Trajectory Plots', {
      inputPorts: ['aligned_vars'],
      outputPorts: ['figures'],
      params: {
        variant: 'full',
        replicate_axis: 3,
        _draft: true,
      },
    }),
    [unitPrefsDraftId]: makeNode(unitPrefsDraftId, 'UnitPreferences', 'Unit Preferences (draft)', 'Activity Plots', {
      inputPorts: ['states'],
      outputPorts: ['preferences'],
      params: {
        variant: 'full',
        feature_fn: 'efferent.output',
        replicate_axis: 3,
        segment_epochs: ['pre', 'peri', 'post'],
        _draft: true,
      },
    }),
  };

  const wires: AnalysisWire[] = [
    // DataSource -> AlignedVars (trivial)
    makeWire(DATA_SOURCE_ID, 'states', alignedVarsTrivialId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),

    // DataSource -> Hidden PCA
    makeWire(DATA_SOURCE_ID, 'states', hiddenPcaId, 'states', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),

    // AlignedVars -> ResponseVars
    makeWire(alignedVarsTrivialId, 'aligned_vars', responseVarsId, 'input'),

    // AlignedVars -> Profiles
    makeWire(alignedVarsTrivialId, 'aligned_vars', profilesId, 'vars'),

    // ResponseVars -> Stim Response Violins
    makeWire(responseVarsId, 'output', stimResponseDistsId, 'input'),

    // Draft wiring
    makeWire(DATA_SOURCE_ID, 'model', jacUSsDraftId, 'fns', {
      implicit: true,
      fieldPath: 'model.step.net.hidden',
    }),
    makeWire(DATA_SOURCE_ID, 'states', jacUSsDraftId, 'fn_args', {
      implicit: true,
      fieldPath: 'states.net.input',
    }),
    makeWire(responseVarsId, 'output', regressionDraftId, 'regressor_tree'),
    makeWire(jacUSsDraftId, 'jacobians', jacRegressionPrepDraftId, 'input'),
    makeWire(jacRegressionPrepDraftId, 'output', jacRegressionDraftId, 'regressor_tree'),
    makeWire(DATA_SOURCE_ID, 'states', fbGainsDraftId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),
    makeWire(regressionDraftId, 'regression_results', regressionFigsDraftId, 'regression_results'),
    makeWire(jacRegressionDraftId, 'regression_results', regressionFigsDraftId, 'unit_fb_gains'),
    makeWire(DATA_SOURCE_ID, 'states', hiddenPcPlotDraftId, 'input', {
      implicit: true,
      fieldPath: 'states.net.hidden',
    }),
    makeWire(jacUSsDraftId, 'jacobians', jacViolinsDraftId, 'input'),
    makeWire(fbGainsDraftId, 'gains', fbGainViolinsDraftId, 'input'),
    makeWire(alignedVarsTrivialId, 'aligned_vars', alignedEffTrajDraftId, 'aligned_vars'),
    makeWire(DATA_SOURCE_ID, 'states', unitPrefsDraftId, 'states', {
      implicit: true,
      fieldPath: 'states',
    }),
  ];

  return makePage('p2_unit_perts', 'Unit Perturbations', nodes, wires, {
    sisu: [-3, -2, -1, 0, 1, 2, 3],
    pert_plant_type: 'curl_field',
    pert_plant_amp: [0, 1],
    pert_unit_start_step: 20,
    pert_unit_duration: 10,
    pert_unit_amp: [0, 0.5],
    hidden_size: 100,
    n_pca: 10,
    pca_start_step: 0,
    pca_end_step: 60,
    unit_idxs_profiles: [2, 83, 95, 97, 43, 48],
    scale_by_readout_length: false,
  });
}

// ---------------------------------------------------------------------------
// Public API: create the full project template
// ---------------------------------------------------------------------------

/**
 * Create the "rlrmp: Part 2" project template with all 7 analysis pages.
 *
 * Returns an `AnalysisSnapshot` ready for `restoreSnapshot()`.
 */
export function createRlrmpPart2Project(): AnalysisSnapshot {
  resetCounters();

  const pages: AnalysisPageSpec[] = [
    buildPlantPertsPage(),
    buildFeedbackPertsPage(),
    buildFpsSteadyPage(),
    buildFpsReachPage(),
    buildSisuPertPage(),
    buildTanglingPage(),
    buildUnitPertsPage(),
  ];

  return {
    pages,
    activePageId: pages[0].id,
  };
}

/**
 * Metadata for the project template, used by the UI for project selection.
 */
export const RLRMP_PART2_META = {
  id: 'rlrmp-part2',
  name: 'rlrmp: Part 2',
  description: 'Complete Part 2 analysis pipeline: plant perturbations, feedback perturbations, fixed points (steady state & reach), SISU perturbation, tangling, and unit perturbations.',
  pageCount: 7,
  pageNames: [
    'Plant Perturbations',
    'Feedback Perturbations',
    'Fixed Points (Steady State)',
    'Fixed Points (Reach)',
    'SISU Perturbation',
    'Tangling',
    'Unit Perturbations',
  ],
} as const;
