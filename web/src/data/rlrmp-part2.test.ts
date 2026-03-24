/**
 * Interaction tests for the rlrmp: Part 2 project template.
 *
 * Verifies that the project loads with all 7 pages, each page has the
 * correct nodes and wiring, and that all structural invariants hold.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createRlrmpPart2Project, RLRMP_PART2_META } from './rlrmp-part2';
import type { AnalysisSnapshot, AnalysisPageSpec, AnalysisNodeSpec, AnalysisWire } from '@/types/analysis';

let project: AnalysisSnapshot;

beforeEach(() => {
  project = createRlrmpPart2Project();
});

// ---------------------------------------------------------------------------
// Top-level project structure
// ---------------------------------------------------------------------------

describe('createRlrmpPart2Project', () => {
  it('returns a valid AnalysisSnapshot', () => {
    expect(project).toBeDefined();
    expect(project.pages).toBeDefined();
    expect(project.activePageId).toBeDefined();
  });

  it('creates exactly 7 pages', () => {
    expect(project.pages).toHaveLength(7);
  });

  it('sets the first page as active', () => {
    expect(project.activePageId).toBe(project.pages[0].id);
  });

  it('has unique page IDs', () => {
    const ids = project.pages.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('pages have expected names in order', () => {
    const names = project.pages.map((p) => p.name);
    expect(names).toEqual([
      'Plant Perturbations',
      'Feedback Perturbations',
      'Fixed Points (Steady State)',
      'Fixed Points (Reach)',
      'SISU Perturbation',
      'Tangling',
      'Unit Perturbations',
    ]);
  });

  it('metadata matches the project', () => {
    expect(RLRMP_PART2_META.pageCount).toBe(7);
    expect(RLRMP_PART2_META.pageNames).toHaveLength(7);
    expect(RLRMP_PART2_META.id).toBe('rlrmp-part2');
  });
});

// ---------------------------------------------------------------------------
// Per-page structural invariants
// ---------------------------------------------------------------------------

describe('page structural invariants', () => {
  it('every page has a dataSourceId', () => {
    for (const page of project.pages) {
      expect(page.graphSpec.dataSourceId).toBe('__data_source__');
    }
  });

  it('every page has at least one node', () => {
    for (const page of project.pages) {
      expect(Object.keys(page.graphSpec.nodes).length).toBeGreaterThan(0);
    }
  });

  it('every page has at least one wire', () => {
    for (const page of project.pages) {
      expect(page.graphSpec.wires.length).toBeGreaterThan(0);
    }
  });

  it('every wire references valid node IDs or data source', () => {
    for (const page of project.pages) {
      const nodeIds = new Set(Object.keys(page.graphSpec.nodes));
      nodeIds.add(page.graphSpec.dataSourceId);

      for (const wire of page.graphSpec.wires) {
        expect(nodeIds.has(wire.sourceId)).toBe(true);
        expect(nodeIds.has(wire.targetId)).toBe(true);
      }
    }
  });

  it('every wire has unique ID within its page', () => {
    for (const page of project.pages) {
      const wireIds = page.graphSpec.wires.map((w) => w.id);
      expect(new Set(wireIds).size).toBe(wireIds.length);
    }
  });

  it('every node ID matches its key in the nodes record', () => {
    for (const page of project.pages) {
      for (const [key, node] of Object.entries(page.graphSpec.nodes)) {
        expect(node.id).toBe(key);
      }
    }
  });

  it('every page has evalParams', () => {
    for (const page of project.pages) {
      expect(page.evalParams).toBeDefined();
      expect(typeof page.evalParams).toBe('object');
    }
  });

  it('every page has a viewport', () => {
    for (const page of project.pages) {
      expect(page.viewport).toBeDefined();
      expect(typeof page.viewport.x).toBe('number');
      expect(typeof page.viewport.y).toBe('number');
      expect(typeof page.viewport.zoom).toBe('number');
    }
  });
});

// ---------------------------------------------------------------------------
// Page 1: Plant Perturbations
// ---------------------------------------------------------------------------

describe('Page 1: Plant Perturbations', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[0];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Plant Perturbations');
  });

  it('contains StatesPCA dependency', () => {
    const pcaNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'StatesPCA',
    );
    expect(pcaNodes.length).toBeGreaterThanOrEqual(1);
    expect(pcaNodes[0].role).toBe('dependency');
  });

  it('contains AlignedVars dependency', () => {
    const avNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'AlignedVars',
    );
    expect(avNodes.length).toBe(1);
    expect(avNodes[0].role).toBe('dependency');
  });

  it('contains ApplyFns (measures) dependency', () => {
    const measNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'ApplyFns',
    );
    expect(measNodes.length).toBe(1);
    expect(measNodes[0].role).toBe('dependency');
  });

  it('contains active aligned trajectory analyses', () => {
    const trajNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'AlignedTrajectories' && n.role === 'analysis',
    );
    expect(trajNodes.length).toBe(2); // by_sisu and by_train_std
  });

  it('contains Profiles analysis', () => {
    const profNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Profiles' && !n.params._draft,
    );
    expect(profNodes.length).toBe(1);
  });

  it('contains Violins (measures) analysis', () => {
    const violinNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Violins' && !n.params._draft,
    );
    expect(violinNodes.length).toBe(1);
  });

  it('contains draft tangling and jacobians nodes', () => {
    const draftNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.params._draft === true,
    );
    expect(draftNodes.length).toBeGreaterThanOrEqual(4);
  });

  it('has SISU eval params', () => {
    expect(page.evalParams.sisu).toBeDefined();
    expect(Array.isArray(page.evalParams.sisu)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Page 2: Feedback Perturbations
// ---------------------------------------------------------------------------

describe('Page 2: Feedback Perturbations', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[1];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Feedback Perturbations');
  });

  it('contains AlignedVars with impulse directions', () => {
    const avNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'AlignedVars',
    );
    expect(avNodes.length).toBe(1);
    expect(avNodes[0].params.directions_fn).toBe('get_impulse_directions');
  });

  it('contains custom measures (early_command_mean, early_force_mean)', () => {
    const measNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'ApplyFns',
    );
    expect(measNodes.length).toBe(1);
    const customMeasures = measNodes[0].params.custom_measures as Record<string, unknown>;
    expect(customMeasures).toBeDefined();
    expect(customMeasures.early_command_mean).toBeDefined();
    expect(customMeasures.early_force_mean).toBeDefined();
  });

  it('contains Profiles and Violins analyses', () => {
    const profileNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Profiles' && !n.params._draft,
    );
    const violinNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Violins' && !n.params._draft,
    );
    expect(profileNodes.length).toBe(1);
    expect(violinNodes.length).toBe(1);
  });

  it('has perturbation variable eval params', () => {
    expect(page.evalParams.pert_vars).toEqual(['fb_pos', 'fb_vel']);
  });
});

// ---------------------------------------------------------------------------
// Page 3: Fixed Points (Steady State) — most complex DAG
// ---------------------------------------------------------------------------

describe('Page 3: Fixed Points (Steady State)', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[2];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Fixed Points (Steady State)');
  });

  it('contains StatesPCA dependency', () => {
    const pcaNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'StatesPCA',
    );
    expect(pcaNodes.length).toBe(1);
    expect(pcaNodes[0].params.n_components).toBe(50);
  });

  it('contains FixedPoints dependency', () => {
    const fpNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'FixedPoints',
    );
    expect(fpNodes.length).toBe(1);
    expect(fpNodes[0].params.stride_candidates).toBe(16);
  });

  it('contains PlotInPCSpace analysis', () => {
    const plotNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'PlotInPCSpace',
    );
    expect(plotNodes.length).toBe(1);
  });

  it('contains Jacobians analysis', () => {
    const jacNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Jacobians' && !n.params._draft,
    );
    expect(jacNodes.length).toBe(1);
  });

  it('contains Eig analysis for state Jacobians', () => {
    const eigNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Eig',
    );
    expect(eigNodes.length).toBe(1);
  });

  it('contains SVD analysis for input Jacobians', () => {
    const svdNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'SVD',
    );
    expect(svdNodes.length).toBe(1);
  });

  it('contains EigvalsPlot', () => {
    const eigPlotNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'EigvalsPlot',
    );
    expect(eigPlotNodes.length).toBe(1);
  });

  it('contains violin plots for eigenvalues and singular values', () => {
    const violinNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Violins',
    );
    expect(violinNodes.length).toBe(2); // jac-x eigvals + jac-u singvals
  });

  it('has correct DAG wiring depth: DataSource -> PCA -> FPs -> Jacobians -> Eig/SVD -> Plots', () => {
    const nodes = page.graphSpec.nodes;
    const wires = page.graphSpec.wires;

    // Find specific node types
    const pcaNode = Object.values(nodes).find((n) => n.type === 'StatesPCA')!;
    const fpNode = Object.values(nodes).find((n) => n.type === 'FixedPoints')!;
    const jacNode = Object.values(nodes).find((n) => n.type === 'Jacobians' && !n.params._draft)!;
    const eigNode = Object.values(nodes).find((n) => n.type === 'Eig')!;
    const svdNode = Object.values(nodes).find((n) => n.type === 'SVD')!;
    const eigPlotNode = Object.values(nodes).find((n) => n.type === 'EigvalsPlot')!;

    // Verify key connections exist
    const hasWire = (src: string, tgt: string) =>
      wires.some((w) => w.sourceId === src && w.targetId === tgt);

    // PCA -> PlotInPCSpace
    const plotPcNode = Object.values(nodes).find((n) => n.type === 'PlotInPCSpace')!;
    expect(hasWire(pcaNode.id, plotPcNode.id)).toBe(true);

    // FP -> Jacobians
    expect(hasWire(fpNode.id, jacNode.id)).toBe(true);

    // Jacobians -> Eig
    expect(hasWire(jacNode.id, eigNode.id)).toBe(true);

    // Jacobians -> SVD
    expect(hasWire(jacNode.id, svdNode.id)).toBe(true);

    // Eig -> EigvalsPlot
    expect(hasWire(eigNode.id, eigPlotNode.id)).toBe(true);
  });

  it('contains draft Hessians node', () => {
    const draftNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Hessians' && n.params._draft === true,
    );
    expect(draftNodes.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Page 4: Fixed Points (Reach)
// ---------------------------------------------------------------------------

describe('Page 4: Fixed Points (Reach)', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[3];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Fixed Points (Reach)');
  });

  it('contains StatesPCA dependency', () => {
    const pcaNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'StatesPCA',
    );
    expect(pcaNodes.length).toBe(1);
    expect(pcaNodes[0].params.n_components).toBe(30);
    expect(pcaNodes[0].params.start_step).toBe(50);
  });

  it('contains FixedPoints analysis for reach trajectories', () => {
    const fpNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'FixedPoints',
    );
    expect(fpNodes.length).toBe(1);
  });

  it('contains draft visualization and dynamics nodes', () => {
    const draftNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.params._draft === true,
    );
    expect(draftNodes.length).toBeGreaterThanOrEqual(5);
  });
});

// ---------------------------------------------------------------------------
// Page 5: SISU Perturbation
// ---------------------------------------------------------------------------

describe('Page 5: SISU Perturbation', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[4];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('SISU Perturbation');
  });

  it('contains EffectorTrajectories (steady)', () => {
    const effNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'EffectorTrajectories',
    );
    expect(effNodes.length).toBe(1);
    expect(effNodes[0].params.variant).toBe('steady');
  });

  it('contains NetworkActivity_SampleUnits', () => {
    const actNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'NetworkActivity_SampleUnits',
    );
    expect(actNodes.length).toBe(1);
  });

  it('contains AlignedTrajectories for reach', () => {
    const trajNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'AlignedTrajectories',
    );
    expect(trajNodes.length).toBe(1);
    expect(trajNodes[0].params.colorscale_key).toBe('pert__sisu__amp');
  });

  it('contains Profiles for reach', () => {
    const profNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Profiles' && !n.params._draft,
    );
    expect(profNodes.length).toBe(1);
    expect(profNodes[0].params.variant).toBe('reach');
  });

  it('has SISU perturbation eval params', () => {
    expect(page.evalParams.sisu_init).toBeDefined();
    expect(page.evalParams.sisu_final).toBeDefined();
    expect(page.evalParams.sisu_step).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Page 6: Tangling
// ---------------------------------------------------------------------------

describe('Page 6: Tangling', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[5];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Tangling');
  });

  it('contains StatesPCA dependency', () => {
    const pcaNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'StatesPCA',
    );
    expect(pcaNodes.length).toBe(1);
    expect(pcaNodes[0].role).toBe('dependency');
  });

  it('contains Tangling analysis with PCA projection', () => {
    const tangNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Tangling',
    );
    expect(tangNodes.length).toBe(1);
    expect(tangNodes[0].params.pca_projection).toBe(true);
  });

  it('has PCA -> Tangling wiring', () => {
    const nodes = page.graphSpec.nodes;
    const wires = page.graphSpec.wires;

    const pcaNode = Object.values(nodes).find((n) => n.type === 'StatesPCA')!;
    const tangNode = Object.values(nodes).find((n) => n.type === 'Tangling')!;

    const hasWire = wires.some(
      (w) => w.sourceId === pcaNode.id && w.targetId === tangNode.id,
    );
    expect(hasWire).toBe(true);
  });

  it('has minimal node count (2 nodes: PCA + Tangling)', () => {
    expect(Object.keys(page.graphSpec.nodes)).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Page 7: Unit Perturbations — most complex
// ---------------------------------------------------------------------------

describe('Page 7: Unit Perturbations', () => {
  let page: AnalysisPageSpec;

  beforeEach(() => {
    page = project.pages[6];
  });

  it('has the correct name', () => {
    expect(page.name).toBe('Unit Perturbations');
  });

  it('contains AlignedVars with trivial directions', () => {
    const avNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'AlignedVars',
    );
    expect(avNodes.length).toBe(1);
    expect(avNodes[0].params.directions_fn).toBe('get_trivial_reach_directions');
  });

  it('contains StatesPCA dependency', () => {
    const pcaNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'StatesPCA',
    );
    expect(pcaNodes.length).toBe(1);
    expect(pcaNodes[0].params.n_components).toBe(10);
    expect(pcaNodes[0].params.end_step).toBe(60);
  });

  it('contains IdentityNode for response variables', () => {
    const idNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'IdentityNode',
    );
    // At least the response vars one (non-draft)
    expect(idNodes.length).toBeGreaterThanOrEqual(1);
  });

  it('contains Profiles analysis with unit stim config', () => {
    const profNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Profiles' && !n.params._draft,
    );
    expect(profNodes.length).toBe(1);
    expect(profNodes[0].params.vrect_kws_fn).toBe('get_impulse_vrect_kws');
  });

  it('contains Violins for stim response distributions', () => {
    const violinNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.type === 'Violins' && !n.params._draft,
    );
    expect(violinNodes.length).toBe(1);
    expect(violinNodes[0].params.violinmode).toBe('group');
  });

  it('has unit perturbation eval params', () => {
    expect(page.evalParams.pert_unit_start_step).toBe(20);
    expect(page.evalParams.pert_unit_duration).toBe(10);
    expect(page.evalParams.hidden_size).toBe(100);
    expect(page.evalParams.unit_idxs_profiles).toEqual([2, 83, 95, 97, 43, 48]);
  });

  it('has many draft nodes (regression, jacobians, etc.)', () => {
    const draftNodes = Object.values(page.graphSpec.nodes).filter(
      (n) => n.params._draft === true,
    );
    // Regression, JacRegPrep, JacRegression, FBGains, RegressionFigs,
    // HiddenPCPlot, JacViolins, FBGainViolins, AlignedEffTraj, UnitPrefs, Jac-u-ss
    expect(draftNodes.length).toBeGreaterThanOrEqual(8);
  });

  it('has the most nodes of any page', () => {
    const unitPertsNodeCount = Object.keys(page.graphSpec.nodes).length;
    for (const otherPage of project.pages.slice(0, 6)) {
      expect(unitPertsNodeCount).toBeGreaterThanOrEqual(
        Object.keys(otherPage.graphSpec.nodes).length,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// Cross-page consistency
// ---------------------------------------------------------------------------

describe('cross-page consistency', () => {
  it('no two pages share node IDs', () => {
    const allNodeIds = new Set<string>();
    for (const page of project.pages) {
      for (const id of Object.keys(page.graphSpec.nodes)) {
        expect(allNodeIds.has(id)).toBe(false);
        allNodeIds.add(id);
      }
    }
  });

  it('no two pages share wire IDs', () => {
    const allWireIds = new Set<string>();
    for (const page of project.pages) {
      for (const wire of page.graphSpec.wires) {
        expect(allWireIds.has(wire.id)).toBe(false);
        allWireIds.add(wire.id);
      }
    }
  });

  it('implicit wires only originate from the data source', () => {
    for (const page of project.pages) {
      for (const wire of page.graphSpec.wires) {
        if (wire.implicit) {
          expect(wire.sourceId).toBe('__data_source__');
        }
      }
    }
  });

  it('wires with fieldPath are always from data source', () => {
    for (const page of project.pages) {
      for (const wire of page.graphSpec.wires) {
        if (wire.fieldPath) {
          expect(wire.sourceId).toBe('__data_source__');
        }
      }
    }
  });

  it('every node has a non-empty label', () => {
    for (const page of project.pages) {
      for (const node of Object.values(page.graphSpec.nodes)) {
        expect(node.label.length).toBeGreaterThan(0);
      }
    }
  });

  it('every node has a non-empty type', () => {
    for (const page of project.pages) {
      for (const node of Object.values(page.graphSpec.nodes)) {
        expect(node.type.length).toBeGreaterThan(0);
      }
    }
  });

  it('calling createRlrmpPart2Project twice produces identical structure', () => {
    const p1 = createRlrmpPart2Project();
    const p2 = createRlrmpPart2Project();

    expect(p1.pages.length).toBe(p2.pages.length);
    for (let i = 0; i < p1.pages.length; i++) {
      expect(p1.pages[i].name).toBe(p2.pages[i].name);
      expect(Object.keys(p1.pages[i].graphSpec.nodes).length).toBe(
        Object.keys(p2.pages[i].graphSpec.nodes).length,
      );
      expect(p1.pages[i].graphSpec.wires.length).toBe(
        p2.pages[i].graphSpec.wires.length,
      );
    }
  });
});
