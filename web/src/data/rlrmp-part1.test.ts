import { describe, expect, it } from 'vitest';
import { createRlrmpPart1Analysis, RLRMP_PART1_TEMPLATE } from './rlrmp-part1';
import type { AnalysisPageSpec } from '../types/analysis';

interface PageChecks {
  minNodes: number;
  minWires: number;
  nodeTypes: string[];
  hasEvalParams: string[];
  fieldPaths?: string[];
}

function validatePage(
  page: AnalysisPageSpec,
  expectedName: string,
  checks: PageChecks
): void {
  expect(page.name).toBe(expectedName);
  expect(page.graphSpec).not.toBeNull();
  expect(page.graphSpec.dataSourceId).toBe('__data_source__');

  const nodes = Object.values(page.graphSpec.nodes);
  expect(nodes.length).toBeGreaterThanOrEqual(checks.minNodes);
  expect(page.graphSpec.wires.length).toBeGreaterThanOrEqual(checks.minWires);

  const nodeTypes = new Set(nodes.map((node) => node.type));
  for (const type of checks.nodeTypes) {
    expect(nodeTypes.has(type)).toBe(true);
  }

  for (const key of checks.hasEvalParams) {
    expect(page.evalParams).toHaveProperty(key);
  }

  if (checks.fieldPaths) {
    const wirePaths = page.graphSpec.wires
      .filter((wire) => wire.fieldPath)
      .map((wire) => wire.fieldPath);
    expect(wirePaths).toEqual(expect.arrayContaining(checks.fieldPaths));
  }

  const validNodeIds = new Set([
    page.graphSpec.dataSourceId,
    ...Object.keys(page.graphSpec.nodes),
  ]);
  for (const wire of page.graphSpec.wires) {
    if (wire.transform) validNodeIds.add(wire.transform.id);
  }
  for (const wire of page.graphSpec.wires) {
    expect(validNodeIds.has(wire.sourceId)).toBe(true);
    expect(validNodeIds.has(wire.targetId)).toBe(true);
  }

  const wireIds = page.graphSpec.wires.map((wire) => wire.id);
  expect(new Set(wireIds).size).toBe(wireIds.length);

  for (const [key, node] of Object.entries(page.graphSpec.nodes)) {
    expect(node.id).toBe(key);
  }

  expect(typeof page.viewport.x).toBe('number');
  expect(typeof page.viewport.y).toBe('number');
  expect(typeof page.viewport.zoom).toBe('number');
}

describe('RLRMP Part 1 analysis template', () => {
  it('creates a valid four-page analysis snapshot', () => {
    const snapshot = createRlrmpPart1Analysis();

    expect(snapshot.pages).toHaveLength(4);
    expect(snapshot.activePageId).toBe(snapshot.pages[0].id);
    expect(RLRMP_PART1_TEMPLATE.name).toBe('RLRMP: Part 1');
    expect(RLRMP_PART1_TEMPLATE.pageNames).toHaveLength(4);
    expect(snapshot.pages.map((page) => page.name)).toEqual([
      'plant_perts',
      'feedback_perts',
      'freq_response',
      'unit_prefs',
    ]);
    expect(new Set(snapshot.pages.map((page) => page.id)).size).toBe(4);
  });

  it('keeps each page graph structurally valid', () => {
    const snapshot = createRlrmpPart1Analysis();

    validatePage(snapshot.pages[0], 'plant_perts', {
      minNodes: 7,
      minWires: 7,
      nodeTypes: [
        'GetBestReplicate',
        'AlignedVars',
        'ApplyFns',
        'Violins',
        'EffectorTrajectories',
        'Profiles',
      ],
      hasEvalParams: ['perturbation_type', 'perturbation_amplitudes'],
      fieldPaths: ['states'],
    });

    validatePage(snapshot.pages[1], 'feedback_perts', {
      minNodes: 3,
      minWires: 3,
      nodeTypes: ['AlignedVars', 'ApplyFns', 'Violins'],
      hasEvalParams: [
        'perturbation_type',
        'perturbation_amplitudes',
        'perturbation_variables',
        'perturbation_direction',
      ],
      fieldPaths: ['states'],
    });

    validatePage(snapshot.pages[2], 'freq_response', {
      minNodes: 1,
      minWires: 1,
      nodeTypes: ['FrequencyResponse'],
      hasEvalParams: ['perturbation_type'],
      fieldPaths: ['states'],
    });

    validatePage(snapshot.pages[3], 'unit_prefs', {
      minNodes: 4,
      minWires: 4,
      nodeTypes: ['GetBestReplicate', 'SegmentEpochs', 'UnitPreferences'],
      hasEvalParams: ['perturbation_type', 'perturbation_amplitudes'],
      fieldPaths: ['states'],
    });
  });

  it('keeps node and wire identifiers unique across pages', () => {
    const snapshot = createRlrmpPart1Analysis();

    const allNodeIds = snapshot.pages.flatMap((page) => Object.keys(page.graphSpec.nodes));
    const allWireIds = snapshot.pages.flatMap((page) =>
      page.graphSpec.wires.map((wire) => wire.id)
    );

    expect(new Set(allNodeIds).size).toBe(allNodeIds.length);
    expect(new Set(allWireIds).size).toBe(allWireIds.length);
  });

  it('preserves measure names from the Python source', () => {
    const snapshot = createRlrmpPart1Analysis();
    const plantMeasures = Object.values(snapshot.pages[0].graphSpec.nodes).find(
      (node) => node.type === 'ApplyFns'
    );
    const feedbackMeasures = Object.values(snapshot.pages[1].graphSpec.nodes).find(
      (node) => node.type === 'ApplyFns'
    );

    expect(plantMeasures).toBeDefined();
    expect(plantMeasures?.params.measure_names).toEqual(
      expect.arrayContaining([
        'initial_command',
        'max_net_command',
        'end_position_error',
        'sum_lateral_force_abs',
      ])
    );
    expect(plantMeasures?.params.measure_names).toHaveLength(16);

    expect(feedbackMeasures).toBeDefined();
    expect(feedbackMeasures?.params.measure_names).toEqual(
      expect.arrayContaining(['max_net_force', 'max_deviation', 'sum_deviation'])
    );
    expect(feedbackMeasures?.params.measure_names).toHaveLength(9);
  });

  it('preserves frequency response and unit preference settings', () => {
    const snapshot = createRlrmpPart1Analysis();
    const freqNode = Object.values(snapshot.pages[2].graphSpec.nodes).find(
      (node) => node.type === 'FrequencyResponse'
    );
    const unitPrefNodes = Object.values(snapshot.pages[3].graphSpec.nodes).filter(
      (node) => node.type === 'UnitPreferences'
    );

    expect(freqNode).toBeDefined();
    expect(freqNode?.params.input_field).toBe('states.feedback.noise');
    expect(freqNode?.params.output_field).toBe('states.net.output');
    expect(freqNode?.params.fb_var_names).toEqual(
      expect.arrayContaining(['fb_pos', 'fb_vel'])
    );

    expect(unitPrefNodes).toHaveLength(2);
    expect(unitPrefNodes.map((node) => node.params.feature_fn).sort()).toEqual([
      'control_forces',
      'goal_positions',
    ]);
  });

  it('keeps transform wires needed by the imported analysis graph', () => {
    const snapshot = createRlrmpPart1Analysis();
    const transformWires = snapshot.pages.flatMap((page) =>
      page.graphSpec.wires.filter((wire) => wire.transform)
    );
    const transformTypes = new Set(transformWires.map((wire) => wire.transform?.type));

    expect(transformWires.length).toBeGreaterThanOrEqual(2);
    expect(transformTypes.has('GetitemAtLevel')).toBe(true);
  });
});
