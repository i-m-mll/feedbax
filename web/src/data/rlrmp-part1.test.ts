/**
 * Validation tests for the RLRMP Part 1 project template.
 *
 * These are structural tests that verify the template generates a
 * well-formed AnalysisSnapshot with correct pages, nodes, wires,
 * and eval parametrization. They do not require a DOM or browser
 * environment.
 *
 * Run: npx tsx web/src/data/rlrmp-part1.test.ts
 * (from the feedbax repo root)
 */

import { createRlrmpPart1Analysis, RLRMP_PART1_TEMPLATE } from './rlrmp-part1';
import type {
  AnalysisSnapshot,
  AnalysisPageSpec,
  AnalysisNodeSpec,
  AnalysisWire,
} from '../types/analysis';

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${message}`);
  }
}

function assertEq<T>(actual: T, expected: T, message: string): void {
  if (actual === expected) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${message} (got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)})`);
  }
}

function section(name: string): void {
  console.log(`\n--- ${name} ---`);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const snapshot = createRlrmpPart1Analysis();

section('Snapshot structure');
assert(snapshot !== null && snapshot !== undefined, 'snapshot is defined');
assert(Array.isArray(snapshot.pages), 'pages is an array');
assertEq(snapshot.pages.length, 4, 'has 4 pages');
assert(snapshot.activePageId !== null, 'activePageId is set');
assertEq(snapshot.activePageId, snapshot.pages[0].id, 'first page is active');

section('Template metadata');
assertEq(RLRMP_PART1_TEMPLATE.name, 'RLRMP: Part 1', 'template name');
assertEq(RLRMP_PART1_TEMPLATE.pageNames.length, 4, 'template lists 4 page names');

section('Page names and IDs');
const expectedPageNames = ['plant_perts', 'feedback_perts', 'freq_response', 'unit_prefs'];
for (let i = 0; i < expectedPageNames.length; i++) {
  assertEq(snapshot.pages[i].name, expectedPageNames[i], `page ${i} name`);
  assert(snapshot.pages[i].id.length > 0, `page ${i} has non-empty id`);
}

// Check unique page IDs
const pageIds = snapshot.pages.map((p) => p.id);
assertEq(new Set(pageIds).size, 4, 'all page IDs are unique');

// ---------------------------------------------------------------------------
// Validate each page's graph structure
// ---------------------------------------------------------------------------

function validatePage(page: AnalysisPageSpec, expectedName: string, checks: {
  minNodes: number;
  minWires: number;
  nodeTypes: string[];
  hasEvalParams: string[];
  fieldPaths?: string[];
}): void {
  section(`Page: ${expectedName}`);

  assertEq(page.name, expectedName, 'page name');
  assert(page.graphSpec !== null, 'graphSpec exists');
  assertEq(page.graphSpec.dataSourceId, '__data_source__', 'data source ID');

  const nodeCount = Object.keys(page.graphSpec.nodes).length;
  assert(nodeCount >= checks.minNodes, `has >= ${checks.minNodes} nodes (got ${nodeCount})`);
  assert(page.graphSpec.wires.length >= checks.minWires, `has >= ${checks.minWires} wires (got ${page.graphSpec.wires.length})`);

  // Check expected node types exist
  const nodeTypes = new Set(Object.values(page.graphSpec.nodes).map((n) => n.type));
  for (const nt of checks.nodeTypes) {
    assert(nodeTypes.has(nt), `has node type "${nt}"`);
  }

  // Check eval params
  for (const key of checks.hasEvalParams) {
    assert(key in page.evalParams, `evalParams has "${key}"`);
  }

  // Check field paths on wires
  if (checks.fieldPaths) {
    const wirePaths = page.graphSpec.wires
      .filter((w) => w.fieldPath)
      .map((w) => w.fieldPath!);
    for (const fp of checks.fieldPaths) {
      assert(wirePaths.includes(fp), `has wire with fieldPath "${fp}"`);
    }
  }

  // Validate all wires reference existing nodes (or the data source)
  const validNodeIds = new Set([page.graphSpec.dataSourceId, ...Object.keys(page.graphSpec.nodes)]);
  // Transform nodes are also valid wire endpoints (referenced by transform.id)
  for (const wire of page.graphSpec.wires) {
    if (wire.transform) {
      validNodeIds.add(wire.transform.id);
    }
  }
  for (const wire of page.graphSpec.wires) {
    assert(validNodeIds.has(wire.sourceId), `wire ${wire.id} source "${wire.sourceId}" exists`);
    assert(validNodeIds.has(wire.targetId), `wire ${wire.id} target "${wire.targetId}" exists`);
  }

  // Validate wire IDs are unique within the page
  const wireIds = page.graphSpec.wires.map((w) => w.id);
  assertEq(new Set(wireIds).size, wireIds.length, 'all wire IDs unique');

  // Validate node IDs match their keys in the nodes map
  for (const [key, node] of Object.entries(page.graphSpec.nodes)) {
    assertEq(node.id, key, `node "${key}" id matches its key`);
  }

  // Validate viewport
  assert(typeof page.viewport.x === 'number', 'viewport.x is number');
  assert(typeof page.viewport.y === 'number', 'viewport.y is number');
  assert(typeof page.viewport.zoom === 'number', 'viewport.zoom is number');
}

// Page 1: plant_perts
// Profiles receives the full state tree (fieldPath: 'states'), not states.net.hidden
validatePage(snapshot.pages[0], 'plant_perts', {
  minNodes: 7,
  minWires: 7,
  nodeTypes: ['GetBestReplicate', 'AlignedVars', 'ApplyFns', 'Violins', 'EffectorTrajectories', 'Profiles'],
  hasEvalParams: ['perturbation_type', 'perturbation_amplitudes'],
  fieldPaths: ['states'],
});

// Page 2: feedback_perts
validatePage(snapshot.pages[1], 'feedback_perts', {
  minNodes: 3,
  minWires: 3,
  nodeTypes: ['AlignedVars', 'ApplyFns', 'Violins'],
  hasEvalParams: ['perturbation_type', 'perturbation_amplitudes', 'perturbation_variables', 'perturbation_direction'],
  fieldPaths: ['states'],
});

// Page 3: freq_response
validatePage(snapshot.pages[2], 'freq_response', {
  minNodes: 1,
  minWires: 1,
  nodeTypes: ['FrequencyResponse'],
  hasEvalParams: ['perturbation_type'],
  fieldPaths: ['states'],
});

// Page 4: unit_prefs
// UnitPreferences extracts features internally via feature_fn — no separate
// DataSource wires for feature fields (states.efferent.output, task targets).
validatePage(snapshot.pages[3], 'unit_prefs', {
  minNodes: 4,
  minWires: 4,
  nodeTypes: ['GetBestReplicate', 'SegmentEpochs', 'UnitPreferences'],
  hasEvalParams: ['perturbation_type', 'perturbation_amplitudes'],
  fieldPaths: ['states'],
});

// ---------------------------------------------------------------------------
// Cross-page consistency checks
// ---------------------------------------------------------------------------

section('Cross-page consistency');

// All node IDs should be globally unique (prefixed by page name)
const allNodeIds = snapshot.pages.flatMap((p) => Object.keys(p.graphSpec.nodes));
assertEq(new Set(allNodeIds).size, allNodeIds.length, 'all node IDs globally unique');

// All wire IDs should be globally unique
const allWireIds = snapshot.pages.flatMap((p) => p.graphSpec.wires.map((w) => w.id));
assertEq(new Set(allWireIds).size, allWireIds.length, 'all wire IDs globally unique');

// Check measure names match the Python source
section('Measure name fidelity');

const plantPertsPage = snapshot.pages[0];
const plantMeasures = Object.values(plantPertsPage.graphSpec.nodes).find((n) => n.type === 'ApplyFns');
assert(plantMeasures !== undefined, 'plant_perts has ApplyFns node');
if (plantMeasures) {
  const names = plantMeasures.params.measure_names as string[];
  assert(names.includes('initial_command'), 'has initial_command');
  assert(names.includes('max_net_command'), 'has max_net_command');
  assert(names.includes('end_position_error'), 'has end_position_error');
  assert(names.includes('sum_lateral_force_abs'), 'has sum_lateral_force_abs');
  assertEq(names.length, 16, 'plant_perts has 16 measures');
}

const fbPertsPage = snapshot.pages[1];
const fbMeasures = Object.values(fbPertsPage.graphSpec.nodes).find((n) => n.type === 'ApplyFns');
assert(fbMeasures !== undefined, 'feedback_perts has ApplyFns node');
if (fbMeasures) {
  const names = fbMeasures.params.measure_names as string[];
  assert(names.includes('max_net_force'), 'has max_net_force');
  assert(names.includes('max_deviation'), 'has max_deviation');
  assert(names.includes('sum_deviation'), 'has sum_deviation');
  assertEq(names.length, 9, 'feedback_perts has 9 measures');
}

// Check frequency response node parameters
section('FrequencyResponse fidelity');
const freqPage = snapshot.pages[2];
const freqNode = Object.values(freqPage.graphSpec.nodes).find((n) => n.type === 'FrequencyResponse');
assert(freqNode !== undefined, 'freq_response has FrequencyResponse node');
if (freqNode) {
  assertEq(freqNode.params.input_field as string, 'states.feedback.noise', 'input_field matches Python INPUT_WHERE');
  assertEq(freqNode.params.output_field as string, 'states.net.output', 'output_field matches Python OUTPUT_WHERE');
  const fbVarNames = freqNode.params.fb_var_names as string[];
  assert(Array.isArray(fbVarNames), 'fb_var_names is an array');
  assert(fbVarNames.includes('fb_pos'), 'fb_var_names includes fb_pos');
  assert(fbVarNames.includes('fb_vel'), 'fb_var_names includes fb_vel');
}

// Check unit prefs has two instances
section('UnitPreferences fidelity');
const unitPrefsPage = snapshot.pages[3];
const unitPrefNodes = Object.values(unitPrefsPage.graphSpec.nodes).filter((n) => n.type === 'UnitPreferences');
assertEq(unitPrefNodes.length, 2, 'unit_prefs has 2 UnitPreferences nodes');
const featureFns = unitPrefNodes.map((n) => n.params.feature_fn as string).sort();
assert(featureFns.includes('control_forces'), 'has control_forces UnitPreferences');
assert(featureFns.includes('goal_positions'), 'has goal_positions UnitPreferences');

// Check transforms
section('Transform fidelity');
const transformWires = snapshot.pages.flatMap((p) =>
  p.graphSpec.wires.filter((w) => w.transform)
);
assert(transformWires.length >= 2, `at least 2 transform wires (got ${transformWires.length})`);
const transformTypes = new Set(transformWires.map((w) => w.transform!.type));
assert(transformTypes.has('GetitemAtLevel'), 'has GetitemAtLevel transform (discard varset)');

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`);
if (failed > 0) {
  process.exit(1);
} else {
  console.log('All tests passed.');
}
