import {
  normalizeStudioValueSpec,
  type StudioValueSpecEnumerable,
} from '@/features/scenario/valueSpecs';
import type {
  StudioScenarioSpec,
  StudioStageSpec,
  StudioValueSpec,
  StudioWorkspaceSpec,
} from '@/types/workspace';
import type { TrainingRunSummary } from '@/utils/pipelineCollections';

export type TrainMatrixMode = 'cross' | 'zip';
export type BulkEditVerb = 'keep' | 'set' | 'distribute' | 'cross';

export interface TrainMatrixAxisDraft {
  id: string;
  label: string;
  path: string;
  values: unknown[];
  source: 'selection' | 'value_spec' | 'manual';
  valueSpec?: StudioValueSpec | null;
}

export interface TrainMatrixSpec {
  name: string;
  mode: TrainMatrixMode;
  axes: TrainMatrixAxisDraft[];
  manualCoordinates?: Array<Record<string, number>>;
}

export interface TrainMatrixCoordinate {
  index: number;
  valueIndices: Record<string, number>;
  values: Record<string, unknown>;
  label: string;
}

export interface TrainMatrixGhostRow {
  id: string;
  label: string;
  status: 'ghost';
  runSetId: string;
  coordinateIndex: number;
  axisCoordinates: Record<string, unknown>;
}

export interface TrainAxisColumn {
  id: string;
  label: string;
  path?: string | null;
}

export interface MatrixFromGhostRowsResult {
  matrix: TrainMatrixSpec | null;
  error: string | null;
}

const DEFAULT_MATRIX_NAME = 'Training matrix';
const FIRST_CLASS_AXIS_PATHS = new Set(['seed', 'training_seed']);

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function stableAxisId(path: string): string {
  return path
    .replace(/^(training_spec|task_spec|task_binding_spec|graph_spec)\./, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase() || 'axis';
}

function uniqueAxisId(base: string, seen: Set<string>): string {
  let id = base;
  let index = 2;
  while (seen.has(id)) {
    id = `${base}_${index}`;
    index += 1;
  }
  seen.add(id);
  return id;
}

export function formatAxisValue(value: unknown): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (Math.abs(value) >= 100) return value.toFixed(0);
    if (Math.abs(value) >= 1) return Number(value.toFixed(4)).toString();
    return value.toPrecision(4);
  }
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return JSON.stringify(value);
}

export function parseAxisValuesInput(input: string): unknown[] {
  const trimmed = input.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return trimmed
      .split(',')
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const numeric = Number(part);
        if (part !== '' && Number.isFinite(numeric)) return numeric;
        if (part === 'true') return true;
        if (part === 'false') return false;
        return part;
      });
  }
}

function valuesFromEnumerable(enumerable: StudioValueSpecEnumerable | null | undefined): unknown[] {
  if (!enumerable) return [];
  if (enumerable.form === 'list') return [...(enumerable.values ?? [])];
  if (enumerable.form === 'sampler') {
    const n = typeof enumerable.n === 'number' ? Math.max(0, Math.round(enumerable.n)) : 0;
    return Array.from({ length: n }, (_, index) => `sample ${index + 1}`);
  }
  const count = typeof enumerable.count === 'number' ? Math.max(0, Math.round(enumerable.count)) : 0;
  const start = enumerable.start;
  const stop = enumerable.stop;
  if (typeof start !== 'number' || typeof stop !== 'number' || count <= 0) return [];
  if (count === 1) return [start];
  if (enumerable.scale === 'log') {
    if (start <= 0 || stop <= 0) return [];
    const logStart = Math.log10(start);
    const step = (Math.log10(stop) - logStart) / (count - 1);
    return Array.from({ length: count }, (_, index) => 10 ** (logStart + step * index));
  }
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + step * index);
}

function axisFromValueSpec(path: string, label: string, valueSpec: StudioValueSpec): TrainMatrixAxisDraft | null {
  const normalized = normalizeStudioValueSpec(valueSpec);
  const variation = normalized.variation;
  if (variation?.scope !== 'sweep') return null;
  const values = valuesFromEnumerable(variation.enumerable);
  if (values.length === 0) return null;
  return {
    id: stableAxisId(path),
    label,
    path,
    values,
    source: 'value_spec',
    valueSpec: normalized,
  };
}

function visitValueSpecs(
  value: unknown,
  path: string,
  label: string,
  out: TrainMatrixAxisDraft[]
) {
  if (!isRecord(value)) return;
  if (typeof value.mode === 'string' || typeof value.value_form === 'string') {
    const axis = axisFromValueSpec(path, label, value as unknown as StudioValueSpec);
    if (axis) out.push(axis);
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    if (!isRecord(child) && !Array.isArray(child)) continue;
    visitValueSpecs(child, `${path}.${key}`, key, out);
  }
}

export function matrixAxesFromScenario(scenario: StudioScenarioSpec | null | undefined): TrainMatrixAxisDraft[] {
  const axes: TrainMatrixAxisDraft[] = [];
  if (!scenario) return axes;
  if (scenario.training_spec) {
    visitValueSpecs(scenario.training_spec, 'training_spec', 'Training spec', axes);
  }
  if (scenario.task_spec) {
    visitValueSpecs(scenario.task_spec, 'task_spec', 'Task spec', axes);
  }
  if (scenario.task_binding_spec) {
    visitValueSpecs(scenario.task_binding_spec, 'task_binding_spec', 'Task binding', axes);
  }
  return dedupeAxes(axes);
}

export function matrixSpecFromSelection(stage: StudioStageSpec | null | undefined): TrainMatrixSpec | null {
  const raw = stage?.selection_spec.matrix;
  if (!isRecord(raw)) return null;
  const rawAxes = Array.isArray(raw.axes) ? raw.axes : [];
  const axes = rawAxes.flatMap((axis): TrainMatrixAxisDraft[] => {
    if (!isRecord(axis)) return [];
    const path = stringValue(axis.path);
    if (!path) return [];
    const values = Array.isArray(axis.values)
      ? axis.values
      : isRecord(axis.variation) && Array.isArray(axis.variation.values)
        ? axis.variation.values
        : [];
    if (values.length === 0) return [];
    return [{
      id: stringValue(axis.id) ?? stableAxisId(path),
      label: stringValue(axis.label) ?? path,
      path,
      values,
      source: 'selection',
    }];
  });
  const combination = isRecord(raw.combination) ? raw.combination : null;
  const manualCoordinates =
    combination?.mode === 'manual' && Array.isArray(combination.manual_coordinates)
      ? combination.manual_coordinates.flatMap((coordinate): Array<Record<string, number>> => {
          if (!isRecord(coordinate)) return [];
          const normalized = Object.fromEntries(
            Object.entries(coordinate).filter(([, value]) =>
              typeof value === 'number' && Number.isInteger(value) && value >= 0
            )
          ) as Record<string, number>;
          return Object.keys(normalized).length > 0 ? [normalized] : [];
        })
      : undefined;
  return {
    name: stringValue(raw.name) ?? stringValue(raw.label) ?? DEFAULT_MATRIX_NAME,
    mode: raw.mode === 'zip' ? 'zip' : 'cross',
    axes: dedupeAxes(axes),
    manualCoordinates,
  };
}

export function initialMatrixSpec(
  stage: StudioStageSpec | null | undefined,
  scenario: StudioScenarioSpec | null | undefined
): TrainMatrixSpec {
  return matrixSpecFromSelection(stage) ?? {
    name: stage?.label ? `${stage.label} matrix` : DEFAULT_MATRIX_NAME,
    mode: 'cross',
    axes: matrixAxesFromScenario(scenario),
  };
}

function dedupeAxes(axes: TrainMatrixAxisDraft[]): TrainMatrixAxisDraft[] {
  const seenPaths = new Set<string>();
  const seenIds = new Set<string>();
  const out: TrainMatrixAxisDraft[] = [];
  for (const axis of axes) {
    if (seenPaths.has(axis.path)) continue;
    seenPaths.add(axis.path);
    out.push({ ...axis, id: uniqueAxisId(axis.id, seenIds) });
  }
  return out;
}

export function runCountExpression(
  axes: TrainMatrixAxisDraft[],
  mode: TrainMatrixMode,
  manualCoordinates?: Array<Record<string, number>>
): string {
  if (manualCoordinates && manualCoordinates.length > 0) {
    return `${manualCoordinates.length} manual run${manualCoordinates.length === 1 ? '' : 's'}`;
  }
  if (axes.length === 0) return '0 runs';
  const counts = axes.map((axis) => axis.values.length);
  if (mode === 'zip') {
    const mismatch = new Set(counts).size > 1;
    return `${counts.join(' zip ')} = ${mismatch ? 'mismatch' : `${counts[0]} runs`}`;
  }
  const total = counts.reduce((product, count) => product * count, 1);
  return `${counts.join(' x ')} = ${total} run${total === 1 ? '' : 's'}`;
}

export function expandTrainMatrix(
  axes: TrainMatrixAxisDraft[],
  mode: TrainMatrixMode,
  manualCoordinates?: Array<Record<string, number>>
): TrainMatrixCoordinate[] {
  if (axes.length === 0 || axes.some((axis) => axis.values.length === 0)) return [];
  const coordinates: Array<Record<string, number>> = [];
  if (manualCoordinates && manualCoordinates.length > 0) {
    for (const coordinate of manualCoordinates) {
      const normalized: Record<string, number> = {};
      for (const axis of axes) {
        const index = coordinate[axis.id];
        if (!Number.isInteger(index) || index < 0 || index >= axis.values.length) return [];
        normalized[axis.id] = index;
      }
      coordinates.push(normalized);
    }
  } else if (mode === 'zip') {
    const lengths = new Set(axes.map((axis) => axis.values.length));
    if (lengths.size !== 1) return [];
    for (let index = 0; index < axes[0].values.length; index += 1) {
      coordinates.push(Object.fromEntries(axes.map((axis) => [axis.id, index])));
    }
  } else {
    const visit = (axisIndex: number, coordinate: Record<string, number>) => {
      const axis = axes[axisIndex];
      if (!axis) {
        coordinates.push({ ...coordinate });
        return;
      }
      axis.values.forEach((_, valueIndex) => {
        visit(axisIndex + 1, { ...coordinate, [axis.id]: valueIndex });
      });
    };
    visit(0, {});
  }
  return coordinates.map((valueIndices, index) => {
    const values = Object.fromEntries(
      axes.map((axis) => [axis.id, axis.values[valueIndices[axis.id]]])
    );
    return {
      index,
      valueIndices,
      values,
      label: axes.map((axis) => `${axis.label} ${formatAxisValue(values[axis.id])}`).join(', '),
    };
  });
}

export function ghostRowsForMatrix(matrix: TrainMatrixSpec): TrainMatrixGhostRow[] {
  const runSetId = `ghost-run-set:${stableAxisId(matrix.name) || 'matrix'}`;
  return expandTrainMatrix(matrix.axes, matrix.mode, matrix.manualCoordinates).map((coordinate) => ({
    id: `${runSetId}:${coordinate.index}`,
    label: coordinate.label || `Run ${coordinate.index + 1}`,
    status: 'ghost',
    runSetId,
    coordinateIndex: coordinate.index,
    axisCoordinates: coordinate.values,
  }));
}

export function selectionSpecForMatrix(
  current: Record<string, unknown>,
  matrix: TrainMatrixSpec
): Record<string, unknown> {
  const combination =
    matrix.manualCoordinates && matrix.manualCoordinates.length > 0
      ? {
          combination: {
            mode: 'manual',
            manual_coordinates: matrix.manualCoordinates,
          },
        }
      : {};
  return {
    ...current,
    matrix: {
      name: matrix.name,
      mode: matrix.mode,
      axes: matrix.axes.map((axis) => ({
        id: axis.id,
        label: axis.label,
        path: axis.path,
        values: axis.values,
      })),
      ...combination,
    },
  };
}

export function selectionSpecWithoutMatrix(current: Record<string, unknown>): Record<string, unknown> {
  const { matrix: _matrix, sweep_matrix: _sweepMatrix, run_matrix: _runMatrix, ...rest } = current;
  return rest;
}

export function workspaceWithMatrixSelection(
  workspace: StudioWorkspaceSpec,
  stageId: string,
  matrix: TrainMatrixSpec
): StudioWorkspaceSpec {
  return {
    ...workspace,
    stages: workspace.stages.map((stage) =>
      stage.id === stageId
        ? {
            ...stage,
            selection_spec: selectionSpecForMatrix(stage.selection_spec, matrix),
          }
        : stage
    ),
  };
}

export function workspaceWithoutMatrixSelection(
  workspace: StudioWorkspaceSpec,
  stageId: string
): StudioWorkspaceSpec {
  return {
    ...workspace,
    stages: workspace.stages.map((stage) =>
      stage.id === stageId
        ? {
            ...stage,
            selection_spec: selectionSpecWithoutMatrix(stage.selection_spec),
          }
        : stage
    ),
  };
}

export function trainAxisColumns(
  rows: TrainingRunSummary[],
  matrixAxes: TrainMatrixAxisDraft[] = []
): TrainAxisColumn[] {
  const byId = new Map<string, TrainAxisColumn>();
  const matrixAxisIds = new Set<string>();
  for (const axis of matrixAxes) {
    matrixAxisIds.add(axis.id);
    byId.set(axis.id, { id: axis.id, label: axis.label, path: axis.path });
  }
  for (const row of rows) {
    for (const axisId of Object.keys(row.axisCoordinates)) {
      if (!byId.has(axisId)) byId.set(axisId, { id: axisId, label: axisId.replace(/_/g, ' ') });
    }
  }
  return Array.from(byId.values()).filter((axis) =>
    matrixAxisIds.has(axis.id) || rows.some((row) => row.axisCoordinates[axis.id] !== undefined)
  );
}

export function validateAxisPath(path: string, scenario: StudioScenarioSpec | null | undefined): string | null {
  if (FIRST_CLASS_AXIS_PATHS.has(path)) return null;
  const [root, ...parts] = path.split('.');
  const rootValue =
    root === 'training_spec'
      ? scenario?.training_spec
      : root === 'task_spec'
        ? scenario?.task_spec
        : root === 'task_binding_spec'
          ? scenario?.task_binding_spec
          : null;
  if (!rootValue) return 'Axis path must start with training_spec, task_spec, or task_binding_spec.';
  let current: unknown = rootValue;
  for (const part of parts) {
    if (Array.isArray(current)) {
      const index = Number(part);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) {
        return `Axis path segment ${part} does not resolve.`;
      }
      current = current[index];
      continue;
    }
    if (!isRecord(current) || !(part in current)) {
      return `Axis path segment ${part} does not resolve.`;
    }
    current = current[part];
  }
  return parts.length === 0 ? 'Axis path must include a field after the root.' : null;
}

export function matrixSpecToValuesInput(axis: TrainMatrixAxisDraft): string {
  return axis.values.map(formatAxisValue).join(', ');
}

export function bulkEditGhostRows({
  rows,
  axis,
  verb,
  values,
}: {
  rows: TrainingRunSummary[];
  axis: TrainAxisColumn | TrainMatrixAxisDraft | null;
  verb: BulkEditVerb;
  values: unknown[];
}): TrainMatrixGhostRow[] {
  if (!axis || rows.length === 0 || verb === 'keep') return [];
  if (values.length === 0) return [];
  const out: TrainMatrixGhostRow[] = [];
  const addRow = (row: TrainingRunSummary, value: unknown, index: number) => {
    out.push({
      id: `bulk-preview:${row.id}:${axis.id}:${index}`,
      label: `${row.label} - ${axis.label} ${formatAxisValue(value)}`,
      status: 'ghost',
      runSetId: row.runSetId ?? 'bulk-preview',
      coordinateIndex: index,
      axisCoordinates: {
        ...row.axisCoordinates,
        [axis.id]: value,
      },
    });
  };
  if (verb === 'set') {
    rows.forEach((row, index) => addRow(row, values[0], index));
  } else if (verb === 'distribute') {
    rows.forEach((row, index) => {
      if (values[index] !== undefined) addRow(row, values[index], index);
    });
  } else if (verb === 'cross') {
    rows.forEach((row) => {
      values.forEach((value) => addRow(row, value, out.length));
    });
  }
  return out;
}

function valueKey(value: unknown): string {
  return JSON.stringify(value);
}

export function matrixSpecFromGhostRows({
  name,
  rows,
  axes,
}: {
  name: string;
  rows: TrainMatrixGhostRow[];
  axes: TrainAxisColumn[];
}): MatrixFromGhostRowsResult {
  if (rows.length === 0) return { matrix: null, error: 'No preview rows to stage.' };
  const selectedAxes = axes.filter((axis) =>
    rows.some((row) => row.axisCoordinates[axis.id] !== undefined)
  );
  if (selectedAxes.length === 0) {
    return { matrix: null, error: 'Preview rows do not contain axis coordinates.' };
  }
  const missingPath = selectedAxes.find((axis) => !axis.path);
  if (missingPath) {
    return {
      matrix: null,
      error: `Axis ${missingPath.label} is missing its scenario path, so it cannot be restaged.`,
    };
  }

  const matrixAxes: TrainMatrixAxisDraft[] = [];
  const valueIndexes = new Map<string, Map<string, number>>();
  for (const axis of selectedAxes) {
    const values: unknown[] = [];
    const indexes = new Map<string, number>();
    for (const row of rows) {
      if (row.axisCoordinates[axis.id] === undefined) {
        return {
          matrix: null,
          error: `Preview row ${row.label} is missing axis ${axis.label}.`,
        };
      }
      const value = row.axisCoordinates[axis.id];
      const key = valueKey(value);
      if (!indexes.has(key)) {
        indexes.set(key, values.length);
        values.push(value);
      }
    }
    matrixAxes.push({
      id: axis.id,
      label: axis.label,
      path: axis.path ?? axis.id,
      values,
      source: 'manual',
    });
    valueIndexes.set(axis.id, indexes);
  }

  const manualCoordinates = rows.map((row) =>
    Object.fromEntries(
      matrixAxes.map((axis) => [
        axis.id,
        valueIndexes.get(axis.id)?.get(valueKey(row.axisCoordinates[axis.id])) ?? 0,
      ])
    )
  );
  return {
    matrix: {
      name,
      mode: 'cross',
      axes: matrixAxes,
      manualCoordinates,
    },
    error: null,
  };
}
