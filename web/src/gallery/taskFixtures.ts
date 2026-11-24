/**
 * A stand-in for the service's Task view projection and its trial sampler, over four tasks: the delayed-reaches
 * document as shipped, the three-reach task of issue 7f0c01f's acceptance 8 (an Until End, a × 3 repeat, a list
 * variable, `targets[i]` cells), the stabilization document (which leaves the model's input unfed), and a Python task.
 */
import type { FunctionInfo, ScopedValue, TextSpan, ValueType } from '@/generated/studioContracts';
import type { TimelineEpoch, TimelineInterval } from '@/components/timeline/EpochTimeline';
import type { ChangeMark, FieldModel, ValueKind } from '@/components/values/fieldModel';
import { rawField } from '@/components/values/adapters/raw';
import type {
  TaskCell, TaskDocumentView, TaskEpoch, TaskPythonView, TaskSignal, TaskVariable, VariableLevel,
} from '@/features/projectFolder/task/taskModel';
import type { Percentiles, SampledTrial, SignalAggregate, SignalSteps, TrialBatch } from '@/features/projectFolder/task/trialSample';

export interface VariableSpec {
  id: string; name: string; type: string; count?: number; level?: VariableLevel; text: string;
  /** A problem with the value, and the term of its text it is about. */
  reason?: string; span?: TextSpan; changed?: string;
}
export interface EpochSpec { id: string; name: string; end: string; least?: number; changed?: string }
export interface CellSpec { first: string; last: string; text: string | null }
export interface SignalSpec { id: string; name: string; type: string; cells: CellSpec[] }

export interface DocumentSpec {
  kind: 'document';
  variables: VariableSpec[];
  epochs: EpochSpec[];
  repeats: { id: string; first: string; last: string; count: string }[];
  signals: SignalSpec[];
  duration: number;
  stepSeconds: number;
  /** The document this one is based on, which its changed values differ from. */
  base?: string;
}

export interface TaskFixture {
  key: string;
  name: string;
  python?: boolean;
  problem?: string;
  spec: DocumentSpec | PythonSpec;
  /** Trials sampled under `seed`. */
  sample?(seed: number, count: number): SampledTrial[];
}

export interface PythonSpec {
  kind: 'python';
  values: Record<string, string | number | boolean>;
}

/** The catalogue's types by name. */
export const types: Readonly<Record<string, ValueType>> = {
  Number: { base: 'real' }, Integer: { base: 'integer' }, Boolean: { base: 'boolean' },
  Duration: { base: 'real', dimensions: { time: 1 } }, Distance: { base: 'real', dimensions: { length: 1 } },
  Point: { base: 'real', shape: [2], dimensions: { length: 1 } },
  Velocity: { base: 'real', shape: [2], dimensions: { length: 1, time: -1 } }, Vector: { base: 'real', shape: [2] },
};

const functions: FunctionInfo[] = [
  { id: 'U', signature: 'U(low, high)', label: 'Uniform', display_class: 'draw', min_args: 1, max_args: 2,
    parameters: [{ name: 'low', label: 'Low' }, { name: 'high', label: 'High' }] },
  { id: 'N', signature: 'N(mean, sd)', label: 'Normal', display_class: 'draw', min_args: 2, max_args: 2,
    parameters: [{ name: 'mean', label: 'Mean' }, { name: 'sd', label: 'SD' }] },
  { id: 'bernoulli', signature: 'bernoulli(p)', label: 'Bernoulli', display_class: 'draw', min_args: 1, max_args: 1,
    parameters: [{ name: 'p', label: 'P' }] },
  { id: 'polar', signature: 'polar(radius, angle)', display_class: 'math', min_args: 2, max_args: 2 },
  { id: 'dist', signature: 'dist(a, b)', display_class: 'math', min_args: 2, max_args: 2 },
  { id: 'held', signature: 'held(condition, duration)', display_class: 'math', min_args: 2, max_args: 2 },
  { id: 'where', signature: 'where(condition, then, otherwise)', display_class: 'math', min_args: 3, max_args: 3 },
  { id: 'minjerk', signature: 'minjerk(from, to)', display_class: 'shape', min_args: 2, max_args: 2 },
];

/** The kind a text reads as, as the service would parse it. */
export function kindOf(text: string): ValueKind {
  const trimmed = text.trim();
  if (trimmed === 'rest') return 'rest';
  if (trimmed.includes('≤')) return 'until';
  if (/^-?\d*\.?\d+\s*(s|st|m|°)?$/.test(trimmed)) return 'value';
  if (/^-?\d*\.?\d+(\s*\w+)?(\s*,\s*-?\d*\.?\d+(\s*\w+)?)+,?$/.test(trimmed)) return 'sweep';
  if (/^(U|N|bernoulli)\(/.test(trimmed)) return 'draw';
  if (/^minjerk\(/.test(trimmed)) return 'shape';
  if (/^[A-Za-z_]\w*(\[i\])?$/.test(trimmed)) return 'link';
  return 'formula';
}

/** Seconds of a length's text: `0.3 s`, `5 st` at `step`. */
function seconds(text: string, step: number) {
  const match = /^\s*(-?\d*\.?\d+)\s*(s|st)?\s*$/.exec(text);
  if (!match) return undefined;
  return match[2] === 'st' ? Number(match[1]) * step : Number(match[1]);
}

/** An epoch's bounds, from its End's text, before Rest takes what remains. */
function boundsOf(epoch: EpochSpec, step: number): TimelineEpoch {
  const text = epoch.end.trim();
  const base = { id: epoch.id, name: epoch.name };
  if (text === 'rest') return { ...base, ends: 'rest', least: 0, nominal: 0, greatest: 0 };
  const until = /≤\s*(.+)$/.exec(text);
  if (until) {
    const limit = seconds(until[1], step) ?? 1;
    return { ...base, ends: 'condition', least: epoch.least ?? 0, nominal: limit, greatest: limit };
  }
  const draw = /^U\((.+),(.+)\)$/.exec(text);
  if (draw) {
    const [low, high] = [seconds(draw[1], step) ?? 0, seconds(draw[2], step) ?? 0];
    return { ...base, ends: 'drawn', least: low, nominal: (low + high) / 2, greatest: high };
  }
  const fixed = seconds(text, step) ?? 0.1;
  return { ...base, ends: 'length', least: fixed, nominal: fixed, greatest: fixed };
}

/** Each epoch's bounds; Rest's are what the others leave of the trial, a repeated epoch counted at each repetition. */
export function epochBounds(spec: DocumentSpec): TimelineEpoch[] {
  const bounds = spec.epochs.map((epoch) => boundsOf(epoch, spec.stepSeconds));
  const times = (index: number) => {
    const repeat = spec.repeats.find(({ first, last }) => {
      const a = spec.epochs.findIndex(({ id }) => id === first);
      const b = spec.epochs.findIndex(({ id }) => id === last);
      return index >= a && index <= b;
    });
    return repeat ? Number(repeat.count) || 1 : 1;
  };
  const sum = (key: 'least' | 'nominal' | 'greatest') => bounds.reduce((total, epoch, index) =>
    epoch.ends === 'rest' ? total : total + epoch[key] * times(index), 0);
  return bounds.map((epoch) => epoch.ends === 'rest' ? {
    ...epoch,
    least: Math.max(0, spec.duration - sum('greatest')),
    nominal: Math.max(0, spec.duration - sum('nominal')),
    greatest: Math.max(0, spec.duration - sum('least')),
  } : epoch);
}

/** Change callback by field id. */
export type Write = (id: string, text: string) => void;

const level = (variable: VariableSpec): ScopedValue['level'] => variable.level === 'step' ? 2 : 1;

function names(spec: DocumentSpec): ScopedValue[] {
  return spec.variables.map((variable) => ({ id: variable.id, name: variable.name, type: { base: 'real' }, level: level(variable) }));
}

const draws = /^U\(\s*(-?\d*\.?\d+)[^,]*,\s*(-?\d*\.?\d+)[^)]*\)$/;

/** A field as the service projects it: its text, its kind as the text reads, its parts where it has them. */
export function textField({ id, label, text, kinds, write, reason, span, changed, scope, finest = 1 }: {
  id: string; label: string; text: string; kinds: readonly ValueKind[]; write: Write; reason?: string; span?: TextSpan;
  changed?: ChangeMark; scope: readonly ScopedValue[]; finest?: 0 | 1 | 2;
}): FieldModel {
  const kind = kindOf(text);
  const commit: FieldModel['commit'] = async (edit) => {
    if (edit.kind === 'text') {
      if (!kinds.includes(kindOf(edit.text))) return `${label} takes ${kinds.join(', ')}`;
      write(id, edit.text);
    }
    if (edit.kind === 'reset') write(id, '');
    if (edit.kind === 'to_sweep') write(id, `${text},`);
    if (edit.kind === 'to_value') write(id, text.split(',')[0]);
    return undefined;
  };
  const draw = draws.exec(text);
  const until = kind === 'until' ? /^(.*?)\s*≤\s*(.+)$/.exec(text) : null;
  const part = (suffix: string, partLabel: string, partText: string, partKinds: readonly ValueKind[]): FieldModel =>
    textField({ id: `${id}.${suffix}`, label: partLabel, text: partText, kinds: partKinds, scope, finest,
      write: (_, next) => write(id, suffix === 'condition' ? `${next} ≤ ${until![2]}` : `${until![1]} ≤ ${next}`) });
  const members = kind === 'sweep' ? text.split(',').map((value) => value.trim()).filter(Boolean) : undefined;
  return {
    id, label, text, kind, type: { kind: 'number' }, reason, span, change: changed,
    kinds: kinds.map((offer) => ({ kind: offer, enabled: true })),
    keeps: members ? 'm0' : undefined,
    members: members && { values: members.map((value, index) => ({ id: `m${index}`, text: value, selected: index === 1 })) },
    structure: draw ? {
      value: { kind: 'draw', family: 'U', args: [Number(draw[1]), Number(draw[2])].map((value) => ({ kind: 'literal', value })) },
      arguments: [draw[1], draw[2]], density: { low: Number(draw[1]), high: Number(draw[2]), values: Array(24).fill(1) },
    } : undefined,
    parts: until ? [
      part('condition', 'Condition', until[1], ['link', 'formula']),
      part('limit', 'Limit', until[2], ['value']),
    ] : undefined,
    finest, names: scope, functions, commit,
  };
}

const lengthKinds: ValueKind[] = ['value', 'sweep', 'draw', 'formula', 'until', 'rest'];
const valueKinds: ValueKind[] = ['value', 'sweep', 'draw', 'formula', 'link'];
const cellKinds: ValueKind[] = ['value', 'draw', 'formula', 'shape', 'link'];

/** The service's projection of a task document, its texts overridden by `texts`. */
export function documentView(spec: DocumentSpec, texts: Readonly<Record<string, string>>, write: Write): TaskDocumentView {
  const scope = names(spec);
  const text = (id: string, authored: string) => texts[id] || authored;
  const changed = (from?: string): ChangeMark | undefined => from === undefined ? undefined
    : { from: { kind: 'base', document: spec.base ?? '' }, reset: async () => undefined };
  const bounds = epochBounds({ ...spec, epochs: spec.epochs.map((epoch) => ({ ...epoch, end: text(`end:${epoch.id}`, epoch.end) })) });
  const variables: TaskVariable[] = spec.variables.map((variable) => ({
    id: variable.id, name: variable.name, type: variable.type, valueType: types[variable.type], count: variable.count,
    level: variable.level ?? 'trial',
    typeChoice: rawField({ id: `type:${variable.id}`, label: 'Type', value: variable.type, change: () => undefined,
      type: { kind: 'choice', options: Object.keys(types).map((type) => ({ value: type, label: type })) } }),
    value: textField({ id: `value:${variable.id}`, label: 'Value', text: text(`value:${variable.id}`, variable.text), kinds: valueKinds,
      write, scope, reason: variable.reason, span: variable.span, changed: changed(variable.changed),
      finest: variable.level === 'step' ? 2 : 1 }),
  }));
  const epochs: TaskEpoch[] = spec.epochs.map((epoch, index) => ({
    id: epoch.id, name: epoch.name, bounds: bounds[index],
    end: textField({ id: `end:${epoch.id}`, label: 'End', text: text(`end:${epoch.id}`, epoch.end), kinds: lengthKinds, write, scope,
      changed: changed(epoch.changed) }),
  }));
  const signals: TaskSignal[] = spec.signals.map((signal) => ({
    id: signal.id, name: signal.name, type: signal.type, valueType: types[signal.type],
    absent: textField({ id: `absent:${signal.id}`, label: 'Absent', text: text(`absent:${signal.id}`, '0'), kinds: valueKinds, write, scope, finest: 1 }),
    cells: signal.cells.map((cell, index): TaskCell => {
      const id = `${signal.id}:${index}`;
      return cell.text === null ? { id, first: cell.first, last: cell.last, kind: 'absent' } : {
        id, first: cell.first, last: cell.last, kind: 'value',
        value: textField({ id: `cell:${id}`, label: signal.name, text: text(`cell:${id}`, cell.text), kinds: cellKinds, write, scope, finest: 2 }),
      };
    }),
  }));
  return {
    kind: 'document', variables, epochs, signals, duration: spec.duration, stepSeconds: spec.stepSeconds,
    repeats: spec.repeats.map((repeat) => ({ ...repeat, count: textField({
      id: `repeat:${repeat.id}`, label: 'Count', text: text(`repeat:${repeat.id}`, repeat.count), kinds: ['value'], write, scope, finest: 0,
    }) })),
  };
}

export function pythonView(spec: PythonSpec): TaskPythonView {
  const changed: ChangeMark = { from: { kind: 'base', document: 'perturbed-reaches' }, reset: async () => undefined };
  return {
    kind: 'python',
    parameters: [
      rawField({ id: 'n_reaches', label: 'Reaches', type: { kind: 'integer' }, value: spec.values.n_reaches, change: () => undefined }),
      rawField({ id: 'field', label: 'Field', value: spec.values.field, change: () => undefined, type: { kind: 'choice', options: [
        { value: 'curl', label: 'Curl' }, { value: 'constant', label: 'Constant' }, { value: 'none', label: 'None' },
      ] } }),
      rawField({ id: 'field_scale', label: 'Field Scale', type: { kind: 'number' }, value: spec.values.field_scale, unit: 'N·s/m',
        change: () => undefined, changed }),
      rawField({ id: 'catch', label: 'Catch Trials', type: { kind: 'boolean' }, value: spec.values.catch, change: () => undefined }),
    ],
    signals: [
      { id: 'target_pos', name: 'target_pos', type: 'Point' },
      { id: 'force', name: 'force', type: 'Force' },
    ],
  };
}

/** mulberry32: a seeded generator, so a seed always samples the same trials. */
function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const point = (value: readonly number[]) => `(${value.map((part) => part.toFixed(3)).join(', ')})`;

/** Steps of `count` with each epoch interval's value, absent outside the intervals `present` covers. */
function fill(steps: number, step: number, intervals: readonly TimelineInterval[],
  value: (interval: TimelineInterval, index: number) => readonly number[] | null): SignalSteps {
  const out: (readonly number[] | null)[] = Array(steps).fill(null);
  intervals.forEach((interval, index) => {
    for (let at = Math.round(interval.start / step); at < Math.round(interval.end / step); at += 1) out[at] = value(interval, index);
  });
  return out;
}

function delayedTrials(seed: number, count: number): SampledTrial[] {
  const next = random(seed);
  const step = 0.01;
  return Array.from({ length: count }, () => {
    const hold = 5 + Math.floor(next() * 10);
    const cue = 10 + Math.floor(next() * 15);
    const start = [-0.4 + next() * 0.9, -0.3 + next() * 0.75];
    const target = [-0.4 + next() * 0.9, -0.3 + next() * 0.75];
    const catchTrial = next() < 0.5;
    const intervals = [
      { id: 'hold', name: 'hold', start: 0, end: hold * step },
      { id: 'cue', name: 'cue', start: hold * step, end: (hold + cue) * step },
      { id: 'move', name: 'move', start: (hold + cue) * step, end: 0.8 },
    ];
    const at = (id: string) => (interval: TimelineInterval) => interval.id === id;
    const pick = (interval: TimelineInterval, early: readonly number[], late: readonly number[]) => interval.id === 'move' ? late : early;
    return {
      intervals,
      variables: { start: point(start), target: point(target), catch: String(catchTrial) },
      signals: {
        target_pos: fill(80, step, intervals, (interval) => at('hold')(interval) ? null : target),
        target_vel: fill(80, step, intervals, (interval) => at('hold')(interval) ? null : [0, 0]),
        goal: fill(80, step, intervals, (interval) => pick(interval, start, catchTrial ? start : target)),
        fixation: fill(80, step, intervals, (interval) => pick(interval, [1], [catchTrial ? 1 : 0])),
        target_on: fill(80, step, intervals, (interval) => at('hold')(interval) ? null : [1]),
      },
    };
  });
}

function threeReachTrials(seed: number, count: number): SampledTrial[] {
  const next = random(seed);
  const step = 0.01;
  return Array.from({ length: count }, () => {
    const radius = 0.05 * Math.sqrt(next());
    const theta = next() * 2 * Math.PI;
    const start = [radius * Math.cos(theta), radius * Math.sin(theta)];
    const angles = [next(), next(), next()].map((value) => value * 360);
    const targets = angles.map((angle) => [start[0] + 0.15 * Math.cos(angle * Math.PI / 180), start[1] + 0.15 * Math.sin(angle * Math.PI / 180)]);
    const intervals: TimelineInterval[] = [{ id: 'fixate', name: 'fixate', start: 0, end: 0.3 }];
    let time = 0.3;
    for (let reach = 0; reach < 3; reach += 1) {
      const move = Math.round((0.35 + next() * 0.45) / step) * step;
      intervals.push({ id: 'move', name: `move ${reach + 1}`, start: time, end: time + move });
      intervals.push({ id: 'hold', name: `hold ${reach + 1}`, start: time + move, end: time + move + 0.2 });
      time += move + 0.2;
    }
    intervals.push({ id: 'rest', name: 'rest', start: time, end: 4 });
    const reach = (index: number) => Math.floor((index - 1) / 2);
    return {
      intervals,
      variables: {
        start: point(start), reach: '0.15', angles: angles.map((angle) => `${angle.toFixed(0)}°`).join(', '),
        targets: targets.map(point).join(' '), at_target: '', jitter: '',
      },
      signals: {
        target: fill(400, step, intervals, (interval, index) => interval.id === 'move' || interval.id === 'hold' ? targets[reach(index)] : null),
        go: fill(400, step, intervals, (interval) => [interval.id === 'move' || interval.id === 'hold' ? 1 : 0]),
        fixation: fill(400, step, intervals, (interval) => [interval.id === 'fixate' ? 1 : 0]),
      },
    };
  });
}

function stabilizationTrials(seed: number, count: number): SampledTrial[] {
  const next = random(seed);
  return Array.from({ length: count }, () => {
    const start = [-0.1 + next() * 0.2, -0.1 + next() * 0.2];
    const intervals = [{ id: 'hold', name: 'hold', start: 0, end: 1 }];
    return { intervals, variables: { start: point(start) }, signals: { goal: fill(100, 0.01, intervals, () => start) } };
  });
}

function quantile(sorted: readonly number[], q: number) {
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  return sorted[low] + (sorted[Math.min(low + 1, sorted.length - 1)] - sorted[low]) * (position - low);
}

/** The aggregate of `trials`, as the sampling route returns it for a batch. */
export function aggregate(trials: readonly SampledTrial[], steps: number, step: number, epochs: readonly { id: string; name: string }[],
  variables: readonly string[]): TrialBatch {
  const signals: Record<string, SignalAggregate> = {};
  for (const signal of Object.keys(trials[0].signals)) {
    const presence: number[] = [];
    const percentiles: (Percentiles[] | null)[] = [];
    for (let at = 0; at < steps; at += 1) {
      const present = trials.map((trial) => trial.signals[signal][at]).filter((value): value is readonly number[] => !!value);
      presence.push(present.length / trials.length);
      percentiles.push(present.length ? present[0].map((_, component) => {
        const sorted = present.map((value) => value[component]).sort((a, b) => a - b);
        return [0.05, 0.25, 0.5, 0.75, 0.95].map((q) => quantile(sorted, q)) as unknown as Percentiles;
      }) : null);
    }
    signals[signal] = { presence, percentiles };
  }
  const range = (id: string) => {
    const texts = trials.map((trial) => trial.variables[id]).filter(Boolean);
    if (!texts.length) return '';
    const numbers = texts.map((text) => [...text.matchAll(/-?\d+\.?\d*/g)].map(Number));
    if (numbers[0].length === 0) return [...new Set(texts)].join(', ');
    return numbers[0].map((_, index) => {
      const column = numbers.map((values) => values[index]);
      return `${Math.min(...column).toFixed(2)}…${Math.max(...column).toFixed(2)}`;
    }).join(', ');
  };
  return {
    count: trials.length,
    epochs: epochs.map(({ id, name }) => ({ id, name, share: Array.from({ length: steps }, (_, at) =>
      trials.filter((trial) => trial.intervals.some((interval) => interval.id === id &&
        at >= Math.round(interval.start / step) && at < Math.round(interval.end / step))).length / trials.length) })),
    variables: Object.fromEntries(variables.map((id) => [id, range(id)])),
    signals,
  };
}

const delayedReaches: DocumentSpec = {
  kind: 'document', duration: 0.8, stepSeconds: 0.01, base: 'Delayed reaches (example)',
  variables: [
    { id: 'start', name: 'start', type: 'Point', text: 'vector(U(-0.4, 0.5), U(-0.3, 0.45))' },
    { id: 'target', name: 'target', type: 'Point', text: 'vector(U(-0.4, 0.5), U(-0.3, 0.45))' },
    { id: 'catch', name: 'catch', type: 'Boolean', text: 'bernoulli(0.5)' },
  ],
  epochs: [
    { id: 'hold', name: 'hold', end: 'U(5 st, 14 st)' },
    { id: 'cue', name: 'cue', end: 'U(10 st, 24 st)', changed: 'U(10 st, 19 st)' },
    { id: 'move', name: 'move', end: 'rest' },
  ],
  repeats: [],
  signals: [
    { id: 'target_pos', name: 'target_pos', type: 'Point', cells: [
      { first: 'hold', last: 'hold', text: null }, { first: 'cue', last: 'move', text: 'target' }] },
    { id: 'target_vel', name: 'target_vel', type: 'Velocity', cells: [
      { first: 'hold', last: 'hold', text: null }, { first: 'cue', last: 'move', text: 'vector(0, 0)' }] },
    { id: 'goal', name: 'goal', type: 'Point', cells: [
      { first: 'hold', last: 'cue', text: 'start' }, { first: 'move', last: 'move', text: 'where(catch, start, target)' }] },
    { id: 'fixation', name: 'fixation', type: 'Number', cells: [
      { first: 'hold', last: 'cue', text: '1' }, { first: 'move', last: 'move', text: 'where(catch, 1, 0)' }] },
    { id: 'target_on', name: 'target_on', type: 'Number', cells: [
      { first: 'hold', last: 'hold', text: null }, { first: 'cue', last: 'move', text: '1' }] },
  ],
};

const threeReaches: DocumentSpec = {
  kind: 'document', duration: 4, stepSeconds: 0.01,
  variables: [
    { id: 'start', name: 'start', type: 'Point', text: 'U(disk(0, 0.05))' },
    { id: 'reach', name: 'reach', type: 'Distance', text: '0.1, 0.15, 0.2' },
    { id: 'angles', name: 'angles', type: 'Number', count: 3, text: 'U(0°, 360°)' },
    { id: 'targets', name: 'targets', type: 'Point', count: 3, text: 'start + polar(reach, angles)' },
    { id: 'at_target', name: 'at_target', type: 'Boolean', level: 'step', text: 'held(dist(hand, targets[i]) < 0.02, 0.2 s)' },
    { id: 'jitter', name: 'jitter', type: 'Duration', text: '0.1 * scale', reason: 'No variable is named scale',
      span: { start: 6, end: 11 } },
  ],
  epochs: [
    { id: 'fixate', name: 'fixate', end: '0.3 s' },
    { id: 'move', name: 'move', end: 'at_target ≤ 1 s', least: 0.2 },
    { id: 'hold', name: 'hold', end: '0.2 s' },
    { id: 'rest', name: 'rest', end: 'rest' },
  ],
  repeats: [{ id: 'reaches', first: 'move', last: 'hold', count: '3' }],
  signals: [
    { id: 'target', name: 'target', type: 'Point', cells: [
      { first: 'fixate', last: 'fixate', text: null }, { first: 'move', last: 'hold', text: 'targets[i]' },
      { first: 'rest', last: 'rest', text: null }] },
    { id: 'go', name: 'go', type: 'Number', cells: [
      { first: 'fixate', last: 'fixate', text: '0' }, { first: 'move', last: 'hold', text: '1' }, { first: 'rest', last: 'rest', text: '0' }] },
    { id: 'fixation', name: 'fixation', type: 'Number', cells: [
      { first: 'fixate', last: 'fixate', text: '1' }, { first: 'move', last: 'rest', text: '0' }] },
  ],
};

const stabilization: DocumentSpec = {
  kind: 'document', duration: 1, stepSeconds: 0.01,
  variables: [{ id: 'start', name: 'start', type: 'Point', text: 'vector(U(-0.1, 0.1), U(-0.1, 0.1))' }],
  epochs: [{ id: 'hold', name: 'hold', end: 'rest' }],
  repeats: [],
  signals: [{ id: 'goal', name: 'goal', type: 'Point', cells: [{ first: 'hold', last: 'hold', text: 'start' }] }],
};

export const fixtures: TaskFixture[] = [
  { key: 'delayed-reaches', name: 'Delayed reaches', spec: delayedReaches, sample: delayedTrials },
  { key: 'three-reaches', name: 'Three reaches', spec: threeReaches, sample: threeReachTrials },
  { key: 'stabilization', name: 'Stabilization', spec: stabilization, sample: stabilizationTrials,
    problem: 'Leaves the model input target_pos unfed' },
  { key: 'perturbed-reaches', name: 'Perturbed reaches', python: true, spec: {
    kind: 'python', values: { n_reaches: 8, field: 'curl', field_scale: 12, catch: true },
  } },
];

/** What New offers beside Blank: the shipped examples and the installed Python task types. */
export const templates = [
  { key: 'simple-reaches', name: 'Simple reaches' },
  { key: 'delayed-reaches', name: 'Delayed reaches' },
  { key: 'stabilization', name: 'Stabilization' },
  { key: 'rlrmp3.PerturbedReaches', name: 'Perturbed reaches', python: true },
];
