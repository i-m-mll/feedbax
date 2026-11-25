import type { StudioRegionIssue } from '@/components/activity/StudioRegion';
import { studioName } from '@/components/activity/studioVocabulary';
import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';
import { frameAt } from '@/components/timeline/trialClock';
import type {
  DomainDiagnostic, JsonValue, ModelAddress, StudioSceneGeometryFrame, TaskDocument, TrialMean, TrialRecord, ValueType,
} from '@/generated/studioContracts';
import type { ScenePoint } from './planarProjection';
import type { Scene, SceneElement, SceneGroup, SceneMeanPath, SceneShape, SceneTrack } from './sceneView';

/** Values of a Model's states, by state address, that pose its mechanisms. */
export type StateValues = Readonly<Record<string, JsonValue>>;
export type SceneModelGeometry = StudioSceneGeometryFrame & { document_id: string };

/** Whether a value of `type` has a place in the world: a position, two lengths. */
export function hasPlace(type: ValueType) {
  const dimensions = Object.entries(type.dimensions ?? {}).filter(([, power]) => power);
  return type.base === 'real' && type.shape?.length === 1 && type.shape[0] === 2 &&
    dimensions.length === 1 && dimensions[0][0] === 'length' && dimensions[0][1] === 1;
}

function point(value: JsonValue | undefined): ScenePoint | undefined {
  if (!Array.isArray(value) || value.length !== 2) return undefined;
  const [x, y] = value;
  return typeof x === 'number' && typeof y === 'number' && Number.isFinite(x) && Number.isFinite(y) ? [x, y] : undefined;
}

const at = (values: JsonValue | undefined, step: number) =>
  Array.isArray(values) && values.length ? values[Math.min(step, values.length - 1)] : undefined;

/**
 * The state values `record` recorded at each state frame: each configuration and each world-frame track. Undefined
 * when the record holds no recorded state, as a trial drawn before a run does not.
 */
export function recordedStates(record: TrialRecord): StateValues[] | undefined {
  const histories: [string, JsonValue[]][] = [
    ...Object.entries(record.states ?? {}).map(([address, track]): [string, JsonValue[]] => [address, track.samples]),
    ...Object.entries(record.configurations ?? {}).flatMap(([address, history]): [string, JsonValue[]][] =>
      Array.isArray(history) ? [[address, history]] : []),
  ];
  if (!histories.length) return undefined;
  const frames = Math.max(...histories.map(([, history]) => history.length));
  return Array.from({ length: frames }, (_, frame) => Object.fromEntries(histories.flatMap(([address, history]) => {
    const sample = at(history, frame);
    return sample === undefined ? [] : [[address, sample]];
  })));
}

/**
 * Each world-frame path `record` recorded, with where it is at state frame `frame`; under `trails`, drawn up to that
 * frame. States that are one quantity under two names (a mechanism's effector and its skeleton's position) record the
 * same path, which is drawn once.
 */
export function recordedTracks(record: TrialRecord, frame: number, trails: boolean, picked: boolean): SceneTrack[] {
  const paths = new Map(Object.values(record.states ?? {}).filter((track) => track.frame === 'world')
    .map((track) => [JSON.stringify(track.samples), track.samples]));
  return [...paths.values()].map((samples) => {
    const points = samples.map(point).filter((place) => place !== undefined);
    const drawn = trails ? samples.slice(0, frame + 1).map(point).filter((place) => place !== undefined).length : points.length;
    return { points, drawn, head: point(at(samples, frame)), picked };
  });
}

const planar = (rows: readonly (readonly number[])[]) => rows.every((row) => row.length === 2)
  ? rows.map(([x, y]): ScenePoint => [x, y]) : undefined;

/**
 * Each condition's mean of each world-frame path `mean` holds, at `time` in seconds from Align's epoch; a condition
 * whose common span does not hold `time` has no mark. Paths that are one quantity under two names are drawn once.
 */
export function meanPaths(mean: TrialMean, time: number, trails: boolean): SceneMeanPath[] {
  return mean.conditions.flatMap((condition) => {
    const within = condition.span && time >= condition.span.start - 1e-9 && time <= condition.span.end + 1e-9;
    const frame = frameAt(condition.time, time);
    const paths = new Map(Object.values(condition.states).map((summary) => [JSON.stringify(summary.mean), summary]));
    return [...paths.values()].flatMap((summary): SceneMeanPath[] => {
      const [points, lower, upper] = [planar(summary.mean), planar(summary.lower), planar(summary.upper)];
      if (!points?.length || !lower || !upper) return [];
      return [{
        points, lower, upper, drawn: trails ? within ? frame + 1 : 0 : points.length, ...within && { head: points[frame] },
      }];
    });
  });
}

/**
 * Under Mean, each trial's world-frame paths drawn thin, with no mark of their own: under `trails`, each up to the
 * playhead's aligned `time`, which is that trial's alignment origin plus `time` on its own record.
 */
function alignedTracks(mean: TrialMean, records: readonly TrialRecord[], stateTime: readonly number[], time: number, trails: boolean): SceneTrack[] {
  const origins = new Map(mean.conditions.flatMap((condition) =>
    Object.entries(condition.trials).map(([trial, alignment]) => [trial, alignment.origin] as const)));
  return records.flatMap((record) => {
    const origin = origins.get(record.id);
    if (origin === undefined) return [];
    return recordedTracks(record, frameAt(stateTime, origin + time), trails, false).map(({ head: _head, ...track }) => track);
  });
}

const component = (address: ModelAddress) => address.element.kind === 'component' ? address.element.id : undefined;

/** Each mechanism the geometry holds, with its bodies, joints and muscles, and each diagnostic on the element it names. */
export function mechanismGroups(geometry: SceneModelGeometry): SceneGroup[] {
  const document = geometry.document_id;
  const located = (address: ModelAddress): ElementAddress => ({ document, address });
  const named = new Set<string>();
  const problem = (address: ModelAddress) => {
    const id = component(address);
    if (id === undefined) return undefined;
    named.add(id);
    return geometry.diagnostics.find((item: DomainDiagnostic) => item.node_ids?.[0] === id)?.message;
  };
  const element = (address: ModelAddress, name: string, layer: SceneElement['layer'], shape: SceneShape,
    values: SceneElement['values']): SceneElement => {
    const at = located(address);
    return { key: elementKey(at), address: at, name, layer, shape, values, problem: problem(address) };
  };
  const groups = geometry.mechanisms.map((mechanism): SceneGroup => {
    const address = located(mechanism.address);
    const elements = mechanism.kind === 'point'
      ? [element(mechanism.point, mechanism.name, 'bodies', { kind: 'point', point: mechanism.position },
        [{ label: mechanism.name, point: mechanism.position }])]
      : [
        ...mechanism.links.flatMap((link) => [
          element(link.body, link.body_name, 'bodies', { kind: 'segment', points: [link.start, link.end] }, [
            { label: `${link.body_name}.proximal`, point: link.start },
            { label: `${link.body_name}.distal`, point: link.end },
          ]),
          element(link.joint, link.joint_name, 'joints', { kind: 'point', point: link.pivot },
            [{ label: link.joint_name, point: link.pivot }]),
        ]),
        ...mechanism.muscles.map((muscle) => element(muscle.address, muscle.name, 'muscles', { kind: 'path', points: muscle.points },
          muscle.points.map((place, index) => ({ label: `${muscle.name} ${index + 1}`, point: place })))),
      ];
    return { key: elementKey(address), address, name: mechanism.name, elements, problem: problem(mechanism.address) };
  });
  const unplaced = geometry.diagnostics.find((item) => !item.node_ids?.[0] || !named.has(item.node_ids[0]))?.message;
  return unplaced && groups.length ? [{ ...groups[0], problem: groups[0].problem ?? unplaced }, ...groups.slice(1)] : groups;
}

/** The shapes of each mechanism the geometry holds. */
export const mechanismShapes = (geometry: SceneModelGeometry): SceneShape[] =>
  mechanismGroups(geometry).flatMap((group) => group.elements.flatMap((element) => element.shape ?? []));

/** A task variable's place at `step` of `record`: its value, or its value at that step when it varies within the trial. */
function variablePlace(record: TrialRecord, id: string, step: number): ScenePoint | undefined {
  const value = record.variables[id];
  const fixed = point(value);
  if (fixed) return fixed;
  const defined = record.variable_defined[id];
  if (Array.isArray(defined) && at(defined, step) === false) return undefined;
  return point(at(value, step));
}

/** A task signal's place at `step` of `record`, while it is present there. */
function signalPlace(record: TrialRecord, id: string, step: number): ScenePoint | undefined {
  return at(record.present[id], step) === true ? point(at(record.inputs[id], step)) : undefined;
}

/** The task a Scene draws: the document that holds it, and its definition. */
export interface SceneTask {
  readonly document: string;
  readonly value: TaskDocument;
}

/**
 * The task's variables and signals that have a place, at `step`: in the picked trial, and at each of the trials that
 * may be shown. A variable linked to a Model field is the trial's start; every other place is a goal.
 */
export function taskGroup(task: SceneTask, record: TrialRecord | undefined, possible: readonly TrialRecord[], step: number): SceneGroup | undefined {
  const address: ElementAddress = { document: task.document, element: { kind: 'named', id: 'task' } };
  const starts = new Set(task.value.links?.map((link) => link.variable));
  const element = (kind: 'variable' | 'signal', id: string, name: string,
    place: (record: TrialRecord, id: string, step: number) => ScenePoint | undefined): SceneElement => {
    const located: ElementAddress = { document: task.document, element: { kind: 'named', id: `${kind}:${id}` } };
    const shown = record && place(record, id, step);
    return {
      key: elementKey(located), address: located, name, layer: kind === 'variable' ? 'variables' : 'signals',
      mark: kind === 'variable' && starts.has(id) ? 'start' : 'goal',
      ...shown && { shape: { kind: 'point', point: shown } },
      possible: possible.map((trial) => place(trial, id, step)).filter((spot) => spot !== undefined),
      values: shown ? [{ label: name, point: shown }] : [],
    };
  };
  const elements = [
    ...task.value.variables.filter((variable) => hasPlace(variable.type))
      .map((variable) => element('variable', variable.id, variable.name, variablePlace)),
    ...task.value.signals.filter((signal) => hasPlace(signal.type))
      .map((signal) => element('signal', signal.id, signal.name, signalPlace)),
  ];
  return elements.length ? { key: elementKey(address), address, name: studioName('task'), elements } : undefined;
}

/**
 * What the clock gives the Scene: the picked trial (the one shown, or the one drawn heavier under All), at the state
 * frame the playhead is at; the trials drawn beside it (the samples it may be, or every recorded trial under All); the
 * task; whether paths end at the playhead; and, when the picked trial could not be read, why.
 */
export interface SceneTrial {
  readonly picked: TrialRecord | undefined;
  /**
   * Under Mean, each condition's mean, the playhead's time from Align's epoch, and the recorded trials' state times;
   * `possible` then holds every recorded trial and nothing is picked.
   */
  readonly mean?: { readonly value: TrialMean; readonly time: number; readonly stateTime: readonly number[] };
  readonly possible: readonly TrialRecord[];
  readonly frame: number;
  readonly task: SceneTask | undefined;
  readonly trails: boolean;
  readonly failure?: { readonly kind: 'failed'; readonly issue: StudioRegionIssue; readonly retry: () => void };
}

/**
 * The Scene of what the clock shows: the mechanisms posed by the picked trial, the rest pose beneath, each recorded
 * track (the picked trial's heavier) and the task. Under Mean, the mechanisms at rest alone, drawn faint, each trial's
 * tracks thin, each condition's mean path with its spread, and where the task's places may be.
 */
export function trialScene({ posed, rest, task, picked, possible, frame, trails, mean }: {
  posed: SceneModelGeometry;
  rest: SceneModelGeometry | undefined;
  task: SceneTask | undefined;
  picked: TrialRecord | undefined;
  possible: readonly TrialRecord[];
  /** The state frame at the playhead; the task's values are read at the step it begins. */
  frame: number;
  trails: boolean;
  mean?: SceneTrial['mean'];
}): Scene {
  if (mean) {
    const places = task && taskGroup(task, undefined, possible, 0);
    return {
      groups: places ? [places] : [],
      rest: mechanismShapes(posed),
      tracks: alignedTracks(mean.value, possible, mean.stateTime, mean.time, trails),
      means: meanPaths(mean.value, mean.time, trails),
    };
  }
  const drawn = task && taskGroup(task, picked, possible, frame);
  return {
    groups: [...mechanismGroups(posed), ...drawn ? [drawn] : []],
    rest: rest ? mechanismShapes(rest) : [],
    tracks: [
      ...possible.filter((record) => record.id !== picked?.id).flatMap((record) => recordedTracks(record, frame, trails, false)),
      ...picked ? recordedTracks(picked, frame, trails, true) : [],
    ],
    means: [],
  };
}
