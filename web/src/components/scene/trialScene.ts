import type { StudioRegionIssue } from '@/components/activity/StudioRegion';
import { studioName } from '@/components/activity/studioVocabulary';
import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';
import type {
  DomainDiagnostic, JsonValue, ModelAddress, StudioSceneGeometryResponse, TaskDocument, TrialRecord, ValueType,
} from '@/generated/studioContracts';
import type { TrialShown } from '@/features/projectFolder/task/trialSample';
import type { ScenePoint } from './planarProjection';
import type { Scene, SceneElement, SceneGroup, SceneShape, SceneTrack } from './sceneView';

/** Values of a Model's states, by state address, that pose its mechanisms. */
export type StateValues = Readonly<Record<string, JsonValue>>;

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
 * The state values `record` recorded at state frame `frame`: each configuration and each world-frame track. Undefined
 * when the record holds no recorded state, as a trial drawn before a run does not.
 */
export function recordedState(record: TrialRecord, frame: number): StateValues | undefined {
  const configurations = Object.entries(record.configurations ?? {});
  const tracks = Object.entries(record.states ?? {});
  if (!configurations.length && !tracks.length) return undefined;
  return Object.fromEntries([
    ...tracks.flatMap(([address, track]) => {
      const sample = at(track.samples, frame);
      return sample === undefined ? [] : [[address, sample]];
    }),
    ...configurations.flatMap(([address, history]) => {
      const sample = at(history, frame);
      return sample === undefined ? [] : [[address, sample]];
    }),
  ]);
}

/**
 * Each world-frame path `record` recorded, with where it is at state frame `frame`. States that are one quantity under
 * two names (a mechanism's effector and its skeleton's position) record the same path, which is drawn once.
 */
export function recordedTracks(record: TrialRecord | undefined, frame: number): SceneTrack[] {
  const paths = new Map(Object.values(record?.states ?? {}).filter((track) => track.frame === 'world')
    .map((track) => [JSON.stringify(track.samples), track.samples]));
  return [...paths.values()].map((samples) => ({
    points: samples.map(point).filter((place) => place !== undefined),
    head: point(at(samples, frame)),
  }));
}

const component = (address: ModelAddress) => address.element.kind === 'component' ? address.element.id : undefined;

/** Each mechanism the geometry holds, with its bodies, joints and muscles, and each diagnostic on the element it names. */
export function mechanismGroups(geometry: StudioSceneGeometryResponse): SceneGroup[] {
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
export const mechanismShapes = (geometry: StudioSceneGeometryResponse): SceneShape[] =>
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
 * The task's variables and signals that have a place, at `step`: solid where `shown` is one trial, hollow at each
 * of the trials that may be shown.
 */
export function taskGroup(task: SceneTask, shown: TrialShown | undefined, possible: readonly TrialRecord[], step: number): SceneGroup | undefined {
  const record = shown?.kind === 'one' ? shown.record : undefined;
  const address: ElementAddress = { document: task.document, element: { kind: 'named', id: 'task' } };
  const element = (kind: 'variable' | 'signal', id: string, name: string,
    place: (record: TrialRecord, id: string, step: number) => ScenePoint | undefined): SceneElement => {
    const located: ElementAddress = { document: task.document, element: { kind: 'named', id: `${kind}:${id}` } };
    const solid = record && place(record, id, step);
    return {
      key: elementKey(located), address: located, name, layer: kind === 'variable' ? 'variables' : 'signals',
      ...solid && { shape: { kind: 'point', point: solid } },
      possible: possible.map((trial) => place(trial, id, step)).filter((spot) => spot !== undefined),
      values: solid ? [{ label: name, point: solid }] : [],
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
 * What the clock gives the Scene: the shown trial, at the state frame the playhead is at; the trials that may be shown
 * beside it, before a run; the task; and, when the shown trial could not be read, why.
 */
export interface SceneTrial {
  readonly shown: TrialShown | undefined;
  readonly possible: readonly TrialRecord[];
  readonly frame: number;
  readonly task: SceneTask | undefined;
  readonly failure?: { readonly kind: 'failed'; readonly issue: StudioRegionIssue; readonly retry: () => void };
}

/** The Scene of what the clock shows: the posed mechanisms, the rest pose beneath, the recorded tracks and the task. */
export function trialScene({ posed, rest, task, shown, possible, frame }: {
  posed: StudioSceneGeometryResponse;
  rest: StudioSceneGeometryResponse | undefined;
  task: SceneTask | undefined;
  shown: TrialShown | undefined;
  possible: readonly TrialRecord[];
  /** The state frame at the playhead; the task's values are read at the step it begins. */
  frame: number;
}): Scene {
  const drawn = task && taskGroup(task, shown, possible, frame);
  return {
    groups: [...mechanismGroups(posed), ...drawn ? [drawn] : []],
    rest: rest ? mechanismShapes(rest) : [],
    tracks: recordedTracks(shown?.kind === 'one' ? shown.record : undefined, frame),
  };
}
