import type { StudioPlace } from '@/components/activity/studioVocabulary';
import { addressKey } from './model/modelGraph';
import { initializesAnchor, startAnchor } from './StudioTrialAuthoring';
import type { JsonValue, StudioProjectFilePatch, StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import {
  authoredMember, authoredPointer, authoredPointerParts, removeAuthoredMember, type AuthoredProject,
} from './authoredEdits';

const authoredGraphVariation = 'authored_graph_variation';

function isOverrideRecord(value: JsonValue | undefined): value is { path: string; op: 'add' | 'replace' | 'remove' } {
  return value != null && typeof value === 'object' && !Array.isArray(value) && typeof value.path === 'string' &&
    (value.op === 'add' || value.op === 'replace' || value.op === 'remove');
}

function scopedVariationWriter(path: readonly string[]): number | undefined {
  if (path[0] === 'deltas' && path[2] === 'patches') return 4;
  if (path[0] !== 'sweep') return;
  if (path[1] === 'axes' && path[3] === 'values' && path[5] === 'patches') return 7;
  if ((path[1] === 'exceptions' || path[1] === 'rows') && path[3] === 'patches') return 5;
}

function canRemoveWithoutReindexing(value: JsonValue | undefined, key: string) {
  return !Array.isArray(value) || Number(key) === value.length - 1;
}

export function removeGraphVariation(
  project: AuthoredProject, problem: StudioProjectWindowMapProblem,
): StudioProjectFilePatch | undefined {
  if (problem.code !== authoredGraphVariation || !problem.step ||
      problem.address !== `steps/${problem.step}.json` || !problem.authored_path ||
      project.steps?.[problem.step]?.kind !== 'evaluate') return;
  const address = { kind: 'step', id: problem.step } as const;
  const path = authoredPointerParts(problem.authored_path);
  const writerDepth = scopedVariationWriter(path);
  const member = (parts: readonly string[]) => authoredMember(project, address, parts);
  const recordPath = path.slice(0, -1);
  let target: string[];
  if (writerDepth !== undefined && path[writerDepth] === 'value') {
    const writer = path.slice(0, writerDepth);
    const suffix = path.slice(writerDepth + 1);
    const storedWriter = member(writer);
    const key = path[0] === 'deltas'
      ? (isOverrideRecord(storedWriter) ? storedWriter.path : undefined)
      : writer.at(-1);
    const writesRecord = /^params\.variation\.[^.]+$/.test(key ?? '');
    const writesPath = /^params\.variation\.[^.]+\.path$/.test(key ?? '');
    if (writesPath && suffix.length === 0 && typeof member(path) === 'string') {
      target = writer;
    } else if (writesRecord && suffix.length === 1 && suffix[0] === 'path' && isOverrideRecord(member(recordPath))) {
      if (!storedWriter || typeof storedWriter !== 'object' || Array.isArray(storedWriter) ||
          storedWriter.op !== 'add' || /^params\.variation\.[0-9]+$/.test(key ?? '')) return;
      target = writer;
    } else if (key === 'params.variation' && suffix.length === 2 && suffix[1] === 'path' &&
        isOverrideRecord(member(recordPath))) {
      if (!canRemoveWithoutReindexing(member(path.slice(0, writerDepth + 1)), suffix[0])) return;
      target = recordPath;
    } else return;
  } else if (path.length === 6 && path.slice(0, 4).join('/') === 'base/inline/params/variation' &&
      path.at(-1) === 'path' && isOverrideRecord(member(recordPath))) {
    if (!canRemoveWithoutReindexing(member(path.slice(0, 4)), path[4])) return;
    target = recordPath;
  } else return;
  const patch = removeAuthoredMember(project, address, target);
  return patch && { address, patches: [patch] };
}

export type RevealView = 'model' | 'task' | 'objective' | 'analysis' | 'document' | 'runs';

/** The view of a step that draws a place. A step's own fields are drawn in the Runs view's inspector. */
export function placeView(place: StudioPlace | null): RevealView {
  switch (place?.part) {
    case 'model': case 'task': case 'objective': case 'analysis': return place.part;
    default: return 'runs';
  }
}

/**
 * Whether a place is a field its view reveals itself, with its own mark, and which: `target` is the `data-field` the
 * view gives that field, absent when the place names the fields as a whole. A step's own fields are in the inspector.
 */
export function placeField(place: StudioPlace | null): { target?: string } | null {
  switch (place?.part) {
    case 'task': return place.initializes ? { target: initializesAnchor(place.initializes) }
      : place.start ? { target: startAnchor(addressKey(place.start.address)) } : null;
    case 'training': return place.setting ? { target: `/training_config/${place.setting}` } : {};
    // The inspector names each top-level setting's row; a nested setting is revealed at the row that holds it.
    case 'method': return { target: place.setting ? authoredPointer(['settings', place.setting.split('.')[0]]) : '/method' };
    case 'step': return place.field ? { target: `/${place.field}` } : null;
    default: return null;
  }
}

/** A place revealed in a step's view, with what the reason that named it says. */
export interface Reveal {
  readonly step: string;
  /** The name of what the reason is about. */
  readonly name: string;
  readonly place: StudioPlace | null;
  readonly view: RevealView;
  readonly reason: string;
  /** The problem that named the place, when one did. */
  readonly problem?: StudioProjectWindowMapProblem;
}

/** The Model member a place names, with the document that holds it. */
export function placeMember(place: StudioPlace | null | undefined) {
  return place?.part === 'model' && place.member ? { document: place.document, address: place.member } : undefined;
}

/** The Analyze element a place names. */
export function placeElement(place: StudioPlace | null | undefined) {
  return place?.part === 'analysis' ? place.element : undefined;
}

/** The field of its analysis element a Problem is about, as a pointer within the element. */
export function placedField(problem: { place?: StudioPlace | null }) {
  return problem.place?.part === 'analysis' ? problem.place.field ?? undefined : undefined;
}
