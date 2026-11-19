import type { StudioPlace } from '@/components/activity/studioVocabulary';
import type { JsonValue, StudioProjectFilePatch, StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import { authoredMember, authoredPointerParts, removeAuthoredMember, type AuthoredProject } from './authoredEdits';

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

export type RevealView = 'model' | 'task' | 'objective' | 'training' | 'analysis' | 'document' | 'runs';

/** The view of a step that draws a place. */
export function placeView(place: StudioPlace | null): RevealView {
  switch (place?.part) {
    case 'model': case 'task': case 'objective': case 'training': case 'analysis': return place.part;
    case 'method': return 'training';
    default: return 'runs';
  }
}

/** A place revealed in a step's view, with what the reason that named it says. */
export interface Reveal {
  readonly step: string;
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
