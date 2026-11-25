import { current, enablePatches, isDraft, produceWithPatches, type Patch } from 'immer';
import type {
  AuthoredFileAddress,
  JsonValue,
  PresentationDocument,
  ProjectValueEdit,
  StudioProjectSnapshot,
} from '@/generated/studioContracts';

export type AuthoredProject = StudioProjectSnapshot['project'];
export type ProjectEditContent = AuthoredProject & {
  readonly presentation?: PresentationDocument | null;
  readonly settings?: StudioProjectSnapshot['settings'];
};
export const presentationAddress = { kind: 'presentation', id: 'presentation' } as const;
export const declarationAddress = { kind: 'declaration', id: 'declaration' } as const;
type JsonObject = { [key: string]: JsonValue };
type Container = JsonObject | JsonValue[];

enablePatches();

export function addressKey({ kind, id }: AuthoredFileAddress) {
  return `${kind}/${id}`;
}

function object(value: JsonValue | undefined): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('The authored address no longer exists.');
  }
  return value;
}

function files(project: JsonObject, kind: AuthoredFileAddress['kind']): JsonObject {
  switch (kind) {
    case 'document': return object(project.documents);
    case 'step': return object(project.steps);
    case 'variant': return object(object(project.sweep).rows);
    case 'presentation': return { [presentationAddress.id]: project.presentation ?? {} };
    case 'declaration': return { [declarationAddress.id]: { settings: project.settings ?? {} } };
  }
}

function own(container: Container, key: string | number) {
  return Object.prototype.hasOwnProperty.call(container, key);
}

export function authoredPointerParts(pointer: string) {
  if (!/^\/(?:[^~]|~[01])*$/.test(pointer)) throw new Error('An authored edit requires a file-relative JSON pointer.');
  return pointer.slice(1).split('/').map((token) => token.replace(/~1/g, '/').replace(/~0/g, '~'));
}

export function authoredPointer(path: Readonly<Patch['path']>) {
  return '/' + path.map((token) => String(token).replace(/~/g, '~0').replace(/\//g, '~1')).join('/');
}

function keyAt(container: Container, token: string) {
  if (token === '__proto__') throw new Error('An authored edit cannot change an object prototype.');
  if (!Array.isArray(container)) return token;
  if (token === '-') return container.length;
  if (!/^(0|[1-9][0-9]*)$/.test(token)) throw new Error('An authored list edit requires an index.');
  return Number(token);
}

function hasIndexedKeys(value: JsonValue | undefined): boolean {
  if (!value || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some(hasIndexedKeys);
  return Object.entries(value).some(([key, child]) => /^[0-9]+$/.test(key) || hasIndexedKeys(child));
}

/** The members after a removed object member, in order: an Undo that adds it back adds them again after it. */
interface Position {
  readonly path: string;
  readonly following: readonly (readonly [string, JsonValue])[];
}

function snapshot(value: JsonValue): JsonValue {
  return structuredClone(isDraft(value) ? current(value) : value);
}

function applyEdit(entry: JsonObject, edit: ProjectValueEdit, positions?: Position[]) {
  const path = authoredPointerParts(edit.path);
  let container: Container = entry;
  for (const token of path.slice(0, -1)) {
    const key = keyAt(container, token);
    if (!own(container, key)) throw new Error(`The field at ${edit.path} no longer exists.`);
    const value: JsonValue | undefined = Array.isArray(container) ? container[key as number] : container[key];
    container = Array.isArray(value) ? value : object(value);
  }
  const key = keyAt(container, path[path.length - 1]);
  const exists = own(container, key);
  if (edit.op !== 'add' && !exists) throw new Error(`The field at ${edit.path} no longer exists.`);
  if (Array.isArray(container)) {
    const index = key as number;
    if (index > container.length) throw new Error(`The list index at ${edit.path} is out of range.`);
    if (edit.op === 'add') container.splice(index, 0, structuredClone(edit.value));
    else if (edit.op === 'remove') container.splice(index, 1);
    else container[index] = structuredClone(edit.value);
  } else {
    if (edit.op === 'add' && exists) throw new Error(`The field at ${edit.path} already exists.`);
    if (edit.op === 'remove') {
      const keys = Object.keys(container);
      const following = keys.slice(keys.indexOf(key as string) + 1);
      if (positions && following.length) {
        const parent = container;
        positions.push({
          path: authoredPointer(path),
          following: following.map((name) => [authoredPointer([...path.slice(0, -1), name]), snapshot(parent[name])]),
        });
      }
      delete container[key];
    } else container[key] = structuredClone(edit.value);
  }
}

function plainObject(value: JsonValue | undefined): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * The edits of individual values that replace `before` at `path` with `after`. A container with indexed keys is
 * edited member by member: JSON carries no order for its keys, so writing it whole would reorder the file.
 */
function addressedReplace(path: readonly string[], before: JsonValue | undefined, after: JsonValue): ProjectValueEdit[] {
  if (!plainObject(before) || !plainObject(after) || !hasIndexedKeys(before)) {
    return [{ op: 'replace', path: authoredPointer(path), value: after }];
  }
  return [
    ...Object.keys(before).filter((key) => !own(after, key)).map((key): ProjectValueEdit =>
      ({ op: 'remove', path: authoredPointer([...path, key]) })),
    ...Object.entries(after).flatMap(([key, value]): ProjectValueEdit[] => !own(before, key)
      ? [{ op: 'add', path: authoredPointer([...path, key]), value }]
      : sameValue(before[key], value) ? [] : addressedReplace([...path, key], before[key], value)),
  ];
}

function addressed(entry: JsonObject, edit: ProjectValueEdit): ProjectValueEdit[] {
  if (edit.op !== 'replace') return [edit];
  const path = authoredPointerParts(edit.path);
  return addressedReplace(path, authoredMemberOf(entry, path), edit.value);
}

function fileEntry(project: ProjectEditContent, address: AuthoredFileAddress) {
  const collection = files(project as unknown as JsonObject, address.kind);
  if (!own(collection, address.id)) throw new Error('The named authored file no longer exists.');
  return object(collection[address.id]);
}

export function authoredMember(project: AuthoredProject, address: AuthoredFileAddress, path: readonly string[]) {
  return authoredMemberOf(fileEntry(project, address), path);
}

function authoredMemberOf(entry: JsonObject, path: readonly string[]) {
  let parent: JsonValue | undefined = entry;
  for (const token of path) {
    if (parent == null || typeof parent !== 'object' || token === '__proto__' ||
        (Array.isArray(parent) && !/^(0|[1-9][0-9]*)$/.test(token))) return;
    const key = keyAt(parent, token);
    if (!own(parent, key)) return;
    parent = Array.isArray(parent) ? parent[key as number] : parent[key];
  }
  return parent;
}

export function removeAuthoredMember(
  project: AuthoredProject, address: AuthoredFileAddress, path: readonly string[],
): ProjectValueEdit | undefined {
  const parent = authoredMember(project, address, path.slice(0, -1));
  const name = path.at(-1);
  if (!name || parent == null || typeof parent !== 'object' || !own(parent, keyAt(parent, name))) return;
  return { op: 'remove', path: authoredPointer(path) };
}

export function sameAuthoredFile(a: ProjectEditContent, b: ProjectEditContent, address: AuthoredFileAddress) {
  const left = Object.entries(fileEntry(a, address));
  const right = fileEntry(b, address);
  return left.length === Object.keys(right).length && left.every(([key, value]) =>
    own(right, key) && JSON.stringify(value) === JSON.stringify(right[key]));
}

function sameValue(a: JsonValue | undefined, b: JsonValue | undefined) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function objectOrMissing(value: JsonValue | undefined): value is JsonObject | undefined {
  return value === undefined || (value !== null && typeof value === 'object' && !Array.isArray(value));
}

function mergePresentationValue(
  before: JsonValue | undefined, after: JsonValue | undefined, current: JsonValue | undefined,
): JsonValue | undefined {
  if (sameValue(before, after)) return current;
  if (sameValue(before, current)) return after;
  if (!objectOrMissing(before) || !objectOrMissing(after) || !objectOrMissing(current)) {
    throw new Error('A placement changed outside this edit.');
  }
  const keys = new Set([...Object.keys(current ?? {}), ...Object.keys(after ?? {}), ...Object.keys(before ?? {})]);
  const entries = Array.from(keys).flatMap((key): [string, JsonValue][] => {
    const child = mergePresentationValue(
      before && own(before, key) ? before[key] : undefined,
      after && own(after, key) ? after[key] : undefined,
      current && own(current, key) ? current[key] : undefined,
    );
    return child === undefined ? [] : [[key, child]];
  });
  return entries.length || after !== undefined ? Object.fromEntries(entries) : undefined;
}

export function presentationHistoryEdits(
  before: PresentationDocument, after: PresentationDocument, current: PresentationDocument,
): ProjectValueEdit[] {
  const source = current as JsonObject;
  const merged = object(mergePresentationValue(before as JsonObject, after as JsonObject, source));
  return [...new Set([...Object.keys(source), ...Object.keys(merged)])].flatMap((key): ProjectValueEdit[] => {
    if (own(source, key) && own(merged, key) && sameValue(source[key], merged[key])) return [];
    const path = authoredPointer([key]);
    return own(merged, key)
      ? [{ op: own(source, key) ? 'replace' : 'add', path, value: merged[key] }]
      : [{ op: 'remove', path }];
  });
}

function shouldReapply(before: Container, latest: Container, edit: ProjectValueEdit) {
  const path = authoredPointerParts(edit.path);
  for (const [index, token] of path.entries()) {
    if (Array.isArray(before) || Array.isArray(latest)) {
      // An outside edit can move the intended element to a different index.
      if (JSON.stringify(before) !== JSON.stringify(latest)) {
        throw new Error(`A list on the path to ${edit.path} changed outside Studio.`);
      }
      return true;
    }
    const key = keyAt(latest, token);
    const exists = own(latest, key);
    if (index === path.length - 1) {
      if (edit.op === 'remove' && !exists) return false;
      if (edit.op === 'add' && exists && JSON.stringify(latest[key]) === JSON.stringify(edit.value)) return false;
      if (edit.op === 'replace' && before[key] !== null && typeof before[key] === 'object' &&
          JSON.stringify(before[key]) !== JSON.stringify(latest[key])) {
        if (JSON.stringify(latest[key]) === JSON.stringify(edit.value)) return false;
        throw new Error(`The authored structure at ${edit.path} changed outside Studio.`);
      }
      return true;
    }
    if (!exists || !own(before, key)) throw new Error(`The field at ${edit.path} no longer exists.`);
    const beforeChild = before[key];
    const latestChild = latest[key];
    before = Array.isArray(beforeChild) ? beforeChild : object(beforeChild);
    latest = Array.isArray(latestChild) ? latestChild : object(latestChild);
  }
  return true;
}

/**
 * Applies `edits` to one authored file. The edits it returns are those it applied, as edits of individual values; its
 * inverse restores each removed object member at its position, by adding the members after it again after it.
 */
export function applyAuthoredEdits(
  project: ProjectEditContent,
  address: AuthoredFileAddress,
  edits: readonly ProjectValueEdit[],
  rebaseFrom?: ProjectEditContent,
) {
  // These models are authored JSON; use one generic traversal over their file envelope.
  const authored = project as unknown as JsonObject;
  const collection = files(authored, address.kind);
  const original = fileEntry(project, address);
  const before = rebaseFrom && structuredClone(fileEntry(rebaseFrom, address));
  const reapplied: ProjectValueEdit[] = [];
  const positions: Position[] = [];
  const [changed, , inverse] = produceWithPatches<JsonObject, JsonObject>(original, (entry) => {
    for (const edit of edits) {
      if (!before || shouldReapply(before, entry, edit)) {
        for (const part of addressed(entry, edit)) {
          applyEdit(entry, part, positions);
          reapplied.push(part);
        }
      }
      if (before) applyEdit(before, edit);
    }
  });
  if (!inverse.length) return { project, inverse: [], edits: [] };
  const nextFiles = { ...collection, [address.id]: changed };
  const updated = address.kind === 'declaration'
    ? { ...authored, settings: changed.settings }
    : address.kind === 'presentation'
    ? { ...authored, presentation: changed }
    : address.kind === 'variant'
      ? { ...authored, sweep: { ...object(authored.sweep), rows: nextFiles } }
      : { ...authored, [address.kind === 'document' ? 'documents' : 'steps']: nextFiles };
  const restored = new Map(positions.map((position) => [position.path, position.following]));
  return {
    project: updated as unknown as ProjectEditContent,
    edits: reapplied,
    inverse: inverse.flatMap((patch): ProjectValueEdit[] => {
      const path = authoredPointer(patch.path);
      if (patch.op === 'remove') return [{ op: 'remove', path }];
      const undone: ProjectValueEdit = { op: patch.op, path, value: patch.value as JsonValue };
      const following = patch.op === 'add' ? restored.get(path) : undefined;
      return following ? [undone, ...following.flatMap(([member, value]): ProjectValueEdit[] =>
        [{ op: 'remove', path: member }, { op: 'add', path: member, value }])] : [undone];
    }),
  };
}

/**
 * The document a model-producing step reads in `role`, by the plan's document ids: a Train's trial documents, or a
 * Synthesize step's payload fields.
 */
export function trialDocumentId(ids: Readonly<Record<string, string>> | undefined, role: 'model' | 'task' | 'objective') {
  return ids?.[`payload.trial_documents.${role}`] ?? (role === 'task' ? ids?.['payload.trial_documents.python_task'] : undefined)
    ?? ids?.[`payload.${role}`];
}

export function modelDocumentId(ids?: Readonly<Record<string, string>>) {
  return trialDocumentId(ids, 'model');
}

/**
 * The inline definition of a Train or Evaluate step whose fields are its own: a Train that reads no input, or an
 * Evaluate, with no deltas; with `includeRowEdits`, also when its sweep has rows or exceptions.
 */
export function ownDefinition(project: AuthoredProject, stepId: string, includeRowEdits = false) {
  const step = project.steps?.[stepId];
  if (step?.kind !== 'train' && step?.kind !== 'evaluate' || !('inline' in step.base) || step.deltas?.length ||
      step.kind === 'train' && Object.keys(step.inputs ?? {}).length ||
      !includeRowEdits && (Object.keys(step.sweep?.rows ?? {}).length || Object.keys(step.sweep?.exceptions ?? {}).length)) return null;
  return step.base.inline;
}
