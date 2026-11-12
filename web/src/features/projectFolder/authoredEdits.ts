import { enablePatches, produceWithPatches, type Patch } from 'immer';
import type {
  AuthoredFileAddress,
  JsonValue,
  ProjectValueEdit,
  StudioProjectSnapshot,
} from '@/generated/studioContracts';

export type AuthoredProject = StudioProjectSnapshot['project'];
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
  }
}

function own(container: Container, key: string | number) {
  return Object.prototype.hasOwnProperty.call(container, key);
}

function parts(pointer: string) {
  if (!/^\/(?:[^~]|~[01])*$/.test(pointer)) throw new Error('An authored edit requires a file-relative JSON pointer.');
  return pointer.slice(1).split('/').map((token) => token.replace(/~1/g, '/').replace(/~0/g, '~'));
}

function pointer(path: Patch['path']) {
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

function applyEdit(entry: JsonObject, edit: ProjectValueEdit, checkOrder: boolean) {
  const path = parts(edit.path);
  let container: Container = entry;
  for (const token of path.slice(0, -1)) {
    const key = keyAt(container, token);
    if (!own(container, key)) throw new Error(`The field at ${edit.path} no longer exists.`);
    const value = Array.isArray(container) ? container[key as number] : container[key];
    container = Array.isArray(value) ? value : object(value);
  }
  const key = keyAt(container, path[path.length - 1]);
  const exists = own(container, key);
  if (edit.op !== 'add' && !exists) throw new Error(`The field at ${edit.path} no longer exists.`);
  const previous = Array.isArray(container) ? container[key as number] : container[key];
  if (checkOrder && edit.op !== 'add' && hasIndexedKeys(previous)) {
    throw new Error('Choose an individual authored value to preserve the existing order.');
  }
  if (Array.isArray(container)) {
    const index = key as number;
    if (index > container.length) throw new Error(`The list index at ${edit.path} is out of range.`);
    if (edit.op === 'add') container.splice(index, 0, structuredClone(edit.value));
    else if (edit.op === 'remove') container.splice(index, 1);
    else container[index] = structuredClone(edit.value);
  } else {
    if (edit.op === 'add' && exists) throw new Error(`The field at ${edit.path} already exists.`);
    if (edit.op === 'remove') {
      // Inverse add appends to an object in the folder codec. Keep removals
      // that need a position behind the order-aware edit contract.
      const keys = Object.keys(container);
      if (checkOrder && keys.length > 1 &&
        (keys[keys.length - 1] !== key || keys.some((name) => /^[0-9]+$/.test(name)))) {
        throw new Error('Removing this authored value requires its position in the ordered file.');
      }
      delete container[key];
    } else container[key] = structuredClone(edit.value);
  }
}

function fileEntry(project: AuthoredProject, address: AuthoredFileAddress) {
  const collection = files(project as unknown as JsonObject, address.kind);
  if (!own(collection, address.id)) throw new Error('The named authored file no longer exists.');
  return object(collection[address.id]);
}

export function sameAuthoredFile(a: AuthoredProject, b: AuthoredProject, address: AuthoredFileAddress) {
  return JSON.stringify(fileEntry(a, address)) === JSON.stringify(fileEntry(b, address));
}

function shouldReapply(before: Container, latest: Container, edit: ProjectValueEdit) {
  const path = parts(edit.path);
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

export function applyAuthoredEdits(
  project: AuthoredProject,
  address: AuthoredFileAddress,
  edits: readonly ProjectValueEdit[],
  checkOrder = false,
  rebaseFrom?: AuthoredProject,
) {
  // These models are authored JSON; use one generic traversal over their file envelope.
  const authored = project as unknown as JsonObject;
  const collection = files(authored, address.kind);
  const original = fileEntry(project, address);
  const before = rebaseFrom && structuredClone(fileEntry(rebaseFrom, address));
  const reapplied: ProjectValueEdit[] = [];
  const [changed, , inverse] = produceWithPatches<JsonObject, JsonObject>(original, (entry) => {
    for (const edit of edits) {
      if (!before || shouldReapply(before, entry, edit)) {
        applyEdit(entry, edit, checkOrder);
        reapplied.push(edit);
      }
      if (before) applyEdit(before, edit, false);
    }
  });
  if (!inverse.length) return { project, inverse: [], edits: [] };
  const nextFiles = { ...collection, [address.id]: changed };
  const updated = address.kind === 'variant'
    ? { ...authored, sweep: { ...object(authored.sweep), rows: nextFiles } }
    : { ...authored, [address.kind === 'document' ? 'documents' : 'steps']: nextFiles };
  return {
    project: updated as unknown as AuthoredProject,
    edits: reapplied,
    inverse: inverse.map((patch): ProjectValueEdit => patch.op === 'remove'
      ? { op: 'remove', path: pointer(patch.path) }
      : { op: patch.op, path: pointer(patch.path), value: patch.value as JsonValue }),
  };
}
