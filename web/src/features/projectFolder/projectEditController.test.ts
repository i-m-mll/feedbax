import { afterEach, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import type { AuthoredFileAddress, ProjectValueEdit, StudioProjectDocument, StudioProjectSnapshot } from '@/generated/studioContracts';
import { addressKey, applyAuthoredEdits } from './authoredEdits';
import { ProjectEditConflictError, ProjectEditController, type ProjectEditTransport } from './projectEditController';

const model: AuthoredFileAddress = { kind: 'document', id: 'model' };
const task: AuthoredFileAddress = { kind: 'document', id: 'task' };
const hex = (number: number) => number.toString(16).padStart(64, '0');
const value = (number: number): ProjectValueEdit => ({ op: 'replace', path: '/payload/value', value: number });
const initial: StudioProjectSnapshot = {
  project_id: 'reach',
  revision: hex(1),
  presentation_revision: hex(2),
  semantic_root: hex(3),
  presentation: null,
  file_revisions: [{ address: model, revision: hex(4) }, { address: task, revision: hex(5) }],
  project: { documents: {
    model: { schema: 'graph', payload: { value: 1 } },
    task: { schema: 'task', payload: { value: 1 } },
  } },
};

function fixture(settleMs = 400) {
  let current = initial;
  let name = 'reach';
  let sequence = 6;
  const document = () => ({ configuration: { project: name, store: 'results' }, snapshot: current });
  const external = (address: AuthoredFileAddress, edits: readonly ProjectValueEdit[]) => {
    const { project } = applyAuthoredEdits(current.project, address, edits);
    current = {
      ...current,
      project,
      revision: hex(sequence++),
      file_revisions: current.file_revisions.map((entry) => addressKey(entry.address) === addressKey(address)
        ? { ...entry, revision: hex(sequence++) } : entry),
    };
    return current;
  };
  const write = vi.fn<ProjectEditTransport['write']>(async (_id, body) => {
    if (body.revision !== current.revision) throw new ApiRequestError('http', '/api/projects/reach', 'Changed', { status: 409 });
    return external(body.address, body.patches);
  });
  const externalRename = (next: string) => {
    name = next;
    current = { ...current, revision: hex(sequence++) };
    return document();
  };
  const read = vi.fn<ProjectEditTransport['read']>(async () => document());
  const rename = vi.fn<ProjectEditTransport['rename']>(async (_id, body) => {
    if (body.revision !== current.revision) throw new ApiRequestError('http', '/api/projects/reach/rename', 'Changed', { status: 409 });
    return externalRename(body.name);
  });
  const create = vi.fn<ProjectEditTransport['create']>();
  const editor = new ProjectEditController(document(), { read, write, rename, create }, settleMs);
  return { editor, write, read, rename, create, external, externalRename, current: () => current, restore: (snapshot: StudioProjectSnapshot) => { current = snapshot; } };
}

afterEach(() => vi.useRealTimers());

it('does not attribute a preexisting identical file to a failed create', async () => {
  const { editor, create } = fixture();
  create.mockRejectedValueOnce(new ApiRequestError('network', '/api/projects/reach/files', 'Offline'));
  await expect(editor.createFile({
    kind: 'document', id: 'model', document: { schema: 'graph', payload: { value: 1 } },
  })).rejects.toMatchObject({ kind: 'network' });
  expect(create).toHaveBeenCalledTimes(1);
  expect(editor.getSnapshot().saved.revision).toBe(initial.revision);
  editor.dispose();
});

it('advances a delayed metadata observation without discarding an edit typed during Open', async () => {
  const { editor, read, write, externalRename } = fixture();
  const renamed = externalRename('outside-name');
  let finish!: (document: StudioProjectDocument) => void;
  read.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  const flushed = editor.flush();
  await vi.waitFor(() => expect(read).toHaveBeenCalledTimes(1));
  editor.edit(model, 'Model value', [value(2)]);
  finish(renamed);

  await flushed;
  expect(write.mock.calls[0][1].revision).toBe(renamed.snapshot.revision);
  expect(editor.getSnapshot()).toMatchObject({
    configuration: { project: 'outside-name' }, outside: null, pending: 0, undoName: 'Model value',
  });
  editor.dispose();
});

it('reports a Rename conflict after reading the current name and never retries the command', async () => {
  const { editor, rename, externalRename } = fixture();
  rename.mockImplementationOnce(async () => {
    externalRename('other-name');
    throw new ApiRequestError('http', '/api/projects/reach/rename', 'Changed', { status: 409 });
  });
  await expect(editor.rename('my-name')).rejects.toMatchObject({ status: 409 });
  expect(editor.getSnapshot().configuration.project).toBe('other-name');
  expect(rename).toHaveBeenCalledTimes(1);
  editor.dispose();
});

it('coalesces settled edits per file and preserves one undo entry per gesture', async () => {
  vi.useFakeTimers();
  const { editor, write, current } = fixture();
  editor.edit(model, 'First value', [value(2)]);
  await vi.advanceTimersByTimeAsync(200);
  editor.edit(model, 'Second value', [value(3)]);
  await vi.advanceTimersByTimeAsync(399);
  expect(write).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1);
  expect(write).toHaveBeenCalledTimes(1);
  expect(write.mock.calls[0][1].patches).toEqual([value(2), value(3)]);
  expect(editor.getSnapshot().saved.revision).toBe(current().revision);
  editor.undo();
  await editor.flush();
  expect(write.mock.calls[1][1].patches).toEqual([value(2)]);
  expect(editor.getSnapshot().undoName).toBe('First value');
  expect(editor.getSnapshot().redoName).toBe('Second value');
  editor.redo();
  expect(await editor.flush()).toBe(current().revision);
  expect(write.mock.calls[2][1].patches).toEqual([value(3)]);
  editor.dispose();
});

it('retains history for an unchanged file and offers both versions for dirty outside changes', async () => {
  const { editor, external, write } = fixture();
  editor.edit(model, 'Model value', [value(2)]);
  editor.edit(task, 'Task value', [value(3)]);
  await editor.flush();
  editor.observe(external(task, [value(4)]));
  await editor.flush();
  expect(editor.getSnapshot().undoName).toBe('Model value');
  editor.edit(task, 'New local value', [value(6)]);
  const outside = external(task, [value(5)]);
  editor.observe(outside);
  await expect(editor.flush()).rejects.toBeInstanceOf(ProjectEditConflictError);
  expect(editor.getSnapshot().outside?.revision).toBe(outside.revision);
  expect(editor.getSnapshot().pending).toBe(1);
  expect(write).toHaveBeenCalledTimes(2);
  await editor.keep('disk');
  expect(editor.getSnapshot().draft).toEqual(outside.project);
  expect(editor.getSnapshot().undoName).toBe('Model value');
  editor.dispose();
});

it('rebases kept edits over the latest revision and gives undo the outside value', async () => {
  const { editor, external, write } = fixture();
  editor.edit(model, 'Keep value', [value(2)]);
  const outside = external(model, [value(3)]);
  await expect(editor.flush()).rejects.toMatchObject({ status: 409 });
  expect(editor.getSnapshot().outside?.revision).toBe(outside.revision);
  await editor.keep('edits');
  expect(write.mock.calls[1][1]).toEqual({ address: model, revision: outside.revision, patches: [value(2)] });
  editor.undo();
  await editor.flush();
  expect(write.mock.calls[2][1].patches).toEqual([value(3)]);
  editor.dispose();
});

it('retries a lost connection after a fresh read and flushes edits arriving during a write', async () => {
  const { editor, current, write, external } = fixture();
  write.mockRejectedValueOnce(new ApiRequestError('network', '/api/projects/reach', 'Offline'));
  editor.edit(model, 'Offline edit', [value(2)]);
  await expect(editor.flush()).rejects.toMatchObject({ kind: 'network' });
  editor.connected();
  await vi.waitFor(() => expect(editor.getSnapshot().pending).toBe(0));

  let complete!: () => void;
  const waiting = new Promise<void>((resolve) => { complete = resolve; });
  write.mockImplementationOnce(async (_id, body) => {
    await waiting;
    return external(body.address, body.patches);
  });
  editor.edit(model, 'Waiting edit', [value(3)]);
  const flushing = editor.flush();
  await vi.waitFor(() => expect(editor.getSnapshot().saving).toBe(true));
  editor.edit(task, 'Queued edit', [value(4)]);
  complete();
  expect(await flushing).toBe(current().revision);
  expect(write.mock.calls.map((call) => call[1].address.id)).toEqual(['model', 'model', 'model', 'task']);
  expect(editor.getSnapshot().pending).toBe(0);
  editor.dispose();
});

it('gets a current saved revision for a clean flush before the next folder notice', async () => {
  const { editor, external, read, write } = fixture();
  const latest = external(model, [value(8)]);
  expect(await editor.flush()).toBe(latest.revision);
  expect(editor.getSnapshot().draft).toEqual(latest.project);
  expect(write).not.toHaveBeenCalled();
  read.mockRejectedValueOnce(new ApiRequestError('network', '/api/projects/reach', 'Offline'));
  await expect(editor.flush()).rejects.toMatchObject({ kind: 'network' });
  editor.dispose();
});

it('keeps an observed postimage after a lost save response without adding a second write or undo entry', async () => {
  const { editor, external, write, current } = fixture();
  write.mockImplementationOnce(async (_id, body) => {
    external(body.address, body.patches);
    throw new ApiRequestError('network', '/api/projects/reach', 'Response lost');
  });
  editor.edit(model, 'Add a field and update it', [
    { op: 'add', path: '/payload/added', value: { field: 1 } },
    { op: 'replace', path: '/payload/added/field', value: 2 },
  ]);
  await expect(editor.flush()).rejects.toMatchObject({ kind: 'network' });
  expect(write.mock.calls[0][1].patches[0]).toMatchObject({ value: { field: 1 } });
  editor.connected();
  await vi.waitFor(() => expect(editor.getSnapshot().outside?.revision).toBe(current().revision));
  await editor.keep('edits');
  expect(editor.getSnapshot()).toMatchObject({ draft: current().project, outside: null, pending: 0, undoName: null });
  expect(write).toHaveBeenCalledTimes(1);
  editor.dispose();
});

it('leaves a conflict open when an outside insertion moves an edited list element', async () => {
  const { editor, external, write, current } = fixture();
  editor.observe(external(model, [{ op: 'add', path: '/payload/items', value: ['a', 'b'] }]));
  await editor.flush();
  editor.edit(model, 'Change b', [{ op: 'replace', path: '/payload/items/1', value: 'B' }]);
  const outside = external(model, [{ op: 'add', path: '/payload/items/0', value: 'new' }]);
  await expect(editor.flush()).rejects.toMatchObject({ status: 409 });
  await expect(editor.keep('edits')).rejects.toThrow('changed outside Studio');
  expect(editor.getSnapshot()).toMatchObject({ outside, pending: 1 });
  expect(write).toHaveBeenCalledTimes(1);
  expect(current()).toBe(outside);
  editor.dispose();
});

it('resumes autosave when an outside revision returns to the saved bytes', async () => {
  vi.useFakeTimers();
  const { editor, external, restore, write } = fixture();
  editor.edit(model, 'Local value', [value(2)]);
  editor.observe(external(model, [value(3)]));
  await vi.advanceTimersByTimeAsync(0);
  expect(editor.getSnapshot().outside).not.toBeNull();
  restore(initial);
  editor.observe(initial);
  await vi.advanceTimersByTimeAsync(400);
  expect(editor.getSnapshot()).toMatchObject({ outside: null, pending: 0 });
  expect(write).toHaveBeenCalledTimes(1);
  editor.dispose();
});
