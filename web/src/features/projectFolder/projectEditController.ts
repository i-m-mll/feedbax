import { ApiRequestError, isHttpConflict } from '@/api/request';
import type {
  AuthoredFileAddress,
  ProjectValueEdit,
  StudioProjectCreateFile,
  StudioProjectDocument,
  StudioProjectPatch,
  StudioProjectRename,
  StudioProjectRevision,
  StudioProjectSnapshot,
} from '@/generated/studioContracts';
import { addressKey, applyAuthoredEdits, sameAuthoredFile, type AuthoredProject } from './authoredEdits';

export interface ProjectEditTransport {
  read: (id: string, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  write: (id: string, body: StudioProjectPatch, signal?: AbortSignal) => Promise<StudioProjectSnapshot>;
  rename: (id: string, body: StudioProjectRename, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  create: (id: string, body: StudioProjectCreateFile, signal?: AbortSignal) => Promise<StudioProjectDocument>;
}

type NamedFileCreation = StudioProjectCreateFile['file'];

interface Gesture {
  readonly address: AuthoredFileAddress;
  readonly name: string;
  readonly edits: readonly ProjectValueEdit[];
  readonly inverse: readonly ProjectValueEdit[];
  readonly validate?: ProjectEditValidation;
}

export type ProjectEditValidation = (before: AuthoredProject, after: AuthoredProject) => void;

interface History {
  readonly undo: readonly Gesture[];
  readonly redo: readonly Gesture[];
}

interface PendingEdit {
  readonly base: AuthoredProject;
  readonly gesture: Gesture;
  readonly direction: 'edit' | 'undo' | 'redo';
  readonly edits: readonly ProjectValueEdit[];
  readonly historyAfter: History;
}

export interface ProjectEditSnapshot {
  readonly saved: StudioProjectSnapshot;
  readonly configuration: StudioProjectDocument['configuration'];
  readonly draft: AuthoredProject;
  readonly pending: number;
  readonly saving: boolean;
  readonly error: unknown;
  readonly outside: StudioProjectSnapshot | null;
  readonly undoName: string | null;
  readonly redoName: string | null;
}

export class ProjectEditConflictError extends Error {
  constructor() {
    super('The project changed before the pending edits could be saved.');
  }
}

const emptyHistory: History = { undo: [], redo: [] };

function moveHistory(history: History, gesture: Gesture, direction: PendingEdit['direction']): History {
  switch (direction) {
    case 'edit': return { undo: [...history.undo, gesture], redo: [] };
    case 'undo': return { undo: history.undo.filter((item) => item !== gesture), redo: [...history.redo, gesture] };
    case 'redo': return { undo: [...history.undo, gesture], redo: history.redo.filter((item) => item !== gesture) };
  }
}

function unchangedFiles(previous: StudioProjectSnapshot, next: StudioProjectSnapshot) {
  const current = new Map(next.file_revisions.map(({ address, revision }) => [addressKey(address), revision]));
  return new Set(previous.file_revisions.flatMap(({ address, revision }) => {
    const key = addressKey(address);
    return current.get(key) === revision ? [key] : [];
  }));
}

function withoutChangedFiles(history: History, unchanged: ReadonlySet<string>): History {
  const usable = ({ address }: Gesture) => unchanged.has(addressKey(address));
  return { undo: history.undo.filter(usable), redo: history.redo.filter(usable) };
}

function versionsMatch(a: StudioProjectRevision, b: StudioProjectRevision) {
  return a.revision === b.revision && a.presentation_revision === b.presentation_revision;
}

function canRetryOnReconnect(error: unknown) {
  return error instanceof ApiRequestError && (error.kind === 'network' || (error.status ?? 0) >= 500);
}

function containsCreatedFile(project: AuthoredProject, file: NamedFileCreation) {
  const collection = file.kind === 'document' ? project.documents : project.steps;
  if (!collection || !Object.prototype.hasOwnProperty.call(collection, file.id)) return false;
  const actual = new Map(Object.entries(collection[file.id]));
  const intended = Object.entries(file.kind === 'document' ? file.document : file.step);
  return actual.size === intended.length && intended.every(([key, value]) =>
    actual.has(key) && JSON.stringify(actual.get(key)) === JSON.stringify(value),
  );
}

function isAcknowledgedAddition(
  before: StudioProjectSnapshot, after: StudioProjectSnapshot,
  unchanged: ReadonlySet<string>, file: NamedFileCreation,
) {
  const key = addressKey(file);
  return unchanged.size === before.file_revisions.length &&
    after.file_revisions.length === unchanged.size + 1 &&
    !unchanged.has(key) &&
    after.file_revisions.some(({ address }) => addressKey(address) === key) &&
    JSON.stringify(before.project.sweep) === JSON.stringify(after.project.sweep) &&
    containsCreatedFile(after.project, file);
}

export class ProjectEditController {
  private readonly abort = new AbortController();
  private readonly listeners = new Set<() => void>();
  private readonly queued: PendingEdit[] = [];
  private document: StudioProjectDocument;
  private draft: AuthoredProject;
  private history: History = emptyHistory;
  private savedHistory: History = emptyHistory;
  private outside: StudioProjectDocument | null = null;
  private error: unknown = null;
  private saving = false;
  private snapshot: ProjectEditSnapshot;
  private settle: ReturnType<typeof setTimeout> | undefined;
  private serial: Promise<unknown> = Promise.resolve();
  private observation: StudioProjectRevision | undefined;
  private observationScheduled = false;

  constructor(
    initial: StudioProjectDocument,
    private readonly transport: ProjectEditTransport,
    private readonly settleMs = 400,
  ) {
    this.document = initial;
    this.draft = initial.snapshot.project;
    this.snapshot = this.capture();
  }

  private get saved() { return this.document.snapshot; }

  getSnapshot = () => this.snapshot;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  edit(address: AuthoredFileAddress, name: string, edits: readonly ProjectValueEdit[], validate?: ProjectEditValidation) {
    this.requireEditable();
    if (!edits.length) return;
    const authored = structuredClone(edits);
    const applied = applyAuthoredEdits(this.draft, address, authored, true);
    if (!applied.inverse.length) return;
    const gesture = { address: { ...address }, name, edits: authored, inverse: applied.inverse, validate };
    this.stage(gesture, 'edit', authored, applied.project);
  }

  undo() {
    this.requireEditable();
    const gesture = this.history.undo[this.history.undo.length - 1];
    if (!gesture) return;
    const { project } = applyAuthoredEdits(this.draft, gesture.address, gesture.inverse);
    this.stage(gesture, 'undo', gesture.inverse, project);
  }

  redo() {
    this.requireEditable();
    const gesture = this.history.redo[this.history.redo.length - 1];
    if (!gesture) return;
    const { project } = applyAuthoredEdits(this.draft, gesture.address, gesture.edits);
    this.stage(gesture, 'redo', gesture.edits, project);
  }

  flush(): Promise<string> {
    this.cancelSettle();
    return this.sequence(async () => {
      if (!this.queued.length) await this.refresh();
      await this.drain();
      return this.saved.revision;
    });
  }

  rename(name: StudioProjectRename['name']): Promise<StudioProjectDocument> {
    return this.authoredCommand(
      () => this.transport.rename(this.saved.project_id, {
        revision: this.saved.revision, name,
      }, this.abort.signal),
      (document) => document.configuration.project === name,
    );
  }

  createFile(file: NamedFileCreation): Promise<StudioProjectDocument> {
    const intended = structuredClone(file);
    return this.authoredCommand(
      () => this.transport.create(this.saved.project_id, {
        revision: this.saved.revision, file: intended,
      }, this.abort.signal),
      (document, before) => !before.file_revisions.some(({ address }) =>
        addressKey(address) === addressKey(intended)) &&
        containsCreatedFile(document.snapshot.project, intended),
      intended,
    );
  }

  private authoredCommand(
    write: () => Promise<StudioProjectDocument>,
    achieved: (document: StudioProjectDocument, before: StudioProjectSnapshot) => boolean,
    created?: NamedFileCreation,
  ): Promise<StudioProjectDocument> {
    this.cancelSettle();
    return this.sequence(async () => {
      await this.refresh();
      await this.drain();
      const before = this.saved;
      let result: StudioProjectDocument;
      try {
        result = await write();
      } catch (error) {
        this.abort.signal.throwIfAborted();
        const ambiguous = canRetryOnReconnect(error);
        if (ambiguous || isHttpConflict(error)) {
          const latest = await this.refresh(ambiguous ? created : undefined).catch(() => undefined);
          if (ambiguous && latest && achieved(latest, before)) return latest;
        }
        throw error;
      }
      this.abort.signal.throwIfAborted();
      this.receive(result, created);
      return result;
    });
  }

  observe(revision: StudioProjectRevision) {
    if (revision.project_id !== this.saved.project_id || this.abort.signal.aborted) return;
    this.observation = revision;
    if (this.observationScheduled) return;
    this.observationScheduled = true;
    void this.sequence(async () => {
      const current = this.observation;
      this.observation = undefined;
      this.observationScheduled = false;
      if (current && !versionsMatch(current, this.outside?.snapshot ?? this.saved)) await this.refresh();
    }).catch(() => {});
  }

  connected() {
    if (this.abort.signal.aborted) return;
    void this.sequence(async () => {
      const retry = this.queued.length > 0 && canRetryOnReconnect(this.error);
      await this.refresh();
      if (retry && !this.outside) await this.drain();
    }).catch(() => {});
  }

  keep(choice: 'disk' | 'edits') {
    return this.sequence(async () => {
      if (!this.outside) return;
      const latest = await this.read().catch((error: unknown) => {
        this.error = error;
        this.publish();
        throw error;
      });
      this.outside = latest;
      const unchanged = unchangedFiles(this.saved, latest.snapshot);
      const kept = withoutChangedFiles(this.savedHistory, unchanged);
      if (choice === 'disk') {
        this.queued.length = 0;
        this.adopt(latest, kept);
        return;
      }
      let project = latest.snapshot.project;
      let history = kept;
      const rebased: PendingEdit[] = [];
      const alreadyOnDisk = new Set<string>();
      try {
        for (const pending of this.queued) {
          const { address, name } = pending.gesture;
          const key = addressKey(address);
          const sameFile = unchanged.has(key);
          // A save response can be lost after the file was written. Keep the
          // observed value without attributing an unacknowledged write to undo.
          if (!sameFile && (alreadyOnDisk.has(key) || sameAuthoredFile(this.draft, latest.snapshot.project, address))) {
            alreadyOnDisk.add(key);
            continue;
          }
          const base = project;
          const applied = applyAuthoredEdits(base, address, pending.edits, !sameFile, sameFile ? undefined : pending.base);
          if (!applied.inverse.length) continue;
          pending.gesture.validate?.(base, applied.project);
          const gesture = sameFile ? pending.gesture : { ...pending.gesture, address, name, edits: applied.edits, inverse: applied.inverse };
          const direction = sameFile ? pending.direction : 'edit';
          project = applied.project;
          history = moveHistory(history, gesture, direction);
          rebased.push({ base, gesture, direction, edits: applied.edits, historyAfter: history });
        }
      } catch (error) {
        this.error = error;
        this.publish();
        throw error;
      }
      this.queued.splice(0, this.queued.length, ...rebased);
      this.document = latest;
      this.savedHistory = kept;
      this.history = history;
      this.draft = project;
      this.outside = null;
      this.error = null;
      this.publish();
      await this.drain();
    });
  }

  dispose() {
    this.cancelSettle();
    this.abort.abort();
    this.listeners.clear();
  }

  private capture(): ProjectEditSnapshot {
    return {
      saved: this.saved,
      configuration: this.document.configuration,
      draft: this.draft,
      pending: this.queued.length,
      saving: this.saving,
      error: this.error,
      outside: this.outside?.snapshot ?? null,
      undoName: this.history.undo[this.history.undo.length - 1]?.name ?? null,
      redoName: this.history.redo[this.history.redo.length - 1]?.name ?? null,
    };
  }

  private publish() {
    if (this.abort.signal.aborted) return;
    this.snapshot = this.capture();
    for (const listener of this.listeners) listener();
  }

  private requireEditable() {
    this.abort.signal.throwIfAborted();
    if (this.outside) throw new ProjectEditConflictError();
  }

  private stage(
    gesture: Gesture,
    direction: PendingEdit['direction'],
    edits: readonly ProjectValueEdit[],
    project: AuthoredProject,
  ) {
    gesture.validate?.(this.draft, project);
    const historyAfter = moveHistory(this.history, gesture, direction);
    this.queued.push({ base: this.draft, gesture, direction, edits, historyAfter });
    this.draft = project;
    this.history = historyAfter;
    this.publish();
    this.scheduleSettle();
  }

  private scheduleSettle() {
    this.cancelSettle();
    this.settle = setTimeout(() => { void this.flush().catch(() => {}); }, this.settleMs);
  }

  private cancelSettle() {
    clearTimeout(this.settle);
    this.settle = undefined;
  }

  private sequence<T>(operation: () => Promise<T>): Promise<T> {
    const work = this.serial.then(() => {
      this.abort.signal.throwIfAborted();
      return operation();
    });
    this.serial = work.catch(() => {});
    return work;
  }

  private async read() {
    const snapshot = await this.transport.read(this.saved.project_id, this.abort.signal);
    this.abort.signal.throwIfAborted();
    return snapshot;
  }

  private adopt(document: StudioProjectDocument, history: History) {
    this.document = document;
    this.draft = document.snapshot.project;
    this.savedHistory = history;
    this.history = history;
    this.outside = null;
    this.error = null;
    this.publish();
  }

  private receive(document: StudioProjectDocument, created?: NamedFileCreation) {
    const { snapshot: latest } = document;
    const unchanged = unchangedFiles(this.saved, latest);
    if (created && this.queued.length && isAcknowledgedAddition(this.saved, latest, unchanged, created)) {
      let draft = latest.project;
      const rebased = this.queued.map((pending) => {
        const base = draft;
        draft = applyAuthoredEdits(base, pending.gesture.address, pending.edits).project;
        return { ...pending, base };
      });
      this.queued.splice(0, this.queued.length, ...rebased);
      this.document = document;
      this.draft = draft;
      this.outside = null;
      this.error = null;
      this.publish();
      return;
    }
    const sameAuthoredFiles = latest.semantic_root === this.saved.semantic_root &&
      unchanged.size === latest.file_revisions.length &&
      unchanged.size === this.saved.file_revisions.length;
    if (latest.revision === this.saved.revision || sameAuthoredFiles) {
      const wasConflicted = this.outside !== null;
      this.document = document;
      this.outside = null;
      if (!this.queued.length) {
        this.draft = latest.project;
        this.error = null;
      } else if (wasConflicted) {
        this.error = null;
        this.scheduleSettle();
      }
      this.publish();
    } else if (this.queued.length) {
      this.cancelSettle();
      this.outside = document;
      this.error = null;
      this.publish();
    } else {
      this.adopt(document, withoutChangedFiles(this.savedHistory, unchanged));
    }
  }

  private async refresh(created?: NamedFileCreation) {
    try {
      const latest = await this.read();
      this.receive(latest, created);
      return latest;
    } catch (error) {
      this.error = error;
      this.publish();
      throw error;
    }
  }

  private async drain() {
    this.requireEditable();
    if (!this.queued.length) return;
    this.saving = true;
    this.error = null;
    this.publish();
    try {
      while (this.queued.length) {
        const first = this.queued[0];
        const address = first.gesture.address;
        const key = addressKey(address);
        const boundary = this.queued.findIndex(({ gesture }) => addressKey(gesture.address) !== key);
        const batch = this.queued.slice(0, boundary === -1 ? this.queued.length : boundary);
        const latest = await this.transport.write(this.saved.project_id, {
          revision: this.saved.revision,
          address,
          patches: batch.flatMap(({ edits }) => edits),
        }, this.abort.signal);
        this.abort.signal.throwIfAborted();
        this.document = { ...this.document, snapshot: latest };
        this.savedHistory = batch[batch.length - 1].historyAfter;
        this.queued.splice(0, batch.length);
        if (!this.queued.length) this.draft = latest.project;
        this.publish();
      }
      this.cancelSettle();
    } catch (error) {
      this.error = error;
      if (isHttpConflict(error)) await this.refresh();
      throw error;
    } finally {
      this.saving = false;
      this.publish();
    }
  }
}
