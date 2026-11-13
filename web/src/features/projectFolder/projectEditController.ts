import { ApiRequestError, isHttpConflict } from '@/api/request';
import { studioName } from '@/components/activity/studioVocabulary';
import type {
  AuthoredFileAddress,
  ModelAddress,
  ModelEditCommand,
  ModelEditRequest,
  ModelEditResponse,
  ModelUndoRequest,
  PresentationDocument,
  ProjectValueEdit,
  StudioProjectCreateFile,
  StudioProjectDocument,
  StudioProjectFilePatch,
  StudioProjectPatch,
  StudioProjectPatchBatch,
  StudioProjectRename,
  StudioProjectReplay,
  StudioProjectRevision,
  StudioProjectSnapshot,
} from '@/generated/studioContracts';
import {
  addressKey, applyAuthoredEdits, presentationAddress, presentationHistoryEdits, sameAuthoredFile,
  type AuthoredProject, type ProjectEditContent,
} from './authoredEdits';

export interface ProjectEditTransport {
  read: (id: string, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  write: (id: string, body: StudioProjectPatch | StudioProjectPatchBatch, signal?: AbortSignal) => Promise<StudioProjectSnapshot>;
  replay: (id: string, body: StudioProjectReplay, signal?: AbortSignal) => Promise<StudioProjectSnapshot>;
  rename: (id: string, body: StudioProjectRename, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  create: (id: string, body: StudioProjectCreateFile, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  editModel?: (id: string, body: ModelEditRequest, signal?: AbortSignal) => Promise<ModelEditResponse>;
  undoModel?: (id: string, body: ModelUndoRequest, signal?: AbortSignal) => Promise<ModelEditResponse>;
}

type NamedFileCreation = StudioProjectCreateFile['file'];

interface FileGesture {
  readonly name: string;
  readonly files: readonly StudioProjectFilePatch[];
  readonly inverse: readonly StudioProjectFilePatch[];
  readonly validate?: ProjectEditValidation;
  readonly reviewedRevision?: string;
  readonly reviewedPresentationRevision?: string;
  readonly dependencies: readonly AuthoredFileAddress[];
  readonly presentation?: { before: PresentationDocument; after: PresentationDocument };
}

interface ModelGesture {
  readonly name: string;
  readonly document: string;
  readonly token: string;
}

interface OutsideGesture {
  readonly name: string;
  readonly undo: readonly ReplayTarget[];
  readonly redo: readonly ReplayTarget[];
}

type Gesture = FileGesture | ModelGesture | OutsideGesture;
type ReplayFile = StudioProjectReplay['files'][number];
interface ReplayTarget {
  readonly address: ReplayFile['address'];
  readonly data: string | null;
  readonly revision: string | null;
  readonly content?: string;
}
const declarationAddress = { kind: 'declaration', id: 'declaration' } as const;

export type ProjectEditValidation = (before: AuthoredProject, after: AuthoredProject) => void;

interface History {
  readonly undo: readonly Gesture[];
  readonly redo: readonly Gesture[];
}

interface OutsideBurst {
  readonly before: StudioProjectSnapshot;
  readonly history: History;
}

interface PendingEdit {
  readonly base: ProjectEditContent;
  readonly gesture: FileGesture;
  readonly direction: 'edit' | 'undo' | 'redo';
  readonly files: readonly StudioProjectFilePatch[];
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
const presentationKey = addressKey(presentationAddress);

function editContent(snapshot: StudioProjectSnapshot): ProjectEditContent {
  return { ...snapshot.project, presentation: snapshot.presentation };
}

function touchesPresentation(files: readonly StudioProjectFilePatch[]) {
  return files.some(({ address }) => address.kind === 'presentation');
}

function moveHistory(history: History, gesture: Gesture, direction: PendingEdit['direction']): History {
  switch (direction) {
    case 'edit': return { undo: [...history.undo, gesture], redo: [] };
    case 'undo': return { undo: history.undo.filter((item) => item !== gesture), redo: [...history.redo, gesture] };
    case 'redo': return { undo: [...history.undo, gesture], redo: history.redo.filter((item) => item !== gesture) };
  }
}

function unchangedFiles(previous: StudioProjectSnapshot, next: StudioProjectSnapshot) {
  const current = new Map(next.file_revisions.map(({ address, revision }) => [addressKey(address), revision]));
  const unchanged = new Set(previous.file_revisions.flatMap(({ address, revision }) => {
    const key = addressKey(address);
    return current.get(key) === revision ? [key] : [];
  }));
  if (previous.presentation_revision === next.presentation_revision) unchanged.add(presentationKey);
  return unchanged;
}

function unchangedAuthoredCount(unchanged: ReadonlySet<string>) {
  return unchanged.size - Number(unchanged.has(presentationKey));
}

function withoutChangedFiles(
  history: History, unchanged: ReadonlySet<string>, declarationUnchanged: boolean,
): History {
  const usable = (gesture: Gesture) => {
    if ('token' in gesture) return unchanged.has(addressKey({ kind: 'document', id: gesture.document }));
    if ('undo' in gesture) return gesture.undo.every(({ address }) =>
      address.kind === 'declaration' ? declarationUnchanged : unchanged.has(addressKey(address)));
    return gesture.files.every(({ address }) => address.kind === 'presentation' || unchanged.has(addressKey(address))) &&
      gesture.dependencies.every((address) => unchanged.has(addressKey(address)));
  };
  return { undo: history.undo.filter(usable), redo: history.redo.filter(usable) };
}

function authoredFileContent(project: AuthoredProject, { kind, id }: ReplayFile['address']) {
  const files = kind === 'document' ? project.documents : kind === 'step' ? project.steps
    : kind === 'variant' ? project.sweep?.rows : undefined;
  return files && Object.prototype.hasOwnProperty.call(files, id) ? JSON.stringify(files[id]) : undefined;
}

function outsideGesture(before: StudioProjectSnapshot, after: StudioProjectSnapshot): OutsideGesture | null {
  const previous = new Map(before.file_revisions.map((file) => [addressKey(file.address), file]));
  const current = new Map(after.file_revisions.map((file) => [addressKey(file.address), file]));
  const undo: ReplayTarget[] = [];
  const redo: ReplayTarget[] = [];
  for (const key of new Set([...previous.keys(), ...current.keys()])) {
    const from = previous.get(key);
    const to = current.get(key);
    if (from?.revision === to?.revision) continue;
    const { kind, id } = (to ?? from)!.address;
    if (kind === 'presentation' || (from && from.data == null) || (to && to.data == null)) return null;
    const address = { kind, id };
    undo.push({
      address, data: from?.data ?? null, revision: from?.revision ?? null,
      content: authoredFileContent(before.project, address),
    });
    redo.push({
      address, data: to?.data ?? null, revision: to?.revision ?? null,
      content: authoredFileContent(after.project, address),
    });
  }
  if (before.declaration?.authored_revision !== after.declaration?.authored_revision) {
    if (!before.declaration || !after.declaration) return null;
    const target = ({ data, authored_revision }: NonNullable<StudioProjectSnapshot['declaration']>) =>
      ({ address: declarationAddress, data, revision: authored_revision });
    undo.push(target(before.declaration));
    redo.push(target(after.declaration));
  }
  return { name: studioName('outsideChange'), undo, redo };
}

function replayFiles(
  from: readonly ReplayTarget[], to: readonly ReplayTarget[], current: StudioProjectSnapshot,
): ReplayFile[] {
  const digests = new Map(current.file_revisions.map(({ address, revision }) => [addressKey(address), revision]));
  return to.map(({ address, data }, index) => {
    const declaration = address.kind === 'declaration';
    const before = declaration ? current.declaration?.revision : digests.get(addressKey(address));
    const actual = declaration ? current.declaration?.authored_revision : before;
    const source = from[index];
    // Earlier local Undo can re-encode a file with the same ordered content.
    const sameContent = !declaration && source.content !== undefined &&
      source.content === authoredFileContent(current.project, address);
    if ((actual ?? null) !== source.revision && !sameContent) throw new ProjectEditConflictError();
    return { address, before: before ?? null, after: data };
  });
}

function replayObserved(
  before: StudioProjectSnapshot, targets: readonly ReplayTarget[], observed: StudioProjectSnapshot,
): boolean {
  if (before.presentation_revision !== observed.presentation_revision) return false;
  const expected = new Map(before.file_revisions.map(({ address, revision }) => [addressKey(address), revision]));
  let declaration = before.declaration?.authored_revision ?? null;
  for (const { address, revision } of targets) {
    if (address.kind === 'declaration') {
      declaration = revision;
    } else {
      const key = addressKey(address);
      if (revision === null) expected.delete(key);
      else expected.set(key, revision);
    }
  }
  return observed.file_revisions.length === expected.size &&
    observed.file_revisions.every(({ address, revision }) => expected.get(addressKey(address)) === revision) &&
    declaration === (observed.declaration?.authored_revision ?? null);
}

function historyFiles(gesture: FileGesture, direction: 'undo' | 'redo', draft: ProjectEditContent) {
  const files = direction === 'undo' ? gesture.inverse : gesture.files;
  if (!gesture.presentation) return files;
  if (!draft.presentation) throw new ProjectEditConflictError();
  const { before, after } = gesture.presentation;
  const patches = direction === 'undo'
    ? presentationHistoryEdits(after, before, draft.presentation)
    : presentationHistoryEdits(before, after, draft.presentation);
  return [
    ...files.filter(({ address }) => address.kind !== 'presentation'),
    ...(patches.length ? [{ address: presentationAddress, patches }] : []),
  ];
}

function applyFiles(project: ProjectEditContent, changes: readonly StudioProjectFilePatch[], checkOrder = false) {
  const files: StudioProjectFilePatch[] = [];
  const inverse: StudioProjectFilePatch[] = [];
  for (const { address, patches } of changes) {
    const applied = applyAuthoredEdits(project, address, patches, checkOrder);
    project = applied.project;
    if (!applied.inverse.length) continue;
    files.push({ address, patches: applied.edits });
    inverse.unshift({ address, patches: applied.inverse });
  }
  return { project, files, inverse };
}

function versionsMatch(a: StudioProjectRevision, b: StudioProjectRevision) {
  return a.revision === b.revision && a.presentation_revision === b.presentation_revision;
}

function canRetryOnReconnect(error: unknown) {
  return error instanceof ApiRequestError && (error.kind === 'network' || (error.status ?? 0) >= 500);
}

export function containsCreatedFile(project: AuthoredProject, file: NamedFileCreation) {
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
  const count = unchangedAuthoredCount(unchanged);
  return count === before.file_revisions.length &&
    after.file_revisions.length === count + 1 &&
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
  private draft: ProjectEditContent;
  private history: History = emptyHistory;
  private savedHistory: History = emptyHistory;
  private outsideBurst: OutsideBurst | undefined;
  private outside: StudioProjectDocument | null = null;
  private error: unknown = null;
  private saving = false;
  private transactionWriting = false;
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
    this.draft = editContent(initial.snapshot);
    this.snapshot = this.capture();
  }

  private get saved() { return this.document.snapshot; }

  get signal() { return this.abort.signal; }

  getSnapshot = () => this.snapshot;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  edit(address: AuthoredFileAddress, name: string, edits: readonly ProjectValueEdit[], validate?: ProjectEditValidation) {
    this.editFiles(name, [{ address, patches: [...edits] }], { validate });
  }

  editFiles(name: string, changes: readonly StudioProjectFilePatch[], options: {
    validate?: ProjectEditValidation; reviewedRevision?: string; reviewedPresentationRevision?: string;
    dependencies?: readonly AuthoredFileAddress[];
  } = {}) {
    this.requireEditable();
    const { reviewedRevision, reviewedPresentationRevision, dependencies = [], validate } = options;
    if (reviewedRevision !== undefined && (this.saved.revision !== reviewedRevision || this.queued.length)) {
      throw new ProjectEditConflictError();
    }
    if (touchesPresentation(changes) &&
        (reviewedPresentationRevision !== this.saved.presentation_revision || this.queued.length)) {
      throw new ProjectEditConflictError();
    }
    const applied = applyFiles(this.draft, structuredClone(changes), true);
    if (!applied.inverse.length) return;
    const before = this.draft.presentation;
    const after = applied.project.presentation;
    const placement = touchesPresentation(applied.files);
    if (placement && (!before || !after)) throw new ProjectEditConflictError();
    const gesture = {
      name, files: applied.files, inverse: applied.inverse, validate,
      reviewedRevision, reviewedPresentationRevision, dependencies,
      presentation: placement && before && after ? { before, after } : undefined,
    };
    this.stage(gesture, 'edit', gesture.files, applied.project);
  }

  undo() {
    return this.replay('undo');
  }

  redo() {
    return this.replay('redo');
  }

  private replay(direction: 'undo' | 'redo'): Promise<readonly ModelAddress[]> | undefined {
    this.requireEditable();
    const gesture = this.history[direction].at(-1);
    if (!gesture) return;
    if ('undo' in gesture) {
      return this.exclusiveTransaction(async (before) => {
        if (this.savedHistory[direction].at(-1) !== gesture) throw new ProjectEditConflictError();
        const target = gesture[direction];
        const source = gesture[direction === 'undo' ? 'redo' : 'undo'];
        const body = { revision: before.revision, files: replayFiles(source, target, before) };
        let latest: StudioProjectDocument;
        try {
          const snapshot = await this.transport.replay(before.project_id, body, this.signal);
          latest = { ...this.document, snapshot };
        } catch (error) {
          if (!canRetryOnReconnect(error)) throw error;
          const observed = await this.read().catch(() => undefined);
          if (!observed || !replayObserved(before, target, observed.snapshot)) throw error;
          latest = observed;
        }
        this.signal.throwIfAborted();
        this.adopt(latest, moveHistory(this.savedHistory, gesture, direction));
        return [];
      });
    }
    if ('token' in gesture) {
      return this.exclusiveTransaction(async (before) => {
        const undo = this.transport.undoModel;
        if (!undo || this.history[direction].at(-1) !== gesture) throw new ProjectEditConflictError();
        const { result } = await undo(before.project_id, {
          revision: before.revision, presentation_revision: before.presentation_revision, token: gesture.token,
        }, this.signal);
        if (result.status !== 'applied' || typeof result.undo_token !== 'string') throw new ProjectEditConflictError();
        const history = moveHistory(this.savedHistory, gesture, direction);
        const destination = direction === 'undo' ? 'redo' : 'undo';
        this.adopt({ ...this.document, snapshot: result.snapshot }, {
          ...history, [destination]: [...history[destination].slice(0, -1), { ...gesture, token: result.undo_token }],
        });
        return result.changed;
      });
    }
    const files = historyFiles(gesture, direction, this.draft);
    const { project } = applyFiles(this.draft, files);
    this.stage(gesture, direction, files, project);
  }

  editModel(name: string, document: string, command: ModelEditCommand, reviewed: StudioProjectRevision) {
    return this.exclusiveTransaction(async (before) => {
      const edit = this.transport.editModel;
      if (!edit || !versionsMatch(before, reviewed)) throw new ProjectEditConflictError();
      const { result } = await edit(before.project_id, {
        revision: before.revision, presentation_revision: before.presentation_revision, document, command,
      }, this.signal);
      if (result.status === 'applied') {
        const history = typeof result.undo_token === 'string'
          ? moveHistory(this.savedHistory, { name, document, token: result.undo_token }, 'edit') : this.savedHistory;
        this.adopt({ ...this.document, snapshot: result.snapshot }, history);
      }
      return result;
    });
  }

  private exclusiveTransaction<T>(operation: (before: StudioProjectSnapshot) => Promise<T>): Promise<T> {
    this.requireEditable();
    this.cancelSettle();
    this.transactionWriting = true;
    this.publish();
    const work = this.sequence(async () => {
      await this.drain();
      if (this.outside) throw new ProjectEditConflictError();
      this.outsideBurst = undefined;
      try {
        return await operation(this.saved);
      } catch (error) {
        if (isHttpConflict(error) || canRetryOnReconnect(error)) await this.refresh().catch(() => undefined);
        throw error;
      }
    });
    return work.finally(() => {
      this.transactionWriting = false;
      this.publish();
    });
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

  createFile(file: NamedFileCreation, reviewedRevision?: string): Promise<StudioProjectDocument> {
    const intended = structuredClone(file);
    return this.authoredCommand(
      () => this.transport.create(this.saved.project_id, {
        revision: this.saved.revision, file: intended,
      }, this.abort.signal),
      (document, before) => !before.file_revisions.some(({ address }) =>
        addressKey(address) === addressKey(intended)) &&
        containsCreatedFile(document.snapshot.project, intended),
      intended,
      reviewedRevision,
    );
  }

  private authoredCommand(
    write: () => Promise<StudioProjectDocument>,
    achieved: (document: StudioProjectDocument, before: StudioProjectSnapshot) => boolean,
    created?: NamedFileCreation,
    reviewedRevision?: string,
  ): Promise<StudioProjectDocument> {
    this.cancelSettle();
    return this.sequence(async () => {
      await this.refresh();
      await this.drain();
      const before = this.saved;
      if (reviewedRevision !== undefined && before.revision !== reviewedRevision) {
        throw new ProjectEditConflictError();
      }
      let result: StudioProjectDocument;
      try {
        result = await write();
      } catch (error) {
        this.abort.signal.throwIfAborted();
        const ambiguous = canRetryOnReconnect(error);
        if (ambiguous || isHttpConflict(error)) {
          const latest = await this.refresh(ambiguous ? created : undefined).catch(() => undefined);
          if (ambiguous && latest && achieved(latest, before)) {
            this.outsideBurst = undefined;
            return latest;
          }
        }
        throw error;
      }
      this.abort.signal.throwIfAborted();
      this.receive(result, created, true);
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
      const kept = withoutChangedFiles(
        this.savedHistory, unchanged, this.saved.declaration?.authored_revision === latest.snapshot.declaration?.authored_revision,
      );
      if (choice === 'disk') {
        this.queued.length = 0;
        this.adopt(latest, kept);
        return;
      }
      const disk = editContent(latest.snapshot);
      let project = disk;
      let history = kept;
      const rebased: PendingEdit[] = [];
      const alreadyOnDisk = new Set<string>();
      try {
        for (const pending of this.queued) {
          const { files, gesture } = pending;
          if (files.length !== 1 || gesture.reviewedRevision !== undefined) {
            if (files.every(({ address }) => sameAuthoredFile(this.draft, disk, address))) continue;
            if (pending.direction === 'edit' && gesture.reviewedRevision !== undefined ||
                !files.every(({ address }) => unchanged.has(addressKey(address))) ||
                !gesture.dependencies.every((address) => unchanged.has(addressKey(address)))) {
              throw new ProjectEditConflictError();
            }
            const base = project;
            project = applyFiles(base, files).project;
            history = moveHistory(history, gesture, pending.direction);
            rebased.push({ ...pending, base, historyAfter: history });
            continue;
          }
          const [{ address, patches }] = files;
          const key = addressKey(address);
          const sameFile = unchanged.has(key);
          // A save response can be lost after the file was written. Keep the
          // observed value without attributing an unacknowledged write to undo.
          if (!sameFile && (alreadyOnDisk.has(key) || sameAuthoredFile(this.draft, disk, address))) {
            alreadyOnDisk.add(key);
            continue;
          }
          const base = project;
          const applied = applyAuthoredEdits(base, address, patches, !sameFile, sameFile ? undefined : pending.base);
          if (!applied.inverse.length) continue;
          pending.gesture.validate?.(base, applied.project);
          const edits = [{ address, patches: applied.edits }];
          const adjusted = sameFile ? gesture : { ...gesture, files: edits, inverse: [{ address, patches: applied.inverse }] };
          const direction = sameFile ? pending.direction : 'edit';
          project = applied.project;
          history = moveHistory(history, adjusted, direction);
          rebased.push({ base, gesture: adjusted, direction, files: edits, historyAfter: history });
        }
      } catch (error) {
        this.error = error;
        this.publish();
        throw error;
      }
      this.queued.splice(0, this.queued.length, ...rebased);
      this.document = latest;
      this.outsideBurst = undefined;
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
    const { presentation: _, ...draft } = this.draft;
    return {
      saved: this.saved,
      configuration: this.document.configuration,
      draft,
      pending: this.queued.length,
      saving: this.saving || this.transactionWriting,
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
    if (this.outside || this.transactionWriting) throw new ProjectEditConflictError();
  }

  private stage(
    gesture: FileGesture,
    direction: PendingEdit['direction'],
    files: readonly StudioProjectFilePatch[],
    project: ProjectEditContent,
  ) {
    gesture.validate?.(this.draft, project);
    this.outsideBurst = undefined;
    const historyAfter = moveHistory(this.history, gesture, direction);
    this.queued.push({ base: this.draft, gesture, direction, files, historyAfter });
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

  private adopt(document: StudioProjectDocument, history: History, burst?: OutsideBurst) {
    this.document = document;
    this.draft = editContent(document.snapshot);
    this.outsideBurst = burst;
    this.savedHistory = history;
    this.history = history;
    this.outside = null;
    this.error = null;
    this.publish();
  }

  private receive(document: StudioProjectDocument, created?: NamedFileCreation, authored = false) {
    const { snapshot: latest } = document;
    const unchanged = unchangedFiles(this.saved, latest);
    const added = created && isAcknowledgedAddition(this.saved, latest, unchanged, created);
    if (authored || added) this.outsideBurst = undefined;
    if (added && this.queued.length) {
      let draft = editContent(latest);
      const rebased = this.queued.map((pending) => {
        const base = draft;
        draft = applyFiles(base, pending.files).project;
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
    const count = unchangedAuthoredCount(unchanged);
    const declarationUnchanged = this.saved.declaration?.authored_revision === latest.declaration?.authored_revision;
    const sameAuthoredFiles = declarationUnchanged && latest.semantic_root === this.saved.semantic_root &&
      count === latest.file_revisions.length && count === this.saved.file_revisions.length;
    const presentationChanged = !unchanged.has(presentationKey);
    const placementPending = this.queued.some(({ files }) => touchesPresentation(files));
    if (presentationChanged && placementPending) {
      this.cancelSettle();
      this.outside = document;
      this.error = null;
      this.publish();
    } else if (latest.revision === this.saved.revision || sameAuthoredFiles) {
      const wasConflicted = this.outside !== null;
      this.document = document;
      this.outside = null;
      if (!this.queued.length) {
        this.draft = editContent(latest);
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
    } else if (authored || added) {
      this.adopt(document, withoutChangedFiles(
        this.savedHistory, unchanged, declarationUnchanged,
      ));
    } else {
      const burst = this.outsideBurst ?? { before: this.saved, history: this.savedHistory };
      const gesture = outsideGesture(burst.before, latest);
      if (!gesture) {
        this.adopt(document, withoutChangedFiles(this.savedHistory, unchanged, declarationUnchanged));
        return;
      }
      const history = gesture.undo.length ? moveHistory(burst.history, gesture, 'edit') : burst.history;
      this.adopt(document, history, burst);
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

  private async savePatch(body: StudioProjectPatch | StudioProjectPatchBatch, exact: boolean) {
    try {
      return await this.transport.write(this.saved.project_id, body, this.abort.signal);
    } catch (error) {
      if (!exact || !canRetryOnReconnect(error)) throw error;
      const observed = await this.read().catch(() => undefined);
      if (!observed) throw error;
      const files = 'files' in body ? body.files : [body];
      const before = this.saved;
      const latest = observed.snapshot;
      const changed = new Set(files.map(({ address }) => addressKey(address)));
      const unchanged = unchangedFiles(before, latest);
      const intended = applyFiles(editContent(before), files).project;
      if (before.file_revisions.length !== latest.file_revisions.length ||
          !latest.file_revisions.every(({ address }) => {
            const key = addressKey(address);
            return unchanged.has(key) || changed.has(key);
          }) ||
          JSON.stringify(intended.sweep) !== JSON.stringify(latest.project.sweep) ||
          !files.every(({ address }) => sameAuthoredFile(intended, editContent(latest), address))) throw error;
      return latest;
    }
  }

  private async drain() {
    this.signal.throwIfAborted();
    if (this.outside) throw new ProjectEditConflictError();
    if (!this.queued.length) return;
    this.saving = true;
    this.error = null;
    this.publish();
    try {
      while (this.queued.length) {
        const first = this.queued[0];
        const revision = this.saved.revision;
        if (first.direction === 'edit' && first.gesture.reviewedRevision !== undefined &&
            first.gesture.reviewedRevision !== revision) throw new ProjectEditConflictError();
        const placement = touchesPresentation(first.files);
        const presentationRevision = this.saved.presentation_revision;
        if (first.direction === 'edit' && placement &&
            first.gesture.reviewedPresentationRevision !== presentationRevision) throw new ProjectEditConflictError();
        const revisions = placement ? { revision, presentation_revision: presentationRevision } : { revision };
        const batch = [first];
        let body: StudioProjectPatch | StudioProjectPatchBatch;
        if (first.files.length === 1) {
          const [{ address, patches }] = first.files;
          const key = addressKey(address);
          if (first.gesture.reviewedRevision === undefined) {
            for (const pending of this.queued.slice(1)) {
              if (pending.gesture.reviewedRevision !== undefined || pending.files.length !== 1 ||
                  addressKey(pending.files[0].address) !== key) break;
              batch.push(pending);
            }
          }
          body = { ...revisions, address, patches: batch.length === 1 ? patches : batch.flatMap(({ files }) => files[0].patches) };
        } else {
          body = { ...revisions, files: [...first.files] };
        }
        const exact = first.gesture.reviewedRevision !== undefined || first.files.length > 1;
        const latest = await this.savePatch(body, exact);
        this.abort.signal.throwIfAborted();
        this.document = { ...this.document, snapshot: latest };
        this.savedHistory = batch[batch.length - 1].historyAfter;
        this.queued.splice(0, batch.length);
        if (!this.queued.length) this.draft = editContent(latest);
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
