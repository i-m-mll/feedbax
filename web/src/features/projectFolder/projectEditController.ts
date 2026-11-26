import { isHttpConflict, isRefusal, isServiceFailure } from '@/api/request';
import { studioName } from '@/components/activity/studioVocabulary';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import type {
  AuthoredFileAddress,
  CreateNamedDocument,
  ModelEditCommand,
  ModelEditRequest,
  ModelEditResponse,
  ModelUndoRequest,
  PresentationDocument,
  StudioMemberCommand,
  StudioMemberCommandResult,
  StudioMemberName,
  StudioMemberNameRequest,
  ProjectValueEdit,
  StudioProjectCreateFile,
  StudioProjectCreateFiles,
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
  addressKey, applyAuthoredEdits, declarationAddress, presentationAddress, presentationHistoryEdits, sameAuthoredFile,
  type AuthoredProject, type ProjectEditContent,
} from './authoredEdits';
import { commandQueue } from './commandQueue';
import { modelSchema } from './model/modelGraph';
import type { ModelChange } from '@/features/projectWindow/elementSelection';

export interface ProjectEditTransport {
  read: (id: string, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  write: (id: string, body: StudioProjectPatch | StudioProjectPatchBatch, signal?: AbortSignal) => Promise<StudioProjectSnapshot>;
  replay: (id: string, body: StudioProjectReplay, signal?: AbortSignal) => Promise<StudioProjectSnapshot>;
  rename: (id: string, body: StudioProjectRename, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  create: (id: string, body: StudioProjectCreateFile | StudioProjectCreateFiles, signal?: AbortSignal) => Promise<StudioProjectDocument>;
  memberCommand?: (id: string, body: StudioMemberCommand, signal?: AbortSignal) => Promise<StudioMemberCommandResult>;
  nameMember?: (id: string, body: StudioMemberNameRequest, signal?: AbortSignal) => Promise<StudioMemberName>;
  editModel?: (id: string, body: ModelEditRequest, signal?: AbortSignal) => Promise<ModelEditResponse>;
  undoModel?: (id: string, body: ModelUndoRequest, signal?: AbortSignal) => Promise<ModelEditResponse>;
}

type NamedFileCreation = StudioProjectCreateFile['file'];

/** An edit as a route returns it: the documents it creates, and the patches that follow their creation. */
export interface StoredEdit {
  readonly files: readonly StudioProjectFilePatch[];
  readonly created?: readonly CreateNamedDocument[];
}
type FollowingEdits = (snapshot: StudioProjectSnapshot) => Promise<readonly StudioProjectFilePatch[]>;

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

interface ReplayGesture {
  readonly name: string;
  readonly undo: readonly ReplayTarget[];
  readonly redo: readonly ReplayTarget[];
  readonly presentation?: { readonly undo: PresentationState; readonly redo: PresentationState };
}

interface PresentationState {
  readonly revision: string;
  readonly document: PresentationDocument;
}

type Gesture = FileGesture | ModelGesture | ReplayGesture;
type ReplayFile = StudioProjectReplay['files'][number];
interface ReplayContent {
  readonly address: ReplayFile['address'];
  readonly data: string | null;
  readonly revision?: string | null;
}
interface ReplayTarget extends ReplayContent {
  readonly revision: string | null;
  readonly content?: string;
}

export type ProjectEditValidation = (before: AuthoredProject, after: AuthoredProject) => void;

interface History {
  readonly undo: readonly Gesture[];
  readonly redo: readonly Gesture[];
}

interface OutsideBurst {
  readonly before: StudioProjectSnapshot;
  readonly history: History;
}

/** How a queued edit ended: on disk, refused with the save's error, or discarded for the disk's version. */
export type EditSettlement =
  | { readonly status: 'saved' }
  | { readonly status: 'refused'; readonly error: unknown }
  | { readonly status: 'dropped' };

interface PendingEdit {
  readonly base: ProjectEditContent;
  readonly gesture: FileGesture;
  readonly direction: 'edit' | 'undo' | 'redo';
  readonly files: readonly StudioProjectFilePatch[];
  readonly historyAfter: History;
  /** Delivers the edit's settlement to the gesture that queued it, whoever drains the queue. */
  readonly settle: (settlement: EditSettlement) => void;
}

const savedEdit: EditSettlement = { status: 'saved' };

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
  /** Moves each time the saved plan may have changed since it was last read, so its readers read it again. */
  readonly planVersion: number;
}

export class ProjectEditConflictError extends Error {
  constructor() {
    super('The project changed before the pending edits could be saved.');
  }
}

const emptyHistory: History = { undo: [], redo: [] };
const presentationKey = addressKey(presentationAddress);
const declarationKey = addressKey(declarationAddress);

function editContent(snapshot: StudioProjectSnapshot): ProjectEditContent {
  return { ...snapshot.project, presentation: snapshot.presentation, settings: snapshot.settings };
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
  if (sameAuthoredFile(editContent(previous), editContent(next), declarationAddress)) unchanged.add(declarationKey);
  return unchanged;
}

function unchangedAuthoredCount(unchanged: ReadonlySet<string>) {
  return unchanged.size - Number(unchanged.has(presentationKey)) - Number(unchanged.has(declarationKey));
}

function withoutSettingsEdits(history: History): History {
  const usable = (gesture: Gesture) => !('files' in gesture) ||
    !gesture.files.some(({ address }) => addressKey(address) === declarationKey) &&
    !gesture.dependencies.some((address) => addressKey(address) === declarationKey);
  return { undo: history.undo.filter(usable), redo: history.redo.filter(usable) };
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
  if (!files || !Object.prototype.hasOwnProperty.call(files, id)) return undefined;
  const file = files[id];
  if (kind === 'document' && project.documents?.[id]?.schema === modelSchema) {
    // Addressed Model Undo uses canonical JSON for its payload preimage.
    return canonicalJsonV2(file);
  }
  return JSON.stringify(file);
}

function changedFiles(
  name: string, before: StudioProjectSnapshot, after: StudioProjectSnapshot, withPresentation = false,
): ReplayGesture | null {
  const previous = new Map(before.file_revisions.map((file) => [addressKey(file.address), file]));
  const current = new Map(after.file_revisions.map((file) => [addressKey(file.address), file]));
  const undo: ReplayTarget[] = [];
  const redo: ReplayTarget[] = [];
  for (const key of new Set([...previous.keys(), ...current.keys()])) {
    const from = previous.get(key);
    const to = current.get(key);
    if (from?.revision === to?.revision) continue;
    const { kind, id } = (to ?? from)!.address;
    if (kind === 'presentation' || kind === 'declaration' ||
        (from && from.data == null) || (to && to.data == null)) return null;
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
  if (!withPresentation || before.presentation_revision === after.presentation_revision) return { name, undo, redo };
  if (!before.presentation || !after.presentation) return null;
  return {
    name, undo, redo, presentation: {
      undo: { revision: before.presentation_revision, document: before.presentation },
      redo: { revision: after.presentation_revision, document: after.presentation },
    },
  };
}

function replayFiles(
  from: readonly ReplayTarget[], to: readonly ReplayContent[], current: StudioProjectSnapshot,
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
  before: StudioProjectSnapshot, targets: readonly ReplayContent[], observed: StudioProjectSnapshot,
): boolean {
  const expected = new Map(before.file_revisions.map(({ address, revision }) => [addressKey(address), revision]));
  const actual = new Map(observed.file_revisions.map((file) => [addressKey(file.address), file]));
  let declaration = before.declaration?.authored_revision;
  for (const { address, data, revision } of targets) {
    if (address.kind === 'declaration') {
      declaration = revision ?? undefined;
    } else {
      const key = addressKey(address);
      const file = actual.get(key);
      if (data === null ? file !== undefined : file?.data !== data) return false;
      expected.delete(key);
      actual.delete(key);
    }
  }
  return actual.size === expected.size &&
    [...actual].every(([key, { revision }]) => expected.get(key) === revision) &&
    declaration === observed.declaration?.authored_revision;
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

function applyFiles(project: ProjectEditContent, changes: readonly StudioProjectFilePatch[]) {
  const files: StudioProjectFilePatch[] = [];
  const inverse: StudioProjectFilePatch[] = [];
  for (const { address, patches } of changes) {
    const applied = applyAuthoredEdits(project, address, patches);
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
  unchanged: ReadonlySet<string>, files: readonly NamedFileCreation[],
) {
  const count = unchangedAuthoredCount(unchanged);
  return count === before.file_revisions.length &&
    unchanged.has(declarationKey) &&
    after.file_revisions.length === count + files.length &&
    JSON.stringify(before.project.sweep) === JSON.stringify(after.project.sweep) &&
    files.every((file) => {
      const key = addressKey(file);
      return !unchanged.has(key) && after.file_revisions.some(({ address }) => addressKey(address) === key) &&
        containsCreatedFile(after.project, file);
    });
}

export class ProjectEditController {
  private readonly abort = new AbortController();
  private readonly listeners = new Set<() => void>();
  private readonly queued: PendingEdit[] = [];
  private document: StudioProjectDocument;
  private draft: ProjectEditContent;
  private history: History = emptyHistory;
  private planVersion = 0;
  private savedHistory: History = emptyHistory;
  private outsideBurst: OutsideBurst | undefined;
  private outside: StudioProjectDocument | null = null;
  private error: unknown = null;
  private saving = false;
  private transactionWriting = false;
  private snapshot: ProjectEditSnapshot;
  private settle: ReturnType<typeof setTimeout> | undefined;
  private serial: Promise<unknown> = Promise.resolve();
  private readonly commands = commandQueue();
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
    return this.editFiles(name, [{ address, patches: [...edits] }], { validate });
  }

  editFiles(name: string, changes: readonly StudioProjectFilePatch[], options: {
    validate?: ProjectEditValidation; reviewedRevision?: string; reviewedPresentationRevision?: string;
    dependencies?: readonly AuthoredFileAddress[];
  } = {}): Promise<EditSettlement> {
    this.requireEditable();
    const { reviewedRevision, reviewedPresentationRevision, dependencies = [], validate } = options;
    if (reviewedRevision !== undefined && (this.saved.revision !== reviewedRevision || this.queued.length)) {
      throw new ProjectEditConflictError();
    }
    if (touchesPresentation(changes) &&
        (reviewedPresentationRevision !== this.saved.presentation_revision || this.queued.length)) {
      throw new ProjectEditConflictError();
    }
    const applied = applyFiles(this.draft, structuredClone(changes));
    if (!applied.inverse.length) return Promise.resolve(savedEdit);
    const before = this.draft.presentation;
    const after = applied.project.presentation;
    const placement = touchesPresentation(applied.files);
    if (placement && (!before || !after)) throw new ProjectEditConflictError();
    const gesture = {
      name, files: applied.files, inverse: applied.inverse, validate,
      reviewedRevision, reviewedPresentationRevision, dependencies,
      presentation: placement && before && after ? { before, after } : undefined,
    };
    return this.stage(gesture, 'edit', gesture.files, applied.project);
  }

  undo() {
    return this.replay('undo');
  }

  redo() {
    return this.replay('redo');
  }

  private replay(direction: 'undo' | 'redo'): Promise<ModelChange | undefined> | undefined {
    this.requireEditable();
    const gesture = this.history[direction].at(-1);
    if (!gesture) return;
    if ('undo' in gesture) {
      return this.exclusiveTransaction(async (before) => {
        if (this.savedHistory[direction].at(-1) !== gesture) throw new ProjectEditConflictError();
        const target = gesture[direction];
        const source = gesture[direction === 'undo' ? 'redo' : 'undo'];
        const presentation = gesture.presentation && {
          before: gesture.presentation[direction === 'undo' ? 'redo' : 'undo'].revision,
          after: gesture.presentation[direction].document,
        };
        if (presentation && presentation.before !== before.presentation_revision) throw new ProjectEditConflictError();
        const latest = await this.publishReplay(before, replayFiles(source, target, before), target, presentation);
        const history = moveHistory(this.savedHistory, gesture, direction);
        const destination = direction === 'undo' ? 'redo' : 'undo';
        // The server re-encodes the restored presentation, so its revision is the one to expect next.
        this.adopt(latest, gesture.presentation ? {
          ...history, [destination]: [...history[destination].slice(0, -1), {
            ...gesture, presentation: {
              ...gesture.presentation,
              [direction]: { ...gesture.presentation[direction], revision: latest.snapshot.presentation_revision },
            },
          }],
        } : history);
        return undefined;
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
        return { document: gesture.document, addresses: result.changed };
      });
    }
    const files = historyFiles(gesture, direction, this.draft);
    const { project } = applyFiles(this.draft, files);
    this.stage(gesture, direction, files, project);
  }

  /** Applies a Model command for `scope`'s steps (every reader when absent), with step files written alongside it. */
  editModel(
    name: string, document: string, command: ModelEditCommand, reviewed: StudioProjectRevision,
    scope?: Pick<ModelEditRequest, 'step' | 'chosen_steps' | 'variant' | 'rows' | 'axes' | 'files' | 'store_choice'>,
  ) {
    return this.exclusiveTransaction(async (before) => {
      const edit = this.transport.editModel;
      if (!edit || !versionsMatch(before, reviewed)) throw new ProjectEditConflictError();
      const { result } = await edit(before.project_id, {
        revision: before.revision, presentation_revision: before.presentation_revision, document, command, ...scope,
      }, this.signal);
      if (result.status === 'applied') {
        // An edit that wrote more than the shown document's file has no token; its entry is the files it changed.
        const gesture = typeof result.undo_token === 'string' ? { name, document, token: result.undo_token }
          : changedFiles(name, before, result.snapshot, true);
        this.adopt({ ...this.document, snapshot: result.snapshot },
          gesture ? moveHistory(this.savedHistory, gesture, 'edit') : this.savedHistory);
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
        if (isHttpConflict(error) || isServiceFailure(error)) await this.refresh().catch(() => undefined);
        throw error;
      }
    });
    return work.finally(() => {
      this.transactionWriting = false;
      this.publish();
    });
  }

  /**
   * Runs a view's edit command after the commands issued before it, once the edits stored before it have saved, with
   * the project as then saved. A field or form stays editable while a save is in flight; its edit waits here.
   */
  command<T>(command: (saved: StudioProjectSnapshot) => Promise<T>): Promise<T> {
    return this.commands(async () => {
      if (this.queued.length || this.saving || this.transactionWriting) await this.flush();
      return command(this.saved);
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

  rename(name: StudioProjectRename['name'], reviewedRevision?: string): Promise<StudioProjectDocument> {
    return this.authoredCommand(
      () => this.transport.rename(this.saved.project_id, {
        revision: this.saved.revision, name,
      }, this.abort.signal),
      (document) => document.configuration.project === name,
      undefined,
      reviewedRevision,
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

  /**
   * Store an edit a route computed, including any new documents and file patches, as one Undo entry.
   */
  async storeEdit(name: string, edit: StoredEdit, options: {
    reviewedRevision: string; validate?: ProjectEditValidation; dependencies?: readonly AuthoredFileAddress[];
  }): Promise<EditSettlement> {
    const created = edit.created ?? [];
    if (!created.length) return this.editFiles(name, edit.files, options);
    try {
      if (!edit.files.length) {
        await this.createFiles(name, created, options.reviewedRevision);
      } else {
        await this.exclusiveTransaction(async (before) => {
          if (before.revision !== options.reviewedRevision) throw new ProjectEditConflictError();
          const latest = await this.transport.write(before.project_id, {
            revision: before.revision,
            presentation_revision: before.presentation_revision,
            created: [...created],
            files: [...edit.files],
          }, this.signal);
          const gesture = changedFiles(name, before, latest, true);
          if (!gesture) throw new ProjectEditConflictError();
          this.adopt({ ...this.document, snapshot: latest }, moveHistory(this.savedHistory, gesture, 'edit'));
        });
      }
      return savedEdit;
    } catch (error) {
      if (isRefusal(error)) return { status: 'refused', error };
      throw error;
    }
  }

  /** Create files as one Undo entry, which the patches `use` computes against the created files join. */
  createFiles(
    name: string, files: readonly NamedFileCreation[], reviewedRevision: string, use?: FollowingEdits,
  ): Promise<StudioProjectSnapshot> {
    const intended = structuredClone([...files]);
    return this.exclusiveTransaction(async (before) => {
      if (before.revision !== reviewedRevision) throw new ProjectEditConflictError();
      const acknowledged = (after: StudioProjectSnapshot) => {
        const unchanged = unchangedFiles(before, after);
        return unchanged.has(presentationKey) && isAcknowledgedAddition(before, after, unchanged, intended);
      };
      let latest: StudioProjectSnapshot;
      try {
        latest = (await this.transport.create(before.project_id, {
          revision: before.revision, files: intended,
        }, this.signal)).snapshot;
      } catch (error) {
        if (!isServiceFailure(error)) throw error;
        const observed = (await this.read().catch(() => undefined))?.snapshot;
        if (!observed || !acknowledged(observed)) throw error;
        latest = observed;
      }
      this.signal.throwIfAborted();
      if (!acknowledged(latest)) throw new ProjectEditConflictError();
      return this.recordWithFollowing(name, before, latest, use);
    });
  }

  private async recordWithFollowing(
    name: string, before: StudioProjectSnapshot, done: StudioProjectSnapshot, use?: FollowingEdits,
  ): Promise<StudioProjectSnapshot> {
    const record = (latest: StudioProjectSnapshot) => {
      const gesture = changedFiles(name, before, latest, true);
      if (!gesture) throw new ProjectEditConflictError();
      this.adopt({ ...this.document, snapshot: latest }, moveHistory(this.savedHistory, gesture, 'edit'));
    };
    let latest = done;
    try {
      const files = use ? await use(done) : [];
      if (files.length) {
        latest = await this.transport.write(before.project_id, { revision: latest.revision, files: [...files] }, this.signal);
      }
      this.signal.throwIfAborted();
    } catch (error) {
      record(done);
      throw error;
    }
    record(latest);
    return latest;
  }

  private async publishReplay(
    before: StudioProjectSnapshot, files: ReplayFile[], targets: readonly ReplayContent[],
    presentation?: StudioProjectReplay['presentation'],
  ): Promise<StudioProjectDocument> {
    let latest: StudioProjectDocument;
    try {
      const snapshot = await this.transport.replay(before.project_id, {
        revision: before.revision, files, ...(presentation ? { presentation } : {}),
      }, this.signal);
      latest = { ...this.document, snapshot };
    } catch (error) {
      if (!isServiceFailure(error)) throw error;
      const observed = await this.read().catch(() => undefined);
      if (!observed || (!presentation && observed.snapshot.presentation_revision !== before.presentation_revision) ||
          !replayObserved(before, targets, observed.snapshot)) throw error;
      latest = observed;
    }
    this.signal.throwIfAborted();
    return latest;
  }

  /** The key and label the Service gives `member`, a member about to be created in the project at `revision`. */
  nameMember(member: StudioMemberNameRequest['member'], revision = this.saved.revision): Promise<StudioMemberName> {
    const run = this.transport.nameMember;
    if (!run) throw new Error('This editor has no Service to name members.');
    return run(this.saved.project_id, { revision, member }, this.signal);
  }

  /**
   * Rename a document or step, or duplicate a document, on the server as one Undo entry.
   * `use` may then return authored patches computed against the command's result, such
   * as pointing a step at the duplicate, which join the same entry.
   */
  memberCommand(
    name: string, command: StudioMemberCommand['command'], reviewed: StudioProjectRevision,
    use?: (result: StudioMemberCommandResult) => Promise<readonly StudioProjectFilePatch[]>,
  ): Promise<StudioMemberCommandResult> {
    return this.exclusiveTransaction(async (before) => {
      const run = this.transport.memberCommand;
      if (!run || !versionsMatch(before, reviewed)) throw new ProjectEditConflictError();
      const result = await run(before.project_id, {
        revision: before.revision, presentation_revision: before.presentation_revision, command,
      }, this.signal);
      return { ...result, snapshot: await this.recordWithFollowing(name, before, result.snapshot, use && (() => use(result))) };
    });
  }

  /**
   * Runs a route that publishes a new revision of the project when it succeeds, such as an import, and records what it
   * published as one Undo entry. An outcome without a snapshot published nothing.
   */
  publishingCommand<T>(
    name: string, run: (before: StudioProjectSnapshot, signal: AbortSignal) => Promise<{ outcome: T; snapshot?: StudioProjectSnapshot }>,
  ): Promise<T> {
    return this.exclusiveTransaction(async (before) => {
      const { outcome, snapshot } = await run(before, this.signal);
      if (snapshot) await this.recordWithFollowing(name, before, snapshot);
      return outcome;
    });
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
        const ambiguous = isServiceFailure(error);
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

  /** The Service says the saved plan may have changed: its facts, such as a run's status, are read again. */
  planChanged() {
    this.planVersion += 1;
    this.publish();
  }

  connected() {
    if (this.abort.signal.aborted) return;
    void this.sequence(async () => {
      const retry = this.queued.length > 0 && isServiceFailure(this.error);
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
        for (const pending of this.queued.splice(0)) pending.settle({ status: 'dropped' });
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
            if (files.every(({ address }) => sameAuthoredFile(this.draft, disk, address))) {
              pending.settle(savedEdit);
              continue;
            }
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
            pending.settle(savedEdit);
            continue;
          }
          const base = project;
          const applied = applyAuthoredEdits(base, address, patches, sameFile ? undefined : pending.base);
          if (!applied.inverse.length) {
            pending.settle(savedEdit);
            continue;
          }
          pending.gesture.validate?.(base, applied.project);
          const edits = [{ address, patches: applied.edits }];
          const adjusted = sameFile ? gesture : { ...gesture, files: edits, inverse: [{ address, patches: applied.inverse }] };
          const direction = sameFile ? pending.direction : 'edit';
          project = applied.project;
          history = moveHistory(history, adjusted, direction);
          rebased.push({ base, gesture: adjusted, direction, files: edits, historyAfter: history, settle: pending.settle });
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
    const { presentation: _, settings, ...draft } = this.draft;
    return {
      saved: this.saved,
      configuration: settings === undefined ? this.document.configuration : { ...this.document.configuration, settings },
      draft,
      pending: this.queued.length,
      saving: this.saving || this.transactionWriting,
      error: this.error,
      outside: this.outside?.snapshot ?? null,
      undoName: this.history.undo[this.history.undo.length - 1]?.name ?? null,
      redoName: this.history.redo[this.history.redo.length - 1]?.name ?? null,
      planVersion: this.planVersion,
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
    let settle!: PendingEdit['settle'];
    const settlement = new Promise<EditSettlement>((resolve) => { settle = resolve; });
    this.queued.push({ base: this.draft, gesture, direction, files, historyAfter, settle });
    this.draft = project;
    this.history = historyAfter;
    this.publish();
    this.scheduleSettle();
    return settlement;
  }

  private scheduleSettle() {
    this.cancelSettle();
    this.settle = setTimeout(() => { void this.settleQueue(); }, this.settleMs);
  }

  /** The settle timer's drain. Its outcomes are already delivered, so only an undelivered failure propagates. */
  private async settleQueue() {
    try {
      await this.flush();
    } catch (error) {
      const delivered = isRefusal(error) || error === this.error || this.abort.signal.aborted ||
        error instanceof ProjectEditConflictError && this.outside;
      if (!delivered) throw error;
    }
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
    const added = created && isAcknowledgedAddition(this.saved, latest, unchanged, [created]);
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
    const sameAuthoredFiles = declarationUnchanged && unchanged.has(declarationKey) &&
      latest.semantic_root === this.saved.semantic_root &&
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
      const gesture = changedFiles(studioName('outsideChange'), burst.before, latest);
      if (!gesture) {
        this.adopt(document, withoutChangedFiles(this.savedHistory, unchanged, declarationUnchanged));
        return;
      }
      const kept = sameAuthoredFile(editContent(burst.before), editContent(latest), declarationAddress)
        ? burst.history : withoutSettingsEdits(burst.history);
      const history = gesture.undo.length ? moveHistory(kept, gesture, 'edit') : kept;
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
      if (!exact || !isServiceFailure(error)) throw error;
      const observed = await this.read().catch(() => undefined);
      if (!observed) throw error;
      const files = 'files' in body ? body.files : [body];
      const before = this.saved;
      const latest = observed.snapshot;
      const changed = new Set(files.map(({ address }) => addressKey(address)));
      const unchanged = unchangedFiles(before, latest);
      const intended = applyFiles(editContent(before), files).project;
      if (before.file_revisions.length !== latest.file_revisions.length ||
          (!unchanged.has(declarationKey) && !changed.has(declarationKey)) ||
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
        for (const pending of this.queued.splice(0, batch.length)) pending.settle(savedEdit);
        if (!this.queued.length) this.draft = editContent(latest);
        this.publish();
      }
      this.cancelSettle();
    } catch (error) {
      if (isRefusal(error)) {
        // A refused save cannot succeed on retry: it and the edits built on it are refused, each to its own gesture.
        this.cancelSettle();
        for (const pending of this.queued.splice(0)) pending.settle({ status: 'refused', error });
        this.draft = editContent(this.saved);
        this.history = this.savedHistory;
      } else {
        this.error = error;
        if (isHttpConflict(error)) await this.refresh();
      }
      throw error;
    } finally {
      this.saving = false;
      this.publish();
    }
  }
}
