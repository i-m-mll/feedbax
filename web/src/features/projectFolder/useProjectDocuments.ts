import { useState, useSyncExternalStore } from 'react';
import { importLoadedModel } from '@/api/projectFolders';
import { catalogEntries, readStudioTrialAuthoring, useAuthoringCatalog } from '@/api/studioAuthoring';
import { landsOnGesture } from '@/api/request';
import { shownError } from '@/api/shownErrors';
import {
  studioAccessibleName, studioName, type StudioNameId,
} from '@/components/activity/studioVocabulary';
import type { DocumentPickerModel, PickerTemplate } from '@/features/projectWindow/DocumentPicker';
import {
  LoadedModelDocumentSchema, blankStoredModel, blankStoredObjective, blankStoredTask, loadedModelDocumentSchemaId,
  analysisDocumentSchemaId, modelDocumentSchemaId, objectiveDocumentSchemaId, taskDocumentSchemaId,
} from '@/generated/studioContracts';
import type {
  Document, RootDocument, StudioProjectFilePatch, StudioSavedTrainKey, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import type { ProjectEditController } from './projectEditController';
import { baseOf, documentPayload } from './documentPayload';
import { importProblems } from './loadedModels';
import { labelOf, type ShownDocuments } from './memberLabels';

export type DocumentRole = 'model' | 'loaded_model' | 'task' | 'objective' | 'training' | 'analysis' | 'report';
/** The roles whose documents are written in Studio, as opposed to imported. */
type AuthoredRole = Exclude<DocumentRole, 'loaded_model'>;

const supervisedTrainMethod = 'feedbax.supervised_gradient';
const pythonTask = 'feedbax.spec.python_task';

export const blankDocuments = {
  model: { schema: modelDocumentSchemaId, payload: blankStoredModel },
  task: { schema: taskDocumentSchemaId, payload: blankStoredTask },
  objective: { schema: objectiveDocumentSchemaId, payload: blankStoredObjective },
  training: {
    schema: 'feedbax.spec.training_document',
    payload: {
      method: supervisedTrainMethod,
      settings: {},
      training_config: { n_batches: 1000, batch_size: 64 },
    },
  },
  analysis: {
    schema: analysisDocumentSchemaId,
    payload: {
      schema_version: analysisDocumentSchemaId, inputs: { states: { kind: 'dataset' } },
    },
  },
  report: {
    schema: 'feedbax.report_document.v1',
    payload: { title: 'Report', blocks: [] },
  },
} satisfies Record<AuthoredRole, UnnamedDocument>;

const schemas: Record<DocumentRole, readonly string[]> = {
  model: [blankDocuments.model.schema],
  loaded_model: [loadedModelDocumentSchemaId],
  task: [blankDocuments.task.schema, pythonTask],
  objective: [blankDocuments.objective.schema],
  training: [blankDocuments.training.schema],
  analysis: [blankDocuments.analysis.schema],
  report: [blankDocuments.report.schema],
};

const labels: Record<DocumentRole, StudioNameId> = {
  model: 'modelDocument', loaded_model: 'loadedModelDocument', task: 'taskDocument', objective: 'objectiveDocument', training: 'trainingDocument',
  analysis: 'analysisDocument', report: 'reportDocument',
};

export interface ProjectDocumentInputs {
  readonly editor: ProjectEditController;
  /** The documents the workspace shows. */
  readonly documents: ShownDocuments;
  readonly stepId?: string;
  /** The row whose definition a Train or Evaluate step's document choice is resolved against. */
  readonly run?: StudioSavedTrainKey;
  readonly scope?: StudioTrialAuthoringRequest['scope'];
  /** The roles in which the step may point at another document, as opposed to editing the one it has. */
  readonly selectable: ReadonlySet<DocumentRole>;
  readonly editable: boolean;
  readonly onError: (error: unknown) => void;
  /** Shows a Loaded Model document just imported. */
  readonly onLoaded?: (document: string) => void;
}

/** The role whose documents `document` is among. */
export function roleOf(document: Pick<Document, 'schema'>) {
  return (Object.keys(schemas) as DocumentRole[]).find((role) => schemas[role].includes(document.schema));
}

/** A document Studio is about to create, with the label it is offered under when it has one. */
export type UnnamedDocument = Omit<RootDocument, 'label'> & { readonly label?: string };

/** `document` as a new file of the project at `revision`, keyed and labelled by the Service from its own label if any. */
export async function namedDocument(editor: ProjectEditController, document: UnnamedDocument, revision?: string) {
  const { id, label } = await editor.nameMember({
    kind: 'document', schema_id: document.schema, ...document.label !== undefined && { label: document.label },
  }, revision);
  return { id, document: { ...document, label } as Document };
}

/** The project's documents a step can read in `role`, by name. */
export function documentsOfRole(project: { documents?: Readonly<Record<string, Document>> | null }, role: DocumentRole) {
  const all = project.documents ?? {};
  return Object.keys(all).filter((id) => schemas[role].includes(all[id].schema)).sort();
}

/** `name` and the documents based on it, to any depth. */
export function documentFamily(project: { documents?: Readonly<Record<string, Document>> | null }, name: string) {
  const all = Object.keys(project.documents ?? {});
  const family = new Set([name]);
  for (let grew = true; grew;) {
    grew = false;
    for (const id of all) {
      const base = baseOf(project.documents, id);
      if (!family.has(id) && base !== undefined && family.has(base)) {
        family.add(id);
        grew = true;
      }
    }
  }
  return family;
}

/** The Model document a Loaded Model document fits, by key. */
export function fittedModel(document: Document | undefined) {
  if (document?.schema !== loadedModelDocumentSchemaId || !('payload' in document)) return undefined;
  const parsed = LoadedModelDocumentSchema.safeParse(document.payload);
  return parsed.success && !parsed.data.model.unresolved ? parsed.data.model.document : undefined;
}

/**
 * The documents of `role`, each followed by the documents based on it, with its depth below its root. A Model is also
 * followed by the Loaded Models that fit it.
 */
export function documentTree(project: { documents?: Readonly<Record<string, Document>> | null }, role: DocumentRole) {
  return documentTreeOf(project, role === 'model' ? [...schemas.model, ...schemas.loaded_model] : schemas[role]);
}

/**
 * The documents of the `schemas`, each followed by the documents based on it, or that fit it, with its depth below its
 * root.
 */
export function documentTreeOf(project: { documents?: Readonly<Record<string, Document>> | null }, schemas: readonly string[]) {
  const all = project.documents ?? {};
  const named = (key: string) => all[key].label;
  /** Ordered by the name each is shown by, as its key is hidden behind its name. */
  const names = Object.keys(all).filter((id) => schemas.includes(all[id].schema))
    .sort((left, right) => named(left).localeCompare(named(right), undefined, { numeric: true }) || left.localeCompare(right));
  const parent = (name: string) => {
    const base = baseOf(all, name) ?? fittedModel(all[name]);
    return base !== undefined && names.includes(base) ? base : undefined;
  };
  const tree: { key: string; name: string; depth: number; python?: boolean; loaded?: boolean }[] = [];
  const visit = (name: string, depth: number) => {
    const { schema } = all[name];
    tree.push({
      key: name, name: named(name), depth, python: schema === pythonTask || undefined,
      loaded: schema === loadedModelDocumentSchemaId || undefined,
    });
    for (const child of names) if (parent(child) === name) visit(child, depth + 1);
  };
  for (const name of names) if (parent(name) === undefined) visit(name, 0);
  return tree;
}

export function useProjectDocuments({ editor, documents: shown, stepId, run, scope, selectable, editable, onError, onLoaded }: ProjectDocumentInputs) {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const projectId = saved.project_id;
  const catalog = useAuthoringCatalog().data;
  /** What a new task starts from beside Blank: the shipped example tasks, then the installed Python task types. */
  const taskTemplates = new Map<string, { template: PickerTemplate; document: UnnamedDocument }>([
    ...(catalog?.tasks.documents ?? []).map(({ name, document }) => [`example:${name}`, {
      template: { key: `example:${name}`, name: document.label }, document,
    }] as const),
    ...catalogEntries(catalog, 'python_tasks').map(({ key, label }) => [`python:${key}`, {
      template: { key: `python:${key}`, name: label, python: true },
      document: { schema: pythonTask, label, payload: { type: key, params: {} } },
    }] as const),
  ]);
  /** The document a step did not take in a role, and why: it lacks a parameter the step sweeps. */
  const [refused, setRefused] = useState<{ step: string; role: DocumentRole; reason: string }>();
  const refuse = (role: DocumentRole, reason?: string) => setRefused(reason && stepId ? { step: stepId, role, reason } : undefined);
  /** Why the last archive loaded against a Model was not taken: every problem the service found. */
  const [unfit, setUnfit] = useState<{ model: string; problems: readonly string[] }>();

  /** Imports `archive` as a Loaded Model of `model`, named after its file, then shows it. */
  async function load(model: string, archive: File) {
    setUnfit(undefined);
    try {
      const outcome = await editor.publishingCommand(studioAccessibleName('load', studioName('loaded_model')), async (before, signal) => {
        const { id: name, label } = await editor.nameMember({
          kind: 'document', schema_id: loadedModelDocumentSchemaId, label: archive.name.replace(/\.[^.]*$/, '') || undefined,
        }, before.revision);
        const result = await importLoadedModel(projectId, { revision: before.revision, name, label, model }, archive, signal);
        return { outcome: result, snapshot: result.status === 'imported' ? result.snapshot : undefined };
      });
      if (outcome.status === 'imported') onLoaded?.(outcome.document);
      else setUnfit({ model, problems: importProblems(outcome) });
    } catch (error) {
      if (landsOnGesture(error)) setUnfit({ model, problems: [shownError(error)] });
      else onError(error);
    }
  }

  async function use(role: DocumentRole, document: string, revision: string): Promise<StudioProjectFilePatch[]> {
    if (!stepId) return [];
    if (role === 'analysis' || role === 'report') {
      return [{ address: { kind: 'step', id: stepId }, patches: [{ op: 'replace', path: '/base/document', value: document }] }];
    }
    if (!run || role === 'loaded_model') return [];
    const result = await readStudioTrialAuthoring(projectId, {
      revision, run, scope, change: { kind: 'select_document', role, document },
    }, editor.signal);
    return result.files;
  }

  /** Runs after the edits before it have saved. A refusal lands on the role's picker; any other failure goes to `onError`. */
  async function attempt<T>(role: DocumentRole, work: () => Promise<T>) {
    refuse(role);
    try {
      return await editor.command(work);
    } catch (error) {
      if (landsOnGesture(error)) refuse(role, shownError(error));
      else onError(error);
    }
  }

  async function point(name: string, role: DocumentRole, document: string) {
    const { saved } = editor.getSnapshot();
    const files = await use(role, document, saved.revision);
    if (!files.length) return;
    editor.editFiles(name, files, {
      reviewedRevision: saved.revision,
      dependencies: [...stepId ? [{ kind: 'step' as const, id: stepId }] : [], { kind: 'document', id: document }],
    });
    await editor.flush();
  }

  return (role: DocumentRole, document: string | undefined, sharing?: { lit: ReadonlySet<string>; users: ReadonlySet<string> }): DocumentPickerModel | undefined => {
    // A view still showing a revision the workspace has moved past can read a document no shown revision names; it gets
    // its picker once it shows the current one.
    if (!document || !(document in shown)) return undefined;
    const count = sharing && { lit: sharing.lit.size, users: sharing.users.size };
    const documents = documentTree(saved.project, role);
    const based = 'base' in (saved.project.documents?.[document] ?? {});
    /** Duplicate and Detach write the document's composed content, which a base chain that does not compose lacks. */
    const composes = documentPayload(saved.project.documents, saved.composed, document) !== undefined;
    const fitted = fittedModel(saved.project.documents?.[document]);
    const shownRole = fitted === undefined ? role : 'loaded_model';
    const kind = studioName(shownRole);
    const repoint = editable && !!stepId && selectable.has(role);
    const model = fitted ?? document;
    return {
      label: labels[shownRole], document, name: labelOf(shown, document), documents, count,
      refusal: refused?.step === stepId && refused?.role === role ? refused.reason : undefined,
      load: role === 'model' && editable ? (archive) => { void load(model, archive); } : undefined,
      unfit: unfit?.model === model ? { problems: unfit.problems, dismiss: () => setUnfit(undefined) } : undefined,
      choose: repoint ? (next) => { void attempt(role, () => point(studioAccessibleName('select', kind), role, next)); } : undefined,
      templates: role === 'task' ? [...taskTemplates.values()].map(({ template }) => template) : undefined,
      create: repoint && role !== 'loaded_model' ? (template) => attempt(role, async () => {
        const { saved } = editor.getSnapshot();
        const { id, document } = await namedDocument(editor,
          (template && taskTemplates.get(template)?.document) || blankDocuments[role], saved.revision);
        await editor.createFiles(studioAccessibleName('new', kind), [{ kind: 'document', id, document }],
          saved.revision, (snapshot) => use(role, id, snapshot.revision));
        return id;
      }) : undefined,
      duplicate: repoint && composes && fitted === undefined ? () => attempt(role, async () => {
        const result = await editor.memberCommand(studioAccessibleName('duplicate', kind),
          { kind: 'duplicate', family: 'documents', member: document }, editor.getSnapshot().saved,
          (result) => use(role, result.member, result.snapshot.revision));
        return result.member;
      }) : undefined,
      rename: editable ? (label) => {
        void attempt(role, () => editor.memberCommand(studioAccessibleName('rename', kind),
          { kind: 'rename', family: 'documents', member: document, label }, editor.getSnapshot().saved));
      } : undefined,
      changeKey: editable ? {
        commit: (key) => {
          void attempt(role, () => editor.memberCommand(studioAccessibleName('changeKey', kind),
            { kind: 'change_key', family: 'documents', member: document, key }, editor.getSnapshot().saved));
        },
        held: new Set(Object.keys(saved.project.documents ?? {})),
      } : undefined,
      detach: editable && based && composes ? () => {
        void attempt(role, () => editor.memberCommand(studioAccessibleName('detach', kind),
          { kind: 'detach', member: document }, editor.getSnapshot().saved));
      } : undefined,
    };
  };
}
