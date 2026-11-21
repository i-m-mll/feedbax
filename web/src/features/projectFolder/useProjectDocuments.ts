import { useState, useSyncExternalStore } from 'react';
import { readStudioTrialAuthoring } from '@/api/studioAuthoring';
import { apiErrorMessage, landsOnGesture } from '@/api/request';
import {
  studioAccessibleName, studioName, studioSweptParameterMissing, type StudioNameId,
} from '@/components/activity/studioVocabulary';
import { shownName } from '@/features/pipeline/planView';
import type { DocumentPickerModel } from '@/features/projectWindow/DocumentPicker';
import type {
  Document, StudioProjectFilePatch, StudioSavedTrainKey, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import type { ProjectEditController } from './projectEditController';
import { generatedName, holds } from './generatedName';

export type DocumentRole = 'model' | 'task' | 'objective' | 'training' | 'analysis' | 'report';

const supervisedTrainMethod = 'feedbax.supervised_gradient';

export const blankDocuments = {
  model: { schema: 'feedbax.spec.model', payload: { step_seconds: 0.01, ports: {}, interior: {} } },
  task: { schema: 'feedbax.spec.python_task', payload: { type: 'feedbax.Reaching', params: {} } },
  objective: { schema: 'feedbax.spec.objective_document', payload: { id: 'objective', terms: [] } },
  training: {
    schema: 'feedbax.spec.training_document',
    payload: {
      method: supervisedTrainMethod,
      settings: { optimizer: { type: 'adam', learning_rate: { type: 'constant_schedule', value: 0.001 } } },
      training_config: { n_batches: 1000, batch_size: 64 },
    },
  },
  analysis: {
    schema: 'feedbax.analysis_document.v1',
    payload: {
      schema_version: 'feedbax.analysis_document.v1', inputs: { states: { kind: 'dataset' } }, operations: {}, views: {},
    },
  },
  report: {
    schema: 'feedbax.report_document.v1',
    payload: { schema_version: 'feedbax.report_document.v1', title: 'Report', blocks: [] },
  },
} satisfies Record<DocumentRole, Document>;

const schemas: Record<DocumentRole, readonly string[]> = {
  model: [blankDocuments.model.schema],
  task: [blankDocuments.task.schema, 'feedbax.spec.task_document'],
  objective: [blankDocuments.objective.schema],
  training: [blankDocuments.training.schema],
  analysis: [blankDocuments.analysis.schema],
  report: [blankDocuments.report.schema],
};

const labels: Record<DocumentRole, StudioNameId> = {
  model: 'modelDocument', task: 'taskDocument', objective: 'objectiveDocument', training: 'trainingDocument',
  analysis: 'analysisDocument', report: 'reportDocument',
};

export interface ProjectDocumentInputs {
  readonly editor: ProjectEditController;
  readonly stepId?: string;
  /** The row whose definition a Train or Evaluate step's document choice is resolved against. */
  readonly run?: StudioSavedTrainKey;
  readonly scope?: StudioTrialAuthoringRequest['scope'];
  /** The roles in which the step may point at another document, as opposed to editing the one it has. */
  readonly selectable: ReadonlySet<DocumentRole>;
  readonly editable: boolean;
  /** The based documents whose Detach would leave an Evaluate step's Model document unrelated to its subject's. */
  readonly detachRefused?: readonly string[];
  readonly onError: (message: string) => void;
}

/** The project's documents a step can read in `role`, by name. */
export function documentsOfRole(project: { documents?: Readonly<Record<string, Document>> | null }, role: DocumentRole) {
  const all = project.documents ?? {};
  return Object.keys(all).filter((id) => schemas[role].includes(all[id].schema)).sort();
}

/** `name` and the documents based on it, to any depth. */
export function documentFamily(project: { documents?: Readonly<Record<string, Document>> | null }, name: string) {
  const all = Object.entries(project.documents ?? {});
  const family = new Set([name]);
  for (let grew = true; grew;) {
    grew = false;
    for (const [id, document] of all) {
      if (!family.has(id) && 'base' in document && family.has(document.base.document)) {
        family.add(id);
        grew = true;
      }
    }
  }
  return family;
}

/** The documents of `role`, each followed by the documents based on it, with its depth below its root. */
export function documentTree(project: { documents?: Readonly<Record<string, Document>> | null }, role: DocumentRole) {
  return documentTreeOf(project, schemas[role]);
}

/** The documents of the `schemas`, each followed by the documents based on it, with its depth below its root. */
export function documentTreeOf(project: { documents?: Readonly<Record<string, Document>> | null }, schemas: readonly string[]) {
  const all = project.documents ?? {};
  const names = Object.keys(all).filter((id) => schemas.includes(all[id].schema)).sort();
  const baseOf = (name: string) => {
    const document = all[name];
    return 'base' in document && names.includes(document.base.document) ? document.base.document : undefined;
  };
  const tree: { key: string; name: string; depth: number }[] = [];
  const visit = (name: string, depth: number) => {
    tree.push({ key: name, name: shownName(name, all[name]), depth });
    for (const child of names) if (baseOf(child) === name) visit(child, depth + 1);
  };
  for (const name of names) if (baseOf(name) === undefined) visit(name, 0);
  return tree;
}

export function useProjectDocuments({ editor, stepId, run, scope, selectable, editable, detachRefused = [], onError }: ProjectDocumentInputs) {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const projectId = saved.project_id;
  /** The document a step did not take in a role, and why: it lacks a parameter the step sweeps. */
  const [refused, setRefused] = useState<{ step: string; role: DocumentRole; reason: string }>();
  const refuse = (role: DocumentRole, reason?: string) => setRefused(reason && stepId ? { step: stepId, role, reason } : undefined);

  async function use(role: DocumentRole, document: string, revision: string): Promise<StudioProjectFilePatch[]> {
    if (!stepId) return [];
    if (role === 'analysis' || role === 'report') {
      return [{ address: { kind: 'step', id: stepId }, patches: [{ op: 'replace', path: '/base/document', value: document }] }];
    }
    if (!run) return [];
    const result = await readStudioTrialAuthoring(projectId, {
      revision, run, scope, change: { kind: 'select_document', role, document },
    }, editor.signal);
    refuse(role, result.refusal ? studioSweptParameterMissing(result.refusal) : undefined);
    return result.files;
  }

  /** Runs after the edits before it have saved. A refusal lands on the role's picker; any other failure goes to `onError`. */
  async function attempt<T>(role: DocumentRole, work: () => Promise<T>) {
    refuse(role);
    try {
      return await editor.command(work);
    } catch (error) {
      if (landsOnGesture(error)) refuse(role, apiErrorMessage(error));
      else onError(apiErrorMessage(error));
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
    if (!document) return undefined;
    const count = sharing && { lit: sharing.lit.size, users: sharing.users.size };
    const documents = documentTree(saved.project, role);
    const based = 'base' in (saved.project.documents?.[document] ?? {});
    const kind = studioName(role);
    const repoint = editable && !!stepId && selectable.has(role);
    return {
      label: labels[role], document, name: shownName(document, saved.project.documents?.[document]), documents, count,
      refusal: refused?.step === stepId && refused?.role === role ? refused.reason : undefined,
      choose: repoint ? (next) => { void attempt(role, () => point(studioAccessibleName('select', kind), role, next)); } : undefined,
      create: repoint ? () => attempt(role, async () => {
        const { saved } = editor.getSnapshot();
        const id = generatedName(role, holds(saved.project.documents));
        await editor.createFiles(studioAccessibleName('new', kind), [{ kind: 'document', id, document: blankDocuments[role] }],
          saved.revision, (snapshot) => use(role, id, snapshot.revision));
        return id;
      }) : undefined,
      duplicate: repoint ? () => attempt(role, async () => {
        const result = await editor.memberCommand(studioAccessibleName('duplicate', kind),
          { kind: 'duplicate', family: 'documents', member: document }, editor.getSnapshot().saved,
          (result) => use(role, result.member, result.snapshot.revision));
        return result.member;
      }) : undefined,
      rename: editable ? (label) => {
        void attempt(role, () => editor.memberCommand(studioAccessibleName('rename', kind),
          { kind: 'rename', family: 'documents', member: document, label }, editor.getSnapshot().saved));
      } : undefined,
      changeKey: editable ? (key) => {
        void attempt(role, () => editor.memberCommand(studioAccessibleName('changeKey', kind),
          { kind: 'change_key', family: 'documents', member: document, key }, editor.getSnapshot().saved));
      } : undefined,
      detach: editable && based && !detachRefused.includes(document) ? () => {
        void attempt(role, () => editor.memberCommand(studioAccessibleName('detach', kind),
          { kind: 'detach', member: document }, editor.getSnapshot().saved));
      } : undefined,
    };
  };
}
