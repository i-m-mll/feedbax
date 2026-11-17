import { useSyncExternalStore } from 'react';
import { readStudioTrialAuthoring } from '@/api/studioAuthoring';
import { apiErrorMessage } from '@/api/request';
import { studioAccessibleName, studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import type { DocumentPickerModel } from '@/features/projectWindow/DocumentPicker';
import type {
  Document, StudioProjectFilePatch, StudioSavedTrainKey, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import type { ProjectEditController } from './projectEditController';

export type DocumentRole = 'model' | 'task' | 'objective' | 'analysis' | 'report';

export const blankDocuments = {
  model: { schema: 'feedbax.spec.model', payload: { step_seconds: 0.01, ports: {}, interior: {} } },
  task: { schema: 'feedbax.spec.python_task', payload: { type: 'feedbax.Reaching', params: {} } },
  objective: { schema: 'feedbax.spec.objective_document', payload: { id: 'objective', terms: [] } },
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
  analysis: [blankDocuments.analysis.schema],
  report: [blankDocuments.report.schema],
};

const labels: Record<DocumentRole, StudioNameId> = {
  model: 'modelDocument', task: 'taskDocument', objective: 'objectiveDocument',
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
  readonly onError: (message: string) => void;
}

export function freeDocumentName(taken: readonly string[], base: string) {
  let name = base;
  for (let number = 2; taken.includes(name); number += 1) name = `${base}-${number}`;
  return name;
}

export function useProjectDocuments({ editor, stepId, run, scope, selectable, editable, onError }: ProjectDocumentInputs) {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const projectId = saved.project_id;

  async function use(role: DocumentRole, document: string, revision: string): Promise<StudioProjectFilePatch[]> {
    if (!stepId) return [];
    if (role === 'analysis' || role === 'report') {
      return [{ address: { kind: 'step', id: stepId }, patches: [{ op: 'replace', path: '/base/document', value: document }] }];
    }
    if (!run) return [];
    const result = await readStudioTrialAuthoring(projectId, {
      revision, run, scope, change: { kind: 'select_document', role, document },
    }, editor.signal);
    return result.files;
  }

  async function attempt<T>(work: () => Promise<T>) {
    try {
      return await work();
    } catch (error) {
      onError(apiErrorMessage(error));
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
    const all = saved.project.documents ?? {};
    const documents = Object.keys(all).filter((id) => schemas[role].includes(all[id].schema)).sort();
    const kind = studioName(role);
    const repoint = editable && !!stepId && selectable.has(role);
    return {
      label: labels[role], document, documents, count,
      choose: repoint ? (next) => { void attempt(() => point(studioAccessibleName('select', kind), role, next)); } : undefined,
      create: repoint ? () => attempt(async () => {
        const { saved } = editor.getSnapshot();
        const id = freeDocumentName(Object.keys(saved.project.documents ?? {}), role);
        await editor.createFiles(studioAccessibleName('new', kind), [{ kind: 'document', id, document: blankDocuments[role] }],
          saved.revision, (snapshot) => use(role, id, snapshot.revision));
        return id;
      }) : undefined,
      duplicate: repoint ? () => attempt(async () => {
        const result = await editor.documentCommand(studioAccessibleName('duplicate', kind),
          { kind: 'duplicate', document }, editor.getSnapshot().saved,
          (result) => use(role, result.document, result.snapshot.revision));
        return result.document;
      }) : undefined,
      rename: editable ? (name) => {
        void attempt(() => editor.documentCommand(studioAccessibleName('rename', kind),
          { kind: 'rename', document, name }, editor.getSnapshot().saved));
      } : undefined,
    };
  };
}
