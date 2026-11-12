import { defaultTaskSpec } from '@/features/scenario/defaultTaskSpec';
import type { CreateNamedDocument, Document, StudioProjectSnapshot } from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

const model: Document = { schema: 'feedbax.spec.graph', payload: { nodes: {}, wires: [] } };
const task: Document = {
  schema: 'feedbax.spec.task',
  payload: { type: defaultTaskSpec.type, params: defaultTaskSpec.params },
};
const methodSchema = 'feedbax.spec.training_method.studio_graph';

export function canCreateFirstTrain(project: StudioProjectSnapshot['project']) {
  return !project.sweep && Object.keys(project.steps ?? {}).length === 0;
}

async function ensureDocument(
  editor: ProjectEditController, saved: StudioProjectSnapshot, name: string, document: Document,
) {
  let id = name;
  for (let number = 2; Object.prototype.hasOwnProperty.call(saved.project.documents, id); number += 1) {
    if (containsCreatedFile(saved.project, { kind: 'document', id, document })) return { id, saved };
    id = `${name}-${number}`;
  }
  const file: CreateNamedDocument = { kind: 'document', id, document };
  const result = await editor.createFile(file, saved.revision);
  return { id, saved: result.snapshot };
}

export async function createFirstTrain(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision || !canCreateFirstTrain(saved.project)) {
    throw new ProjectEditConflictError();
  }

  // Publish the step after both references are durable. A retry can reuse the starter documents.
  const graph = await ensureDocument(editor, saved, 'model', model);
  const taskDocument = await ensureDocument(editor, graph.saved, 'task', task);
  const id = 'train';
  const nBatches = 1000;
  const learningRate = 0.001;
  await editor.createFile({
    kind: 'step', id,
    step: {
      kind: 'train',
      base: { inline: {
        graph: { document: graph.id },
        task: { document: taskDocument.id },
        task_binding_spec: {},
        objective: { kind: 'loss_term', loss: { type: 'Composite', label: 'Loss', children: {} } },
        training_config: {
          n_batches: nBatches,
          batch_size: 64,
          learning_rate: learningRate,
          n_reach_steps: defaultTaskSpec.params.n_steps,
        },
        method_ref: { package: 'feedbax', name: 'studio_graph', version: 'v1' },
        method_payload: {
          schema_id: methodSchema,
          schema_version: `${methodSchema}.v1`,
          payload: {
            optimizer: { type: 'adam', params: { learning_rate: learningRate } },
            target_total: nBatches,
            key_schedule: 'independent_v1',
          },
        },
      } },
    },
  }, taskDocument.saved.revision);
  return id;
}
