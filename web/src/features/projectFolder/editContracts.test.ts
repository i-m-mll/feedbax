import { describe, expect, it } from 'vitest';
import {
  parseContract, StepSchema, type AuthoredFileAddress, type ProjectValueEdit, type Step, type StepReference,
} from '@/generated/studioContracts';
import { applyAuthoredEdits, presentationAddress, presentationHistoryEdits, removeAuthoredMember, type ProjectEditContent } from './authoredEdits';
import { evaluateSubjectEdits } from './evaluateAuthoring';
import { bindingEdit, changedSelections, connection, stepSelections, subjectAxisEdits } from './selectionEdits';

const revision = 'a'.repeat(64);
const base = { inline: {
  evaluation_type: 'feedbax.rollout',
  params: { trial_set: { kind: 'every_condition' }, seed: 0, trial_policy: 'paired', variation: [] },
} };
const subjectAxis = { subject: { input: 'subject' } };
const train: Step = { kind: 'train', label: 'Train', base: { inline: {} } } as unknown as Step;
const evaluate = (inputs: Step['inputs'], sweep?: Step['sweep']): Step =>
  ({ kind: 'evaluate', label: 'Evaluate', base, inputs, ...sweep && { sweep } }) as Step;

/** The shapes an Evaluate step's subject can have. */
const shapes: Record<string, Step> = {
  'one reference, no sweep': evaluate({ subject: { step: 'train', checkpoint: 'last' } }),
  'one reference with a subject axis': evaluate({ subject: { step: 'train', checkpoint: 'last' } }, { axes: subjectAxis }),
  'a list': evaluate({ subject: [{ step: 'train', checkpoint: 'last' }, { step: 'train_b', checkpoint: { at: 4 } }] },
    { axes: subjectAxis }),
  'a sweep without a subject axis': evaluate({ subject: [{ step: 'train', checkpoint: 'last' }, { step: 'train_b' }] },
    { axes: { mass: { path: 'payload.mass', values: [1, 2] } } }),
  'a where': evaluate({ subject: { step: 'train', checkpoint: 'last', where: {
    clauses: [{ seed: [{ id: '0', label: '0' }] }],
  } } }, { axes: subjectAxis }),
  'a named subset': evaluate({ subject: [{ step: 'train', checkpoint: 'last', where: { subset: 'best' } }, { step: 'train_b' }] },
    { axes: subjectAxis }),
  'an initial Model': evaluate({
    subject: { kind: 'initial_model', source: { document: 'arm' } },
    task: { kind: 'task', source: { document: 'reach' } },
  }),
};

function project(step: Step): ProjectEditContent {
  return { steps: { train, train_b: train, train_c: train, evaluate: step } } as unknown as ProjectEditContent;
}

/** Every edit Studio can make of the step's subject, by the builders its fields and wires use. */
function subjectEdits(step: Step): Record<string, ProjectValueEdit[]> {
  const edits: Record<string, ProjectValueEdit[]> = {
    'point at a Train step': evaluateSubjectEdits(step, { kind: 'step', step: 'train_b', reads: 'checkpoint' }),
    'point at an initial Model': evaluateSubjectEdits(step, { kind: 'document', reads: 'initial_model', document: 'arm', task: 'reach', objective: undefined }),
    'wire a Train step': connection(project(step).steps, 'train_c', 'evaluate')?.edits ?? [],
  };
  const selections: StepReference[] = stepSelections(step.inputs?.subject) ?? [];
  if (selections.length && !('kind' in (step.inputs?.subject ?? {}))) {
    const bind = (binding: ReturnType<typeof changedSelections>, fans: boolean) =>
      [bindingEdit(step.inputs, 'subject', binding), ...fans ? subjectAxisEdits(step) : []];
    edits['add a subject'] = bind(changedSelections(selections, {
      kind: 'add', reference: { step: 'train_c', checkpoint: selections[0].checkpoint ?? 'last' },
    }), true);
    edits['replace a subject'] = bind(changedSelections(selections, { kind: 'replace', index: 0, reference: { step: 'train_c' } }),
      selections.length > 1);
    if (selections.length > 1) edits['remove a subject'] = bind(changedSelections(selections, { kind: 'remove', index: 0 }), false);
    const where = removeAuthoredMember(project(step), { kind: 'step', id: 'evaluate' },
      Array.isArray(step.inputs?.subject) ? ['inputs', 'subject', '0', 'where'] : ['inputs', 'subject', 'where']);
    if (where) edits['clear a where'] = [where];
  }
  return edits;
}

function expectWritable(content: ProjectEditContent, address: AuthoredFileAddress, patches: ProjectValueEdit[]) {
  parseContract('StudioProjectPatch', { revision, address, patches });
  const { project: after, inverse } = applyAuthoredEdits(content, address, patches);
  parseContract('StudioProjectPatch', { revision, address, patches: inverse });
  return after;
}

// Catches an edit Studio refuses to send as its own defect: a value with an undefined member, NaN, or a path the
// contract does not accept, in the edit or in its Undo.
describe('every edit builder writes what its contract accepts', () => {
  for (const [shape, step] of Object.entries(shapes)) {
    for (const [gesture, patches] of Object.entries(subjectEdits(step))) {
      if (!patches.length) continue;
      it(`${gesture}, from ${shape}`, () => {
        const after = expectWritable(project(step), { kind: 'step', id: 'evaluate' }, patches);
        expect(StepSchema.safeParse(after.steps?.evaluate).error?.issues).toBeUndefined();
      });
    }
  }

  it('moves a placement and undoes it', () => {
    const before = { steps: { train: { sweep_order: ['seed'] } } };
    const after = { ...before, documents: { arm: { positions: { node: { x: 12.5, y: -3 } } } } };
    parseContract('StudioPresentationWrite', { revision, presentation: after });
    expectWritable({ presentation: after } as ProjectEditContent, presentationAddress,
      presentationHistoryEdits(after, before, after));
  });

  // Catches a Model view that cannot read one subject's runs: a subject value's ID is the Train row it reads.
  it('scopes a Model read to one value of any axis', () => {
    for (const value of ['nominal', 'seed=0', 'default/seed=0']) {
      parseContract('EvaluateModelRequest', {
        revision, run: { step: 'evaluate', variant: 'default', row: 'subject=seed=0' }, scope: { kind: 'axis_value', axis: 'subject', value },
      });
    }
  });
});

/** The file's members in the order a server applying `patches` in turn writes them: removes delete, adds append. */
function written(file: Record<string, unknown>, patches: readonly ProjectValueEdit[]) {
  const entries = new Map(Object.entries(structuredClone(file)));
  for (const patch of patches) {
    const [key, ...rest] = patch.path.slice(1).split('/');
    if (rest.length) throw new Error('A top-level member edit was expected.');
    if (patch.op === 'remove') entries.delete(key);
    else if (patch.op !== 'order') entries.set(key, patch.value);
  }
  return [...entries.keys()];
}

describe('an authored edit keeps the order of the file it writes', () => {
  const address: AuthoredFileAddress = { kind: 'document', id: 'task' };
  const content = (task: Record<string, unknown>) => ({ documents: { task } }) as unknown as ProjectEditContent;

  // Catches Undo of a removed member putting it back last, which reorders the authored file.
  it('restores a removed member where it was', () => {
    const task = { schema: 'x', reach: 1, hold: 2, rest: 3 };
    const { edits, inverse } = applyAuthoredEdits(content(task), address, [{ op: 'remove', path: '/hold' }]);
    expect(written(task, [...edits, ...inverse])).toEqual(['schema', 'reach', 'hold', 'rest']);
  });

  // Catches an edit of a value inside members keyed by number being refused, or written whole in another order.
  it('edits a value among members keyed by number', () => {
    const task = { epochs: { 2: { length: 5 }, 10: { length: 7 } } };
    const { project, edits, inverse } = applyAuthoredEdits(content(task), address,
      [{ op: 'replace', path: '/epochs', value: { 2: { length: 6 }, 10: { length: 7 } } }]);
    expect(edits).toEqual([{ op: 'replace', path: '/epochs/2', value: { length: 6 } }]);
    expect(applyAuthoredEdits(project, address, inverse).project).toEqual(content(task));
  });
  // Catches a move among keyed members writing the whole collection, or Undo of a move leaving the new order.
  it('moves a keyed member with one order edit, and Undo puts it back with another', () => {
    const terms = { reach: { weight: 1 }, hold: { weight: 2 }, rest: { weight: 3 } };
    const objective = { id: 'objective', terms };
    const order: ProjectValueEdit = { op: 'order', path: '/terms', order: ['rest', 'reach'] };
    const { project, edits, inverse } = applyAuthoredEdits(content(objective), address, [order]);
    const moved = (project as unknown as { documents: { task: typeof objective } }).documents.task.terms;
    expect(Object.keys(moved)).toEqual(['rest', 'reach', 'hold']);
    expect(moved).toEqual(terms);
    expect(edits).toEqual([order]);
    expect(inverse).toEqual([{ op: 'order', path: '/terms', order: ['reach', 'hold', 'rest'] }]);
    const undone = applyAuthoredEdits(project, address, inverse).project as unknown as { documents: { task: typeof objective } };
    expect(Object.keys(undone.documents.task.terms)).toEqual(['reach', 'hold', 'rest']);
  });
});
