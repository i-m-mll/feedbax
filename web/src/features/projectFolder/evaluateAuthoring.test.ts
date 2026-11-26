import { expect, it } from 'vitest';
import type { Step } from '@/generated/studioContracts';
import { applyAuthoredEdits, type ProjectEditContent } from './authoredEdits';
import {
  checkpointDraft, checkpointDraftRule, conditionLabels, editableCheckpointRule, evaluateSubject, evaluateSubjectEdits,
  selectionCheckpoints,
  type EvaluateSubject,
} from './evaluateAuthoring';

it('keeps explicit update order and distinguishes a series from one checkpoint', () => {
  expect(checkpointDraftRule({ kind: 'list', updates: '14, 7, 21' })).toEqual({ rule: [{ at: 14 }, { at: 7 }, { at: 21 }] });
  expect(checkpointDraftRule({ kind: 'list', updates: '14' })).toEqual({ rule: [{ at: 14 }] });
  expect(checkpointDraftRule({ kind: 'at', at: 14 })).toEqual({ rule: { at: 14 } });
  expect(checkpointDraftRule({ kind: 'last' })).toEqual({ rule: 'last' });
});

it('keeps open and bounded cadences as authored rules', () => {
  expect(checkpointDraftRule({ kind: 'every', every: 7, from: null, through: null })).toEqual({ rule: { every: 7 } });
  const bounded = { every: 7, from: 8, through: 21 };
  expect(checkpointDraftRule(checkpointDraft(bounded))).toEqual({ rule: bounded });
  expect(editableCheckpointRule({ every_k: 7, through: 21 })).toBeUndefined();
  expect(editableCheckpointRule({ batches: [14, 7] })).toBeUndefined();
});

// Catches a list that says no rule dropped on Enter instead of saved as its text, and a number no rule takes saved.
it('saves nothing for an empty draft, a list that says no rule as its text, and an invalid number not at all', () => {
  expect(checkpointDraftRule({ kind: 'at', at: null })).toEqual({});
  expect(checkpointDraftRule({ kind: 'every', every: null, from: 3, through: null })).toEqual({});
  for (const updates of ['7, 14, 7', '7, 0', '7 14']) {
    expect(checkpointDraftRule({ kind: 'list', updates })).toMatchObject({
      rule: { kind: 'unresolved', text: updates }, problem: { field: 'updates' },
    });
  }
  expect(checkpointDraft({ kind: 'unresolved', text: '7 14' })).toEqual({ kind: 'list', updates: '7 14' });
  expect(checkpointDraftRule({ kind: 'at', at: 9007199254740992 })).toEqual({ problem: expect.objectContaining({ field: 'at' }) });
  expect(checkpointDraftRule({ kind: 'every', every: 0, from: null, through: null })).toEqual({ problem: expect.objectContaining({ field: 'every' }) });
  expect(checkpointDraftRule({ kind: 'every', every: 7, from: 21, through: 14 })).toEqual({ problem: expect.objectContaining({ field: 'through' }) });
  expect(checkpointDraftRule({ kind: 'every', every: 1, from: null, through: 9007199254740991 })).toHaveProperty('rule');
});

it('moves an Evaluate subject between a checkpoint and an initial Model and back without residue', () => {
  const evaluate: Step = {
    kind: 'evaluate', label: 'Evaluate',
    base: { inline: { evaluation_type: 'feedbax.rollout', params: { trial_set: { kind: 'every_condition' } } } },
    sweep: { axes: { subject: { input: 'subject' } } },
    inputs: { subject: { step: 'train', checkpoint: 'last' } },
  };
  const apply = (step: Step, next: EvaluateSubject) => {
    const project = { steps: { evaluate: step } } as unknown as ProjectEditContent;
    return (applyAuthoredEdits(project, { kind: 'step', id: 'evaluate' }, evaluateSubjectEdits(step, next)).project
      .steps as Record<string, Step>).evaluate;
  };
  const initial = apply(evaluate, { kind: 'document', reads: 'initial_model', document: 'arm', task: 'reach', objective: 'loss' });
  expect(initial.inputs).toEqual({
    subject: { kind: 'initial_model', source: { document: 'arm' } },
    task: { kind: 'task', source: { document: 'reach' } },
    objective: { kind: 'objective', source: { document: 'loss' } },
  });
  expect(initial.sweep).toBeUndefined();
  expect(evaluateSubject(initial)).toEqual({ kind: 'document', reads: 'initial_model', document: 'arm', task: 'reach', objective: 'loss' });
  expect(apply(initial, { kind: 'step', step: 'train', reads: 'checkpoint' })).toEqual(evaluate);
  // A Synthesize step's model is read with no checkpoint, which the planner would refuse on it.
  expect(apply(initial, { kind: 'step', step: 'synthesize', reads: 'model' }).inputs).toEqual({ subject: { step: 'synthesize' } });
  expect(apply(evaluate, { kind: 'step', step: 'synthesize', reads: 'model' }).inputs).toEqual({ subject: { step: 'synthesize' } });
});

// Catches a Loaded Model chosen as the subject but still read as an initial Model, which would evaluate untrained weights.
it('reads a Loaded Model subject with its weights, on the Task and Objective the step already declares', () => {
  const initial: Step = {
    kind: 'evaluate', label: 'Evaluate',
    base: { inline: { evaluation_type: 'feedbax.rollout', params: { trial_set: { kind: 'every_condition' } } } },
    inputs: {
      subject: { kind: 'initial_model', source: { document: 'arm' } },
      task: { kind: 'task', source: { document: 'reach' } },
    },
  };
  const project = { steps: { evaluate: initial } } as unknown as ProjectEditContent;
  const loaded = (applyAuthoredEdits(project, { kind: 'step', id: 'evaluate' }, evaluateSubjectEdits(initial, {
    kind: 'document', reads: 'loaded_model', document: 'arm-trained', task: 'reach',
  })).project.steps as Record<string, Step>).evaluate;
  expect(loaded.inputs).toEqual({
    subject: { kind: 'loaded_model', source: { document: 'arm-trained' } },
    task: { kind: 'task', source: { document: 'reach' } },
  });
  expect(evaluateSubject(loaded)).toEqual({ kind: 'document', reads: 'loaded_model', document: 'arm-trained', task: 'reach' });
});

// Catches a condition shown as its raw JSON id, or named by a variable every condition shares.
it('names each condition by the variables that vary across its cohort', () => {
  const a = '{"goal":[1,0],"origin":[0,0]}';
  const b = '{"goal":[0,1],"origin":[0,0]}';
  expect([...conditionLabels([a, b]).values()]).toEqual(['goal=[1,0]', 'goal=[0,1]']);
  expect(conditionLabels(['{"goal":[1,0]}']).get('{"goal":[1,0]}')).toBe('goal=[1,0]');
  expect(conditionLabels(['left']).get('left')).toBe('left');
});

// Catches a listed subject drawing no Checkpoint editor, or one editor hiding selections whose rules differ.
it('reads one checkpoint rule for selections that share it and one per selection otherwise', () => {
  expect(selectionCheckpoints([{ step: 'a', checkpoint: { at: 5 } }, { step: 'b', checkpoint: { at: 5 } }]))
    .toEqual([{ rule: { at: 5 }, selections: [0, 1] }]);
  expect(selectionCheckpoints([{ step: 'a', checkpoint: 'last' }, { step: 'b', checkpoint: { at: 5 } }]))
    .toEqual([{ rule: 'last', selections: [0] }, { rule: { at: 5 }, selections: [1] }]);
});

// Catches a checkpoint outside a subject's training, before its origin or past its end, reaching a job.
it('refuses an update outside the range every run read has, on its field, naming the step that bounds it', () => {
  const range = { after: { step: 'train', update: 100 }, through: { step: 'train-2', update: 8000 } };
  expect(checkpointDraftRule({ kind: 'at', at: 8000 }, range)).toEqual({ rule: { at: 8000 } });
  expect(checkpointDraftRule({ kind: 'at', at: 101 }, range)).toEqual({ rule: { at: 101 } });
  const late = checkpointDraftRule({ kind: 'at', at: 9000 }, range);
  expect(late).toMatchObject({ rule: { at: 9000 }, problem: { field: 'at', reason: expect.stringMatching(/^train-2 · .*101\b.*8000/) } });
  const early = checkpointDraftRule({ kind: 'list', updates: '100, 200' }, range);
  expect(early).toMatchObject({ rule: [{ at: 100 }, { at: 200 }], problem: { field: 'updates', reason: expect.stringMatching(/^train · .*101\b.*8000/) } });
  expect(checkpointDraftRule({ kind: 'last' }, range)).toEqual({ rule: 'last' });
});
