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

it('holds an empty draft without a reason and puts an invalid one on its field', () => {
  expect(checkpointDraftRule({ kind: 'at', at: null })).toEqual({});
  expect(checkpointDraftRule({ kind: 'every', every: null, from: 3, through: null })).toEqual({});
  expect(checkpointDraftRule({ kind: 'list', updates: '7, 14, 7' })).toMatchObject({ field: 'updates' });
  expect(checkpointDraftRule({ kind: 'list', updates: '7, 0' })).toMatchObject({ field: 'updates' });
  expect(checkpointDraftRule({ kind: 'list', updates: '7 14' })).toMatchObject({ field: 'updates' });
  expect(checkpointDraftRule({ kind: 'at', at: 9007199254740992 })).toMatchObject({ field: 'at' });
  expect(checkpointDraftRule({ kind: 'every', every: 0, from: null, through: null })).toMatchObject({ field: 'every' });
  expect(checkpointDraftRule({ kind: 'every', every: 7, from: 21, through: 14 })).toMatchObject({ field: 'through' });
  expect(checkpointDraftRule({ kind: 'every', every: 1, from: null, through: 9007199254740991 })).toHaveProperty('rule');
});

it('moves an Evaluate subject between a checkpoint and an initial Model and back without residue', () => {
  const evaluate: Step = {
    kind: 'evaluate',
    base: { inline: { evaluation_type: 'feedbax.rollout', params: { trial_set: { kind: 'every_condition' } } } },
    sweep: { axes: { subject: { input: 'subject' } } },
    inputs: { subject: { step: 'train', checkpoint: 'last' } },
  };
  const apply = (step: Step, next: EvaluateSubject) => {
    const project = { steps: { evaluate: step } } as unknown as ProjectEditContent;
    return (applyAuthoredEdits(project, { kind: 'step', id: 'evaluate' }, evaluateSubjectEdits(step, next), true).project
      .steps as Record<string, Step>).evaluate;
  };
  const initial = apply(evaluate, { kind: 'document', document: 'arm', task: 'reach', objective: 'loss' });
  expect(initial.inputs).toEqual({
    subject: { kind: 'initial_model', source: { document: 'arm' } },
    task: { kind: 'task', source: { document: 'reach' } },
    objective: { kind: 'objective', source: { document: 'loss' } },
  });
  expect(initial.sweep).toBeUndefined();
  expect(evaluateSubject(initial)).toEqual({ kind: 'document', document: 'arm', task: 'reach', objective: 'loss' });
  expect(apply(initial, { kind: 'step', step: 'train' })).toEqual(evaluate);
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

// Catches a checkpoint past the subject's training reaching a job.
it('refuses an update past the fewest batches of the runs read, on its field', () => {
  const limit = { step: 'train-2', batches: 8000 };
  expect(checkpointDraftRule({ kind: 'at', at: 8000 }, limit)).toEqual({ rule: { at: 8000 } });
  expect(checkpointDraftRule({ kind: 'at', at: 9000 }, limit)).toMatchObject({ field: 'at', reason: expect.stringContaining('8000') });
  expect(checkpointDraftRule({ kind: 'list', updates: '10, 9000' }, limit)).toMatchObject({ field: 'updates' });
  expect(checkpointDraftRule({ kind: 'last' }, limit)).toEqual({ rule: 'last' });
});
