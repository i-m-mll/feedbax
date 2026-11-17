import { expect, it } from 'vitest';
import type { Step } from '@/generated/studioContracts';
import { applyAuthoredEdits, type ProjectEditContent } from './authoredEdits';
import {
  editableCheckpointRule, evaluateSubject, evaluateSubjectEdits, formatCheckpointRule, parseCheckpointRule,
  type EvaluateSubject,
} from './evaluateAuthoring';

it('keeps explicit update order and distinguishes a series from one checkpoint', () => {
  expect(parseCheckpointRule('14, 7, 21')).toEqual({ rule: [{ at: 14 }, { at: 7 }, { at: 21 }] });
  expect(parseCheckpointRule('14')).toEqual({ rule: [{ at: 14 }] });
  expect(parseCheckpointRule('at 14')).toEqual({ rule: { at: 14 } });
  expect(parseCheckpointRule('last')).toEqual({ rule: 'last' });
});

it('keeps open and bounded cadences as authored rules', () => {
  expect(parseCheckpointRule('every 7')).toEqual({ rule: { every: 7 } });
  const bounded = { every: 7, from: 8, through: 21 };
  expect(parseCheckpointRule(formatCheckpointRule(bounded))).toEqual({ rule: bounded });
  expect(editableCheckpointRule({ every_k: 7, through: 21 })).toBeUndefined();
  expect(editableCheckpointRule({ batches: [14, 7] })).toBeUndefined();
});

it('rejects duplicates and invalid numbers without expanding a cadence', () => {
  expect(parseCheckpointRule('7, 14, 7')).toHaveProperty('reason');
  expect(parseCheckpointRule('7, 0')).toHaveProperty('reason');
  expect(parseCheckpointRule('at 9007199254740992')).toHaveProperty('reason');
  expect(parseCheckpointRule('every 0')).toHaveProperty('reason');
  expect(parseCheckpointRule('every 7 from 21 through 14')).toHaveProperty('reason');
  expect(parseCheckpointRule('every 1 through 9007199254740991')).toHaveProperty('rule');
});

it('moves an Evaluate subject between a checkpoint and an initial Model and back without residue', () => {
  const evaluate: Step = {
    kind: 'evaluate',
    base: { inline: { evaluation_type: 'feedbax.rollout', params: { trial_set: { kind: 'every_condition' } } } },
    inputs: { subject: { step: 'train', checkpoint: 'last' } },
    sweep: { axes: { subject: { input: 'subject' } } },
  };
  const apply = (step: Step, next: EvaluateSubject) => {
    const project = { steps: { evaluate: step } } as unknown as ProjectEditContent;
    return (applyAuthoredEdits(project, { kind: 'step', id: 'evaluate' }, evaluateSubjectEdits(step, next)).project
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
