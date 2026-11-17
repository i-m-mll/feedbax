import { expect, it } from 'vitest';
import { editableCheckpointRule, formatCheckpointRule, parseCheckpointRule } from './evaluateAuthoring';

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
