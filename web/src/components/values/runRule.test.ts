import { expect, test } from 'vitest';
import { ruleCompletions, type RuleVocabulary } from './runRule';

const vocabulary: RuleVocabulary = { axes: { seed: ['0', '1', '2'], lr: ['0.01', '0.1'] }, metrics: ['loss', 'objective/total'] };
const offered = (rule: string, caret = rule.length, words = vocabulary) => ruleCompletions(rule, caret, words).map(({ text }) => text);

test('a term starts with an axis or, where the rule reads results, a declared metric or a rank', () => {
  expect(offered('')).toEqual(['seed', 'lr', 'loss', '"objective/total"', 'lowest', 'highest']);
  expect(offered('l')).toEqual(['lr', 'loss', 'lowest']);
  expect(offered('', 0, { axes: vocabulary.axes })).toEqual(['seed', 'lr']);
});

test('a metric offers the operators that order it, and no values', () => {
  expect(offered('seed=0 loss')).toEqual(['seed=0 loss<', 'seed=0 loss<=', 'seed=0 loss>', 'seed=0 loss>=']);
  expect(offered('loss<')).toEqual(['loss<=']);
  expect(offered('"objective/total"')).toEqual(['"objective/total"<', '"objective/total"<=', '"objective/total">', '"objective/total">=']);
  expect(offered('loss<0.')).toEqual([]);
  expect(offered('loss', 4, { axes: vocabulary.axes })).toEqual([]);
});

test('an axis offers its operators, then its values, in a list and a range', () => {
  expect(offered('seed')).toEqual(['seed=', 'seed!=', 'seed>=', 'seed<=', 'seed>', 'seed<']);
  expect(offered('seed!')).toEqual(['seed!=']);
  expect(offered('seed=')).toEqual(['seed=0', 'seed=1', 'seed=2']);
  expect(offered('seed=0,')).toEqual(['seed=0,0', 'seed=0,1', 'seed=0,2']);
  expect(offered('lr>=0.01 seed=0..')).toEqual(['lr>=0.01 seed=0..0', 'lr>=0.01 seed=0..1', 'lr>=0.01 seed=0..2']);
});

test('lowest and highest take a result name, quoted when it is not a rule name, then per and axes', () => {
  expect(offered('lowest 1 ')).toEqual(['lowest 1 loss', 'lowest 1 "objective/total"']);
  expect(offered('lowest 1 "obj')).toEqual(['lowest 1 "objective/total"']);
  expect(offered('lowest 1 loss p')).toEqual(['lowest 1 loss per']);
  expect(offered('lowest 1 loss per ')).toEqual(['lowest 1 loss per seed', 'lowest 1 loss per lr']);
});

test('a completion inside the rule keeps what follows the caret', () => {
  const [first] = ruleCompletions('se lr>=0.1', 2, vocabulary);
  expect(first).toEqual({ text: 'seed lr>=0.1', caret: 4, label: 'seed' });
});
