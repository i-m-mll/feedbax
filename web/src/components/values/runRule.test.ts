import { expect, test } from 'vitest';
import { pickedTerms, ruleCompletions, valuesRule, type RuleVocabulary } from './runRule';

const vocabulary: RuleVocabulary = { axes: { seed: ['0', '1', '2'], lr: ['0.01', '0.1'] }, metrics: ['loss', 'objective/total'] };
const offered = (rule: string, caret = rule.length, words = vocabulary) => ruleCompletions(rule, caret, words).map(({ text }) => text);

test('a comparison starts with an axis, never a result name', () => {
  expect(offered('')).toEqual(['seed', 'lr', 'lowest', 'highest']);
  expect(offered('l')).toEqual(['lr', 'lowest']);
  expect(offered('', 0, { axes: vocabulary.axes })).toEqual(['seed', 'lr']);
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

test('chosen values are written as the rule writes them', () => {
  expect(valuesRule({ seed: { 'seed=0': '0', 'seed=2': '2' }, set: { a: '"a b"' } }, { set: ['a'], seed: ['seed=2', 'seed=0'] }))
    .toBe('seed=0,2 set="a b"');
});

const axes = { seed: { s0: '0', s2: '2', s10: '10' }, lr: { a: '0.01', b: '0.1' } };
const runs = Object.keys(axes.seed).flatMap((seed) => Object.keys(axes.lr).map((lr) => ({ id: `${seed}-${lr}`, values: { seed, lr } })));
const picks = (...ids: string[]) => pickedTerms(runs, new Set(ids), axes, (id) => `row ${id}`);

// Catches picked runs written as a longer rule than needed, or as a rule that chooses other runs too.
test('picked runs write the shortest rule that chooses exactly them, else their pinned rows', () => {
  expect(picks('s0-a', 's0-b', 's10-a', 's10-b')).toEqual({ rule: 'seed=0,10' });
  expect(picks('s2-b')).toEqual({ rule: 'lr=0.1 seed=2' });
  expect(picks(...runs.map(({ id }) => id))).toEqual({ rule: '' });
  expect(picks('s0-a', 's2-b')).toEqual({ rows: ['row s0-a', 'row s2-b'] });
});

// Catches a rule whose values are listed by their ids' spelling (10 before 2) rather than in the axis's order.
test('chosen values are written in their axis\'s order', () => {
  expect(valuesRule(axes, { seed: ['s10', 's2'] })).toBe('seed=2,10');
});
