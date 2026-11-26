import { expect, it } from 'vitest';
import { completions, valueKind, type FieldModel } from './fieldModel';

const model = {
  names: [{ id: 'speed', name: 'speed', type: { base: 'real', dimensions: {} } }],
  functions: [{ id: 'sigmoid', signature: 'sigmoid(x)' }, { id: 'sqrt', signature: 'sqrt(x)' }],
} as unknown as FieldModel;
const offered = (before: string) => completions(model, before).items.map(({ label }) => label);

it('offers nothing for the unit after a number', () => {
  expect(offered('0.02 s')).toEqual([]);
  expect(offered('5ms')).toEqual([]);
  expect(offered('.5 s')).toEqual([]);
});

it('offers names and functions where an identifier can stand', () => {
  expect(offered('s')).toEqual(['speed', 'sigmoid', 'sqrt']);
  expect(offered('2 * s')).toEqual(['speed', 'sigmoid', 'sqrt']);
  expect(offered('x2 + sq')).toEqual(['sqrt']);
});

it('reads a sequence rule at the field root as a Sweep and a choice over one as a Draw', () => {
  const rule = { kind: 'sequence', rule: 'linspace', first: { kind: 'literal', value: 0 }, last: { kind: 'literal', value: 1 },
    count: { kind: 'literal', value: 5 } } as const;
  expect(valueKind(rule, [])).toBe('sweep');
  expect(valueKind({ kind: 'draw', family: 'choice', args: [rule] }, [])).toBe('draw');
});
