import { expect, it } from 'vitest';
import { completions, type FieldModel } from './fieldModel';

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
