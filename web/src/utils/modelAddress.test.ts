import { expect, it } from 'vitest';

import { ModelDocumentSchema } from '@/generated/studioContracts';
import { modelAddressFromText, modelAddressText } from './modelAddress';

it('round-trips the canonical address of a field on a nested port', () => {
  const text = '{"composites":["outer","inner"],"document":"model","element":{"kind":"field",'
    + '"parent":{"end":{"component":"same","direction":"input","owner":"component",'
    + '"port":"value"},"kind":"port"},"path":["fields","array/length",0]}}';
  expect(modelAddressText(modelAddressFromText(text))).toBe(text);
  expect(() => modelAddressFromText(text.replace(',0]', ',true]'))).toThrow();
  expect(() => modelAddressFromText(text + ' ')).toThrow(/canonical JSON/);
});

it('validates a recursive model and refuses a malformed ID below the root', () => {
  const leaf = { kind: 'leaf', name: 'Scale', type: 'feedbax.Scale' };
  const deepest = { components: { same: leaf } };
  const inner = { kind: 'composite', name: 'Inner', ports: {}, interior: deepest };
  const outer = { kind: 'composite', name: 'Outer', ports: {}, interior: { components: { inner } } };
  const model = { step_seconds: 0.02, ports: {}, interior: { components: { outer } } };
  expect(ModelDocumentSchema.parse(model)).toEqual(model);
  const invalidInner = { ...inner, interior: { components: { 'not/an/id': leaf } } };
  const invalidOuter = { ...outer, interior: { components: { inner: invalidInner } } };
  const invalid = { ...model, interior: { components: { outer: invalidOuter } } };
  expect(ModelDocumentSchema.safeParse(invalid).success).toBe(false);
});
