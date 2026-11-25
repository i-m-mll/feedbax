import { expect, it } from 'vitest';
import { SubsetTermsSchema, type Step } from '@/generated/studioContracts';
import { changedSelections, connection, withSubsetTerms } from './selectionEdits';

const train: Step = { kind: 'train', base: { inline: {} } } as unknown as Step;
const evaluate = (subject: unknown): Step => ({ kind: 'evaluate', base: { inline: {} }, inputs: { subject } }) as unknown as Step;

it('changes a selection without carrying the old step restriction, and writes a list of one as its reference', () => {
  const selections = [
    { step: 'a', checkpoint: { at: 4 }, where: { clauses: [{ seed: [{ id: 'one', label: 'One' }] }] } },
    { step: 'b', checkpoint: { at: 4 } },
  ];
  expect(changedSelections(selections, { kind: 'replace', index: 0, reference: { step: 'c', checkpoint: 'last' } }))
    .toEqual([{ step: 'c', checkpoint: { at: 4 } }, selections[1]]);
  // A Synthesize step's model has no checkpoint, so none is carried onto it.
  expect(changedSelections(selections, { kind: 'replace', index: 0, reference: { step: 'synthesize' } }))
    .toEqual([{ step: 'synthesize' }, selections[1]]);
  expect(changedSelections(selections, { kind: 'remove', index: 0 })).toEqual(selections[1]);
});

it('wires a Train card to an Evaluate subject of Train steps once, and to nothing else', () => {
  const steps = {
    a: train, b: train,
    reads: evaluate({ step: 'a', checkpoint: 'last' }),
    model: evaluate({ kind: 'initial_model', source: { document: 'arm' } }),
  };
  expect(connection(steps, 'b', 'reads')?.edits).toEqual([
    { op: 'replace', path: '/inputs/subject', value: [{ step: 'a', checkpoint: 'last' }, { step: 'b', checkpoint: 'last' }] },
    { op: 'add', path: '/sweep', value: { axes: { subject: { input: 'subject' } } } },
  ]);
  expect(connection(steps, 'a', 'reads')).toBeUndefined();
  expect(connection(steps, 'b', 'model')).toBeUndefined();
  expect(connection(steps, 'reads', 'b')).toBeUndefined();
});

it('wires a Synthesize card to an Evaluate subject as a model with no checkpoint', () => {
  const synthesize = { kind: 'synthesize', base: { inline: {} } } as unknown as Step;
  const steps = { a: train, s: synthesize, reads: evaluate({ step: 'a', checkpoint: { at: 7 } }) };
  expect(connection(steps, 's', 'reads')?.edits[0]).toEqual({
    op: 'replace', path: '/inputs/subject', value: [{ step: 'a', checkpoint: { at: 7 } }, { step: 's' }],
  });
  const fromModel = { ...steps, reads: evaluate({ step: 's' }) };
  expect(connection(fromModel, 'a', 'reads')?.edits[0]).toEqual({
    op: 'replace', path: '/inputs/subject', value: [{ step: 's' }, { step: 'a', checkpoint: 'last' }],
  });
});

it('omits an unconstrained selection and rejects explicitly empty clauses', () => {
  const source = { step: 'train', checkpoint: 'last' as const, where: { rule: 'seed=1' } };
  expect(withSubsetTerms(source, {})).toEqual({ step: 'train', checkpoint: 'last' });
  expect(withSubsetTerms(source, { bound_ranks: [] })).toEqual({ ...source, where: { bound_ranks: [] } });
  expect(SubsetTermsSchema.safeParse({}).success).toBe(true);
  expect(SubsetTermsSchema.safeParse({ clauses: [] }).success).toBe(false);
  expect(SubsetTermsSchema.safeParse({ clauses: [{}] }).success).toBe(false);
});
