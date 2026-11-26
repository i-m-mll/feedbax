import { expect, it, vi } from 'vitest';
import { familyOver, type Commit, type TrialView } from './StudioTrialAuthoring';

const view = { changedFrom: {}, fields: [] } as unknown as TrialView;

function removals(over: Record<string, Record<string, number>>, combination?: 'zip') {
  const commit = vi.fn<Commit>(async () => undefined);
  const group = familyOver(view, 'objective', ['terms', 'speed'], { over, ...combination && { combination } }, commit)!;
  for (const index of group.indices) index.remove!();
  return commit.mock.calls.map(([, change]) => change);
}

it('removes one index of several, and the whole rule with the last, so no family is left with an empty `over`', () => {
  expect(removals({ x: { a: 1 }, y: { b: 2 } })).toEqual([
    { kind: 'objective', edits: [{ op: 'remove', path: '/terms/speed/over/x' }] },
    { kind: 'objective', edits: [{ op: 'remove', path: '/terms/speed/over/y' }] },
  ]);
  expect(removals({ x: { a: 1 } }, 'zip')).toEqual([{ kind: 'objective', edits: [
    { op: 'remove', path: '/terms/speed/over' }, { op: 'remove', path: '/terms/speed/combination' },
  ] }]);
});

it('gives a plain member no Over group', () => {
  expect(familyOver(view, 'objective', ['terms', 'speed'], {})).toBeUndefined();
});

it("names a rule's values as the Service does, and writes a rename where the value's label is stored", () => {
  const field = (key: string) => ({ path: `/objective/terms/speed/over/side/${key}`, label: '', text: '1', site: 'constant' });
  const named = {
    ...view, fields: [field('neg'), field('pos')], sites: { constant: { expected: { base: 'real' }, finest: 'run', names: [], kinds: ['value'], parts: [], functions: 'none' } }, function_lists: { none: [] },
    menus: () => ({}),
    families: { '/objective/terms/speed': { member_count: 2, indices: { side: { count: 2, values: [
      { key: 'neg', name: 'Left' }, { key: 'pos', name: 'pos' },
    ] } } } },
  } as unknown as TrialView;
  const commit = vi.fn<Commit>(async () => undefined);
  const [side] = familyOver(named, 'objective', ['terms', 'speed'], { over: { side: { neg: { value: -1, label: 'Left' }, pos: 1 } } }, commit)!.indices;
  expect(side.values!.map(({ name }) => name)).toEqual(['Left', 'pos']);
  side.values![0].rename!('West');
  side.values![1].rename!('East');
  expect(commit.mock.calls.map(([, change]) => change)).toEqual([
    { kind: 'objective', edits: [{ op: 'replace', path: '/terms/speed/over/side/neg/label', value: 'West' }] },
    { kind: 'objective', edits: [{ op: 'add', path: '/terms/speed/over/side/pos/label', value: 'East' }] },
  ]);
});
