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
