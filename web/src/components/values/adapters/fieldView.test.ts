import { expect, it } from 'vitest';
import type { StudioFieldView } from '@/generated/studioContracts';
import { fromFieldView } from './fieldView';

// Catches: a swept duration shown as bare numbers, which reads as steps or loses the unit it was authored in.
it('shows a swept value list in the unit it was authored in', () => {
  const field = {
    label: 'Start offset', text: '', unit: 's', site: 'offset',
    sweep: { kind: 'list', values: [{ id: 'a', value: 0.1 }, { id: 'b', value: 0.2 }] },
  } as unknown as StudioFieldView;
  const sites = { sites: { offset: { kinds: ['value', 'sweep'], expected: { base: 'real' }, functions: '0' } }, function_lists: { 0: [] } };
  expect(fromFieldView(field, { id: 'offset', sites: sites as never }).text).toBe('0.1 s, 0.2 s');
});
