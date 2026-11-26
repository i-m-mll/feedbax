import { expect, it } from 'vitest';
import type { StudioFieldView } from '@/generated/studioContracts';
import { fromFieldView, type FieldSites } from './fieldView';

// Catches: a swept duration shown as bare numbers, which reads as steps or loses the unit it was authored in.
it('shows a swept value list in the unit it was authored in', () => {
  const field = {
    label: 'Start offset', text: '', unit: 's', site: 'duration',
    sweep: { kind: 'list', values: [{ id: 'a', value: 0.1 }, { id: 'b', value: 0.2 }] },
  } as unknown as StudioFieldView;
  const tables = {
    sites: { duration: { expected: { base: 'real' }, finest: 0, kinds: ['value', 'sweep'], functions: 'none', parts: [] } },
    function_lists: { none: [] },
  } as const satisfies FieldSites;
  expect(fromFieldView(field, tables, { id: 'offset' }).text).toBe('0.1 s, 0.2 s');
});
