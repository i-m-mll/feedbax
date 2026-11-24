import { describe, expect, it } from 'vitest';
import type { Document } from '@/generated/studioContracts';
import { baseChain } from './documentPayload';
import { documentTree } from './useProjectDocuments';

const schema = 'feedbax.spec.model';
const documents = {
  arm: { schema, payload: {} },
  narrowed: { schema, base: { document: 'arm' }, differences: {} },
  dangling: { schema, base: { document: 'arm', unresolved: true }, differences: {} },
} as unknown as Record<string, Document>;

describe('the picker tree', () => {
  it('orders documents by the names it shows, not by their keys', () => {
    const labelled = {
      'simple-reaches-2': { schema, label: 'Simple reaches 2', payload: {} },
      'simple-reaches-task': { schema, label: 'Simple reaches', payload: {} },
      'reaches-10': { schema, label: 'Simple reaches 10', payload: {} },
    } as unknown as Record<string, Document>;
    expect(documentTree({ documents: labelled }, 'model').map(({ name }) => name))
      .toEqual(['Simple reaches', 'Simple reaches 2', 'Simple reaches 10']);
  });
});

describe('a base kept unresolved', () => {
  it('reads nothing from the document that later took its spelling, and the picker draws it as a root', () => {
    expect(baseChain(documents, 'narrowed')).toEqual(['narrowed', 'arm']);
    expect(baseChain(documents, 'dangling')).toEqual(['dangling']);
    expect(documentTree({ documents }, 'model').map(({ key, depth }) => [key, depth]))
      .toEqual([['arm', 0], ['narrowed', 1], ['dangling', 0]]);
  });
});
