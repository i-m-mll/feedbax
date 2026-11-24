import { describe, expect, it } from 'vitest';
import type { StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import { studioProblemName } from './studioVocabulary';

const problem = (document_path: string | null): StudioProjectWindowMapProblem => ({
  code: 'invalid_payload', path: 'payload.trial_documents.model', detail: 'Invalid.', variant: null, step: 'train', row: null,
  place: { part: 'model', document: 'model' }, document_path,
});

describe('studioProblemName', () => {
  it('names a problem by the field it points into, never by a pointer or an index', () => {
    expect(studioProblemName(problem('/interior/components'))).toBe('Components');
    expect(studioProblemName(problem('/terms/0'))).toBe('Terms');
    expect(studioProblemName(problem('/links/{"element":{"kind":"field","path":["reset"]}}'))).toBe('Links');
  });

  it('names a problem about a whole document by its place', () => {
    expect(studioProblemName(problem(null))).toBe('Model');
    expect(studioProblemName(problem(''))).toBe('Model');
  });
});
