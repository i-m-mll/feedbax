import { describe, expect, it } from 'vitest';
import { componentLibrary } from '@/data/components';

describe('component registry fallback', () => {
  it('does not carry a hand-maintained offline registry', () => {
    expect(componentLibrary).toEqual([]);
  });
});
