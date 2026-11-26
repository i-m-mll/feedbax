import { describe, expect, it } from 'vitest';
import type { ModelAddress } from '@/generated/studioContracts';
import { addressKey } from './modelGraph';
import { compactLayoutKey, modelLayout } from './modelLayout';

const root: ModelAddress = { composites: [], element: { kind: 'component', id: 'gru' } };
const nested: ModelAddress = { composites: ['core'], element: { kind: 'component', id: 'gru' } };
const unspelled: ModelAddress = { composites: [], element: { kind: 'component', id: 'added' } };

describe('modelLayout under compact keys', () => {
  const texts = { [addressKey(root)]: 'component.gru', [addressKey(nested)]: 'core/component.gru' };
  const layout = modelLayout({
    positions: { 'component.gru': { x: 1, y: 2 }, 'core/component.gru': { x: 30, y: 40 } },
    flipped: { 'core/component.gru': true },
  }, compactLayoutKey(texts));

  it('places a card inside a composite where it was saved, not where a root card of the same key was', () => {
    expect(layout.position(nested)).toEqual({ x: 30, y: 40 });
    expect(layout.position(root)).toEqual({ x: 1, y: 2 });
    expect(layout.flipped(root)).toBeUndefined();
  });

  it('gives an element the Service has not spelled yet no saved layout, rather than another element\'s', () => {
    expect(layout.position(unspelled)).toBeUndefined();
    expect(modelLayout({ positions: { undefined: { x: 9, y: 9 } } }, compactLayoutKey(texts)).position(unspelled)).toBeUndefined();
  });
});
