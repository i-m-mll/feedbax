import { describe, expect, it } from 'vitest';
import type { Project } from '@/generated/studioContracts';
import { withBaseDescendants } from './baseDescendants';

const step = (base: object) => ({ kind: 'train', base }) as unknown as NonNullable<Project['steps']>[string];

describe('withBaseDescendants', () => {
  it('reaches every step based on a given one, at any depth, and no other', () => {
    const project = { steps: {
      'train-3': step({ step: 'train-2' }), 'train-2': step({ step: 'train' }), train: step({ inline: {} }),
      other: step({ inline: {} }), row: step({ step: 'other', row: 'a' }),
    } };
    expect(withBaseDescendants(project, ['train'])).toEqual(new Set(['train', 'train-2', 'train-3']));
    expect(withBaseDescendants(project, ['train-2'])).toEqual(new Set(['train-2', 'train-3']));
    expect(withBaseDescendants(project, ['other'])).toEqual(new Set(['other', 'row']));
  });

  it('leaves out a step whose base kept a spelling another step later took', () => {
    const project = { steps: { train: step({ inline: {} }), dangling: step({ step: 'train', unresolved: true }) } };
    expect(withBaseDescendants(project, ['train'])).toEqual(new Set(['train']));
  });
});
