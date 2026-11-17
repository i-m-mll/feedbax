import { describe, expect, it } from 'vitest';
import type { ModelAddress } from '@/generated/studioContracts';
import { holds } from './modelGraph';

const address = (composites: string[], element: ModelAddress['element']): ModelAddress => ({ composites, element });
const core = address([], { kind: 'component', id: 'core' });

describe('holds', () => {
  it('counts what lies inside a composite at any depth, and the fields and ports of an element', () => {
    expect(holds(core, address(['core', 'inner'], { kind: 'component', id: 'gru' }))).toBe(true);
    expect(holds(core, address(['core'], { kind: 'graph' }))).toBe(true);
    expect(holds(core, address([], { kind: 'field', parent: { kind: 'component', id: 'core' }, path: ['size'] }))).toBe(true);
    expect(holds(core, address([], { kind: 'port', end: { owner: 'component', component: 'core', port: 'in', direction: 'input' } }))).toBe(true);
  });

  it('does not count a sibling or an element of the enclosing graph', () => {
    expect(holds(core, address([], { kind: 'component', id: 'readout' }))).toBe(false);
    expect(holds(core, address(['readout'], { kind: 'component', id: 'core' }))).toBe(false);
    expect(holds(core, address([], { kind: 'graph' }))).toBe(false);
  });
});
