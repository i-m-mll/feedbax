import { describe, expect, it } from 'vitest';
import {
  createRetainedObservable,
  retainedObservableSelectorPatch,
  retentionPolicy,
  selectorToRetainedObservableTarget,
} from './observables';
import type { StudioSelectorRef } from '@/types/workspace';

describe('scenario retained observable operations', () => {
  it('creates a port-backed retained observable from a selector ref', () => {
    const selector: StudioSelectorRef = {
      namespace: 'graph_port',
      compact: 'port:mechanics.effector',
      target_id: 'mechanics',
      path: 'effector',
      role: 'observed',
      expected_shape: ['time', 2],
      dtype: 'float32',
      units: 'cm',
      frame: 'workspace',
      metadata: {
        direction: 'output',
        label: 'Effector',
        value_schema: {
          id: 'value:port:mechanics.effector',
          label: 'Effector',
          kind: 'graph_port',
          dtype: 'float32',
          shape: ['time', 2],
          origin: 'declared',
          metadata: {},
        },
      },
    };

    const observable = createRetainedObservable({
      selector,
      label: 'Effector capture',
      existingIds: new Set(['obs:effector_capture_deadbeef']),
    });

    expect(observable).toMatchObject({
      label: 'Effector capture',
      selector: 'port:mechanics.effector',
      target: {
        kind: 'port',
        selector: 'port:mechanics.effector',
        node_id: 'mechanics',
        port: 'effector',
        timing: 'output',
      },
      retention: { mode: 'trajectory' },
      value_schema: expect.objectContaining({ id: 'value:port:mechanics.effector' }),
    });
    expect(observable?.id).toMatch(/^obs:effector_capture_/);
  });

  it('maps supported structural selector kinds to retained observable targets', () => {
    const edgeSelector: StudioSelectorRef = {
      namespace: 'recurrent_carry',
      compact: 'edge:cell.hidden->cell.hidden',
      target_id: 'cell:hidden->cell:hidden',
      path: null,
      role: 'observed',
      metadata: { edge_id: 'cell:hidden->cell:hidden' },
    };
    const taskSelector: StudioSelectorRef = {
      namespace: 'task_data',
      compact: 'task_data:targets.effector',
      target_id: 'scenario:train',
      path: 'targets.effector',
      role: 'observed',
      metadata: {},
    };

    expect(selectorToRetainedObservableTarget(edgeSelector)).toMatchObject({
      kind: 'recurrent_carry',
      edge_id: 'cell:hidden->cell:hidden',
    });
    expect(selectorToRetainedObservableTarget(taskSelector)).toMatchObject({
      kind: 'task_data',
      path: 'targets.effector',
    });
  });

  it('builds update patches and normalizes retention policy controls', () => {
    const selector: StudioSelectorRef = {
      namespace: 'state_path',
      compact: 'path:states.mechanics.effector.pos',
      target_id: 'mechanics',
      path: 'states.mechanics.effector.pos',
      role: 'observed',
      metadata: { label: 'Effector position' },
    };

    expect(retainedObservableSelectorPatch(selector)).toMatchObject({
      selector: 'path:states.mechanics.effector.pos',
      target: {
        kind: 'state_path',
        path: 'states.mechanics.effector.pos',
      },
    });
    expect(retentionPolicy('window')).toMatchObject({
      mode: 'window',
      window_size: 32,
      reason: 'explicit_observable_authoring',
    });
    expect(retentionPolicy('stream', { mode: 'window', window_size: 8 })).toMatchObject({
      mode: 'stream',
      window_size: null,
    });
  });
});
