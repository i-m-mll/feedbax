import { describe, expect, it } from 'vitest';
import {
  setValueSpecDistributionFamily,
  setValueSpecFunction,
  setValueSpecMode,
  setValueSpecScope,
  valueSpecChipLabel,
} from './valueSpecs';
import type { StudioValueSpec } from '@/types/workspace';

const base: StudioValueSpec = {
  schema_version: 'feedbax.studio.value.v1',
  mode: 'constant',
  value: { active: 1, inactive: 0 },
  dtype: 'float32',
  shape: ['time', 1],
  units: null,
  frame: 'task_time',
  metadata: { value_schema_id: 'value:hold' },
};

describe('value spec helpers', () => {
  it('preserves schema metadata when switching modes', () => {
    const edited = setValueSpecMode(base, 'distribution');

    expect(edited).toMatchObject({
      mode: 'distribution',
      dtype: 'float32',
      shape: ['time', 1],
      metadata: {
        value_schema_id: 'value:hold',
        authored_as: 'value_spec_editor',
      },
    });
  });

  it('builds function and distribution defaults for structured editing', () => {
    expect(setValueSpecFunction(base, 'ramp')).toMatchObject({
      mode: 'function',
      function_id: 'ramp',
      parameters: { start: 0, end: 1 },
      sampling_scope: 'timestep',
    });
    expect(setValueSpecDistributionFamily(base, 'normal')).toMatchObject({
      mode: 'distribution',
      distribution: { family: 'normal', parameters: { mean: 0, std: 1 } },
    });
  });

  it('formats compact chip labels with evaluation scope', () => {
    expect(valueSpecChipLabel(base)).toBe('1/0');
    expect(
      valueSpecChipLabel({
        ...base,
        mode: 'distribution',
        distribution: { family: 'uniform', parameters: {} },
        sampling_scope: 'trial',
      })
    ).toBe('uniform/trial');
    expect(setValueSpecScope(base, 'epoch').sampling_scope).toBe('epoch');
  });
});
