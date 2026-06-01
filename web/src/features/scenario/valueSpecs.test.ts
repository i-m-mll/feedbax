import { describe, expect, it } from 'vitest';
import {
  setValueSpecDistributionFamily,
  setValueSpecDistributionParameter,
  setValueSpecConstantValue,
  setValueSpecFunction,
  setValueSpecFunctionParameter,
  setValueSpecMode,
  setValueSpecScope,
  valueSpecChipLabel,
  valueSpecUsesIndexedValues,
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
      parameters: { domain: 'epoch', start_at: 0, end_at: 1, start: 0, end: 1 },
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
    expect(setValueSpecScope(setValueSpecMode(base, 'distribution'), 'epoch').sampling_scope).toBe(
      'epoch'
    );
  });

  it('keeps constants literal instead of inventing local axis counts', () => {
    const replicated = setValueSpecScope(base, 'replicate');

    expect(valueSpecUsesIndexedValues(replicated)).toBe(false);
    expect(replicated.value).toEqual({ active: 1, inactive: 0 });
    expect(replicated.sampling_scope).toBeNull();

    const edited = setValueSpecConstantValue(replicated, { active: 0.5, inactive: 0 });

    expect(edited.value).toEqual({ active: 0.5, inactive: 0 });
    expect(setValueSpecScope(edited, 'run').sampling_scope).toBeNull();
  });

  it('edits function and distribution parameters directly', () => {
    expect(setValueSpecFunctionParameter(setValueSpecFunction(base, 'step'), 'after', 2)).toMatchObject({
      mode: 'function',
      function_id: 'step',
      parameters: { domain: 'epoch', switch_at: 1, before: 0, after: 2 },
    });
    expect(
      setValueSpecDistributionParameter(setValueSpecDistributionFamily(base, 'uniform'), 'max', 4)
    ).toMatchObject({
      mode: 'distribution',
      distribution: { family: 'uniform', parameters: { min: 0, max: 4 } },
    });
  });
});
