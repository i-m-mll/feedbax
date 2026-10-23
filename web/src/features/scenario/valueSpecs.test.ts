import { describe, expect, it } from 'vitest';
import {
  VALUE_SPEC_SCHEMA_VERSION,
  normalizeStudioValueSpec,
  setValueSpecEnumerable,
  setValueSpecDistributionFamily,
  setValueSpecDistributionParameter,
  setValueSpecConstantValue,
  setValueSpecFunction,
  setValueSpecFunctionParameter,
  setValueSpecMode,
  setValueSpecScope,
  valueSpecEnumerableCount,
  valueSpecChipLabel,
  valueSpecValidationErrors,
  valueSpecUsesIndexedValues,
} from './valueSpecs';
import type { StudioValueSpec } from '@/types/workspace';

const base: StudioValueSpec = {
  schema_version: 'feedbax.studio.value.v1',
  value_form: 'literal',
  variation: { scope: 'fixed', enumerable: null, metadata: {} },
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
      schema_version: VALUE_SPEC_SCHEMA_VERSION,
      value_form: 'distribution',
      mode: 'distribution',
      variation: { scope: 'trial' },
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
      value_form: 'function',
      mode: 'function',
      function_id: 'ramp',
      parameters: { domain: 'epoch', start_at: 0, end_at: 1, start: 0, end: 1 },
      sampling_scope: 'timestep',
    });
    expect(setValueSpecDistributionFamily(base, 'normal')).toMatchObject({
      value_form: 'distribution',
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
    expect(replicated.variation?.scope).toBe('replicate');

    const edited = setValueSpecConstantValue(replicated, { active: 0.5, inactive: 0 });

    expect(edited.value).toEqual({ active: 0.5, inactive: 0 });
    expect(setValueSpecScope(edited, 'run').variation?.scope).toBe('run');
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

  it('normalizes legacy v1 specs into the v2 value and variation split', () => {
    const normalized = normalizeStudioValueSpec({
      ...base,
      mode: 'distribution',
      distribution: { family: 'uniform', parameters: { min: 0, max: 1 } },
      sampling_scope: 'trial',
    });

    expect(normalized).toMatchObject({
      schema_version: VALUE_SPEC_SCHEMA_VERSION,
      value_form: 'distribution',
      mode: 'distribution',
      variation: { scope: 'trial' },
    });
  });

  it('emits enumerable sweep axes for P2 without expanding the matrix', () => {
    const swept = setValueSpecEnumerable(setValueSpecScope(base, 'sweep'), {
      form: 'range',
      start: 0,
      stop: 1,
      count: 5,
      scale: 'linear',
    });

    expect(swept).toMatchObject({
      value_form: 'literal',
      sampling_scope: 'sweep',
      variation: {
        scope: 'sweep',
        enumerable: { form: 'range', start: 0, stop: 1, count: 5 },
      },
    });
    expect(valueSpecEnumerableCount(swept)).toBe(5);
    expect(valueSpecValidationErrors(swept, ['fixed', 'run', 'sweep'])).toEqual([]);
  });

  it('enforces per-field variation eligibility and sampler counts', () => {
    const swept = setValueSpecScope(setValueSpecDistributionFamily(base, 'uniform'), 'sweep');

    expect(swept.variation?.enumerable).toMatchObject({ form: 'sampler', n: 2 });
    expect(valueSpecValidationErrors(swept, ['fixed', 'run'])).toEqual([
      'sweep variation is not eligible for this field',
    ]);
    expect(
      valueSpecValidationErrors(
        {
          ...swept,
          variation: { scope: 'sweep', enumerable: { form: 'sampler', sampler: {} } },
        },
        ['sweep']
      )
    ).toEqual(['Sweep sampler requires a positive sample count']);
  });
});
