import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  ConstantArrayValueSpecSchema,
  SparseCooArrayValueSpecSchema,
  SparseCooEntrySpecSchema,
} from '@/generated/studioContracts';
import type {
  ConstantArrayValueSpec,
  SparseCooArrayValueSpec,
  SparseCooEntrySpec,
} from '@/generated/studioContracts';

const tags = {
  schema_id: 'feedbax.spec.component_param.array_value',
  schema_version: 'feedbax.spec.component_param.array_value.v1',
} as const;

describe('generated component-param array value contracts', () => {
  it('preserves numeric array and scalar union types', () => {
    expectTypeOf<SparseCooEntrySpec['coordinate']>().toEqualTypeOf<number[]>();
    expectTypeOf<SparseCooArrayValueSpec['shape']>().toEqualTypeOf<number[]>();
    expectTypeOf<ConstantArrayValueSpec['shape']>().toEqualTypeOf<number[]>();
    expectTypeOf<SparseCooEntrySpec['value']>().toEqualTypeOf<
      boolean | number | 'nan' | '+inf' | '-inf'
    >();
  });

  it('enforces integer coordinates and shapes', () => {
    expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [0, 1], value: 2 }).success).toBe(
      true,
    );
    expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [0.5], value: 2 }).success).toBe(
      false,
    );
    expect(
      ConstantArrayValueSpecSchema.safeParse({
        ...tags,
        encoding: 'constant',
        shape: [2.5],
        dtype: 'float32',
        nonfinite: 'forbid',
        value: 0,
      }).success,
    ).toBe(false);
  });

  it('rejects arbitrary strings and objects while preserving non-finite tokens', () => {
    for (const invalid of ['arbitrary', { arbitrary: true }]) {
      expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [0], value: invalid }).success).toBe(
        false,
      );
      expect(
        SparseCooArrayValueSpecSchema.safeParse({
          ...tags,
          encoding: 'sparse_coo',
          shape: [1],
          dtype: 'float32',
          nonfinite: 'allow',
          fill: invalid,
          entries: [],
        }).success,
      ).toBe(false);
      expect(
        ConstantArrayValueSpecSchema.safeParse({
          ...tags,
          encoding: 'constant',
          shape: [1],
          dtype: 'float32',
          nonfinite: 'allow',
          value: invalid,
        }).success,
      ).toBe(false);
    }

    for (const token of ['nan', '+inf', '-inf'] as const) {
      expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [0], value: token }).success).toBe(
        true,
      );
    }
  });
});
