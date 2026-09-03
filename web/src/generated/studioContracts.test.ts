import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  AdditiveGraphChannelTargetSpecSchema,
  ConstantArrayValueSpecSchema,
  GraphMetadataSchema,
  SemanticAnchorSchema,
  SparseCooArrayValueSpecSchema,
  SparseCooEntrySpecSchema,
  StudioPersistenceDocumentSchema,
  StudioTaskTimelineSpecSchema,
  StudioValueEnumerableSpecSchema,
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

  it('projects list, item, and safe-integer constraints', () => {
    expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [], value: 2 }).success).toBe(false);
    expect(SparseCooEntrySpecSchema.safeParse({ coordinate: [-1], value: 2 }).success).toBe(false);
    expect(
      SparseCooEntrySpecSchema.safeParse({ coordinate: [Number.MAX_SAFE_INTEGER + 1], value: 2 })
        .success,
    ).toBe(false);
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

describe('generated Studio constraint parity', () => {
  const valueSpec = (value: unknown) => ({
    schema_version: 'feedbax.spec.studio.value.v2' as const,
    value_form: 'literal' as const,
    variation: { scope: 'fixed' as const, metadata: {} },
    mode: 'constant' as const,
    value,
    metadata: {},
  });
  const timeline = () => ({
    schema_id: 'feedbax.spec.studio.task_timeline' as const,
    schema_version: 'feedbax.spec.studio.task_timeline.v2' as const,
    epochs: [{ id: 'epoch:0', label: 'Epoch', index: 0, length: valueSpec(null), metadata: {} }],
    signals: [{
      id: 'hold',
      label: 'Hold',
      kind: 'signal',
      task_data_id: 'hold',
      path: 'inputs.hold',
      value_spec: valueSpec(0),
      metadata: {},
    }],
    epoch_value_specs: [{
      schema_id: 'feedbax.spec.studio.epoch_value' as const,
      schema_version: 'feedbax.spec.studio.epoch_value.v1' as const,
      target_id: 'hold',
      epoch_id: 'epoch:0',
      value_spec: valueSpec(1),
    }],
    segments: [],
    metadata: {},
  });

  it('projects numeric bounds and string patterns', () => {
    expect(
      GraphMetadataSchema.safeParse({
        name: 'test',
        created_at: 'now',
        updated_at: 'now',
        save_revision: -1,
      }).success,
    ).toBe(false);
    expect(
      SemanticAnchorSchema.safeParse({
        semantic_document_sha256: 'not-a-digest',
        authored_path: 'graph',
      }).success,
    ).toBe(false);
  });

  it('runs the same registered cross-field refinements as Python', () => {
    expect(
      AdditiveGraphChannelTargetSpecSchema.safeParse({
        kind: 'edge',
        target_node: 'target',
        target_port: 'input',
      }).success,
    ).toBe(false);
    expect(
      StudioValueEnumerableSpecSchema.safeParse({
        form: 'range',
        start: 0,
        stop: 1,
      }).success,
    ).toBe(false);
  });

  it('fails closed on an unknown persistence-document version', () => {
    expect(
      StudioPersistenceDocumentSchema.safeParse({
        schema_id: 'feedbax.spec.studio.persistence_document',
        schema_version: 'feedbax.spec.studio.persistence_document.v99',
      }).success,
    ).toBe(false);
  });

  it('enforces the generated timeline identity, targets, and overlap rules', () => {
    expect(StudioTaskTimelineSpecSchema.safeParse(timeline()).success).toBe(true);

    const future = timeline();
    (future as { schema_version: string }).schema_version =
      'feedbax.spec.studio.task_timeline.v99';
    expect(StudioTaskTimelineSpecSchema.safeParse(future).success).toBe(false);

    const unknown = timeline();
    unknown.epoch_value_specs[0].target_id = 'missing';
    expect(StudioTaskTimelineSpecSchema.safeParse(unknown).success).toBe(false);

    const overlap = timeline();
    overlap.epoch_value_specs.push({ ...overlap.epoch_value_specs[0] });
    expect(StudioTaskTimelineSpecSchema.safeParse(overlap).success).toBe(false);

    const malformed = timeline();
    (malformed.epoch_value_specs[0] as { value_spec: unknown }).value_spec = {
      ...valueSpec(null),
      value_form: 'distribution',
      mode: 'distribution',
      distribution: { family: 'uniform', parameters: { min: 0 } },
    };
    expect(StudioTaskTimelineSpecSchema.safeParse(malformed).success).toBe(false);
  });
});
