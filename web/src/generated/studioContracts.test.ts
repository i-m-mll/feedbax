import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  AddModelAttachmentSchema,
  AdditiveGraphChannelTargetSpecSchema,
  ConstantArrayValueSpecSchema,
  ChangeModelAttachmentSchema,
  DocumentSchema,
  RootDocumentSchema,
  ExecutionTargetsSchema,
  GraphMetadataSchema,
  ModelDocumentSchema,
  SessionTargetSchema,
  SparseCooArrayValueSpecSchema,
  SparseCooEntrySpecSchema,
  StudioValueEnumerableSpecSchema,
} from '@/generated/studioContracts';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import type {
  ConstantArrayValueSpec,
  SparseCooArrayValueSpec,
  SparseCooEntrySpec,
} from '@/generated/studioContracts';

const tags = {
  schema_id: 'feedbax.spec.component_param.array_value',
  schema_version: 'feedbax.spec.component_param.array_value.v1',
} as const;

it('validates named execution targets and the exact venue session identity', () => {
  const profile = {
    kind: 'session', id: 'cpu-test', label: 'CPU test', driver: 'local-rehearsal',
    resources: { compute: 1 }, lifecycle: { kind: 'keep-open' }, maximum_microusd: 1000,
  };
  const target = {
    kind: 'session', profile: profile.id, driver: profile.driver,
    session_id: '3e030d7a-5884-47cd-a987-3b75a13d812d',
    realization: { environment: 'a'.repeat(64), placement: { device: 'cpu' } },
  };
  const catalog = {
    default: 'local',
    targets: [{
      profile, run_kinds: ['train', 'evaluate'],
      availability: { state: 'off', reason: 'Driver is unavailable.' },
    }],
  };
  expect(ExecutionTargetsSchema.parse(catalog)).toEqual(catalog);
  expect(SessionTargetSchema.parse(target)).toEqual(target);
  expect(SessionTargetSchema.safeParse({ ...target, profile: 'local' }).success).toBe(false);
  expect(SessionTargetSchema.safeParse({ ...target, session_id: 'arbitrary' }).success).toBe(false);
  expect(ExecutionTargetsSchema.safeParse({
    ...catalog, targets: [{ ...catalog.targets[0], profile: { ...profile, id: 'local' } }],
  }).success).toBe(false);
});

it('preserves JSON payloads and reports the nested path for invalid values', () => {
  const payload = { sample: [null, false, '', -0, 1.25, { scalar: Number.MAX_VALUE }], empty: {} };
  const document = { schema: 'example', label: 'Example', payload };
  const parsed = RootDocumentSchema.parse(document);
  expect(parsed).toEqual(document);
  expect(Object.is(parsed.payload, payload)).toBe(false);
  expect(canonicalJsonV2(parsed)).toBe(canonicalJsonV2(document));
  expect(RootDocumentSchema.safeParse({ ...document, extra: true }).success).toBe(false);
  expect(DocumentSchema.safeParse({ schema: 'example' }).success).toBe(false);

  for (const invalid of [Infinity, -Infinity, NaN, undefined, 1n, new Date(0), new Map(), new Set()]) {
    const result = RootDocumentSchema.safeParse({ schema: 'example', label: 'Example', payload: { sample: [0, invalid] } });
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error.issues[0].path).toEqual(['payload', 'sample', 1]);
  }

  const malformedModel = { step_seconds: 'zero', ports: {}, interior: {} };
  expect(DocumentSchema.safeParse({ schema: 'feedbax.spec.model', label: 'Model', payload: malformedModel }).success).toBe(true);
  expect(ModelDocumentSchema.safeParse(malformedModel).success).toBe(false);
});

it('validates an attachment owner, source-end anchor, and retained ID', () => {
  const owner = { composites: ['controller'], element: { kind: 'wire', id: 'feedback' } };
  const attachment = { kind: 'intervention', id: 'gain', operation: 'Scale', operands: { factor: { kind: 'value', value: 0.5 } } };
  const addition = { op: 'add', owner, attachment };
  expect(AddModelAttachmentSchema.safeParse(addition).success).toBe(false);
  expect(AddModelAttachmentSchema.safeParse({ ...addition, after: null }).success).toBe(true);
  const address = { ...owner, element: { kind: 'attachment', id: 'gain', parent: owner.element } };
  expect(ChangeModelAttachmentSchema.safeParse({ op: 'change', address, attachment }).success).toBe(true);
  expect(ChangeModelAttachmentSchema.safeParse({
    op: 'change', address, attachment: { ...attachment, id: 'another-gain' },
  }).success).toBe(false);
});

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
  it('projects numeric bounds', () => {
    expect(
      GraphMetadataSchema.safeParse({
        name: 'test',
        created_at: 'now',
        updated_at: 'now',
        save_revision: -1,
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
});
