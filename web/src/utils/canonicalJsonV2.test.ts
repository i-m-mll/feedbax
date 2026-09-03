import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  CanonicalJsonV2Error,
  canonicalJsonV2,
} from '@/utils/canonicalJsonV2';
import {
  STUDIO_DRAFT_HASH_PIN,
  STUDIO_DRAFT_HASH_SCHEMA_ID,
  STUDIO_DRAFT_HASH_SCHEMA_VERSION,
  admitStudioDraftHashes,
  studioDraftDigestV2,
  studioDraftHashes,
} from '@/utils/studioDraftHash';

interface ConformanceCase {
  case_id: string;
  input: { form: 'json'; value: unknown } | { form: 'special'; name: string };
  expected_utf8_hex?: string;
  expected_draft_digest?: string;
  expected_error?: string;
}

interface ConformanceVector {
  schema_id: string;
  schema_version: string;
  algorithm: string;
  draft_hash: { schema_id: string; schema_version: string; pin: string };
  cases: ConformanceCase[];
}

class UnsupportedLeaf {}

function specialValue(name: string): unknown {
  if (name === 'nan') return Number.NaN;
  if (name === 'positive_infinity') return Number.POSITIVE_INFINITY;
  if (name === 'negative_infinity') return Number.NEGATIVE_INFINITY;
  if (name === 'non_string_key') return new Map<unknown, unknown>([[1, 'value']]);
  if (name === 'unsupported_leaf') return { leaf: new UnsupportedLeaf() };
  if (name === 'array_cycle') {
    const value: unknown[] = [];
    value.push(value);
    return value;
  }
  throw new Error(`Unknown conformance special value: ${name}`);
}

function inputValue(testCase: ConformanceCase): unknown {
  // BigInt preserves the vector's integer kind, which JavaScript Number erases.
  if (testCase.case_id === 'unsafe_integer' && testCase.input.form === 'json') {
    return BigInt(testCase.input.value as number);
  }
  return testCase.input.form === 'json'
    ? testCase.input.value
    : specialValue(testCase.input.name);
}

function hex(value: string): string {
  return Array.from(new TextEncoder().encode(value), (byte) =>
    byte.toString(16).padStart(2, '0')
  ).join('');
}

const vector = JSON.parse(
  readFileSync(new URL('../../../conformance/canonical_json_v2.json', import.meta.url), 'utf8')
) as ConformanceVector;

describe('canonical_json_v2 shared conformance', () => {
  it('pins the exact shared canonical bytes, rejections, and Studio draft digests', () => {
    expect(vector).toMatchObject({
      schema_id: 'feedbax.conformance.canonical_json_v2',
      schema_version: 'feedbax.conformance.canonical_json_v2.v1',
      algorithm: 'canonical_json_v2',
      draft_hash: {
        schema_id: STUDIO_DRAFT_HASH_SCHEMA_ID,
        schema_version: STUDIO_DRAFT_HASH_SCHEMA_VERSION,
        pin: STUDIO_DRAFT_HASH_PIN,
      },
    });

    for (const testCase of vector.cases) {
      const value = inputValue(testCase);
      if (testCase.expected_utf8_hex) {
        expect(hex(canonicalJsonV2(value)), testCase.case_id).toBe(testCase.expected_utf8_hex);
        expect(studioDraftDigestV2(value), testCase.case_id).toBe(
          testCase.expected_draft_digest
        );
      } else {
        try {
          canonicalJsonV2(value);
          throw new Error(`Expected ${testCase.case_id} to be rejected`);
        } catch (error) {
          expect(error, testCase.case_id).toBeInstanceOf(CanonicalJsonV2Error);
          expect((error as CanonicalJsonV2Error).code, testCase.case_id).toBe(
            testCase.expected_error
          );
        }
        expect(() => studioDraftDigestV2(value), testCase.case_id).toThrow(
          CanonicalJsonV2Error
        );
      }
    }
  });

  it('migrates legacy maps, round-trips current hashes, and rejects unknown pins', () => {
    const legacy = admitStudioDraftHashes({ training_spec: 'fnv1a:12345678' });
    expect(legacy).toEqual({
      schema_id: 'feedbax.studio.draft_hashes',
      schema_version: 'feedbax.studio.draft_hashes.v1',
      pin: 'fnv1a32-runtime_local_json_v1',
      rehash_required: true,
      hashes: { training_spec: 'fnv1a:12345678' },
    });
    expect(admitStudioDraftHashes(JSON.parse(JSON.stringify(legacy)))).toEqual(legacy);

    const current = studioDraftHashes({ training_spec: { integral: 1, zero: -0 } });
    expect(admitStudioDraftHashes(JSON.parse(JSON.stringify(current)))).toEqual(current);
    expect(() => admitStudioDraftHashes({ ...current, pin: 'fnv1a32-canonical_json_v999' }))
      .toThrow('Unsupported Studio draft hash pin: fnv1a32-canonical_json_v999');
    expect(() => admitStudioDraftHashes({
      ...current,
      schema_version: 'feedbax.studio.draft_hashes.v999',
    })).toThrow(
      'Unsupported Studio draft hash schema version: feedbax.studio.draft_hashes.v999'
    );
  });
});
