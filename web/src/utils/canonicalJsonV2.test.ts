import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  CanonicalJsonV2Error,
  canonicalJsonV2,
} from '@/utils/canonicalJsonV2';
interface ConformanceCase {
  case_id: string;
  input: { form: 'json'; value: unknown } | { form: 'special'; name: string };
  expected_utf8_hex?: string;
  expected_error?: string;
}

interface ConformanceVector {
  schema_id: string;
  schema_version: string;
  algorithm: string;
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
  readFileSync(new URL('../../../tests/fixtures/canonical_json_v2.json', import.meta.url), 'utf8')
) as ConformanceVector;

describe('canonical_json_v2 shared conformance', () => {
  it('pins the exact shared canonical bytes and rejections', () => {
    expect(vector).toMatchObject({
      schema_id: 'feedbax.conformance.canonical_json_v2',
      schema_version: 'feedbax.conformance.canonical_json_v2.v1',
      algorithm: 'canonical_json_v2',
    });

    for (const testCase of vector.cases) {
      const value = inputValue(testCase);
      if (testCase.expected_utf8_hex) {
        expect(hex(canonicalJsonV2(value)), testCase.case_id).toBe(testCase.expected_utf8_hex);
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
      }
    }
  });
});
