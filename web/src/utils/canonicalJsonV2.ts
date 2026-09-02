export type CanonicalJsonV2ErrorCode =
  | 'cycle'
  | 'lone_surrogate'
  | 'non_finite_number'
  | 'non_string_key'
  | 'unsafe_integer'
  | 'unsupported_type';

export class CanonicalJsonV2Error extends Error {
  constructor(
    readonly code: CanonicalJsonV2ErrorCode,
    readonly path: string,
    detail: string
  ) {
    super(`canonical JSON ${code} at ${path || '<root>'}: ${detail}`);
  }
}

export function canonicalJsonV2(value: unknown): string {
  return encode(value, '', new Set<object>());
}

function encode(value: unknown, path: string, active: Set<object>): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') {
    assertNoLoneSurrogates(value, path);
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new CanonicalJsonV2Error('non_finite_number', path, 'number must be finite');
    }
    return JSON.stringify(value);
  }
  if (typeof value === 'bigint') {
    throw new CanonicalJsonV2Error(
      'unsafe_integer',
      path,
      'integers outside the JavaScript safe range are not admitted'
    );
  }
  if (value instanceof Map) {
    if (Array.from(value.keys()).some((key) => typeof key !== 'string')) {
      throw new CanonicalJsonV2Error('non_string_key', path, 'object keys must be strings');
    }
    throw new CanonicalJsonV2Error('unsupported_type', path, 'Map is not a JSON object');
  }
  if (!value || typeof value !== 'object') {
    throw new CanonicalJsonV2Error(
      'unsupported_type',
      path,
      'value must be null, a boolean, string, finite number, array, or object'
    );
  }
  if (active.has(value)) {
    throw new CanonicalJsonV2Error('cycle', path, 'arrays and objects must not contain cycles');
  }
  active.add(value);
  try {
    if (Array.isArray(value)) {
      return `[${value.map((item, index) => encode(item, `${path}/${index}`, active)).join(',')}]`;
    }
    if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
      throw new CanonicalJsonV2Error('unsupported_type', path, 'value must be a plain JSON object');
    }
    const symbols = Object.getOwnPropertySymbols(value);
    if (symbols.length > 0) {
      throw new CanonicalJsonV2Error('non_string_key', path, 'object keys must be strings');
    }
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => {
        assertNoLoneSurrogates(key, path);
        return `${JSON.stringify(key)}:${encode(record[key], `${path}/${pointerToken(key)}`, active)}`;
      })
      .join(',')}}`;
  } finally {
    active.delete(value);
  }
}

function assertNoLoneSurrogates(value: string, path: string): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        index += 1;
        continue;
      }
      throw new CanonicalJsonV2Error('lone_surrogate', path, 'string contains a lone surrogate');
    }
    if (code >= 0xdc00 && code <= 0xdfff) {
      throw new CanonicalJsonV2Error('lone_surrogate', path, 'string contains a lone surrogate');
    }
  }
}

function pointerToken(value: string): string {
  return value.replace(/~/g, '~0').replace(/\//g, '~1');
}
