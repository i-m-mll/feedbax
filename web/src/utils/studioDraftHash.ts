import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';

export const STUDIO_DRAFT_HASH_SCHEMA_ID = 'feedbax.studio.draft_hashes' as const;
export const STUDIO_DRAFT_HASH_SCHEMA_VERSION = 'feedbax.studio.draft_hashes.v2' as const;
export const STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION =
  'feedbax.studio.draft_hashes.v1' as const;
export const STUDIO_DRAFT_HASH_PIN = 'fnv1a32-canonical_json_v2' as const;
export const STUDIO_DRAFT_HASH_LEGACY_PIN = 'fnv1a32-runtime_local_json_v1' as const;

export interface CurrentStudioDraftHashes {
  schema_id: typeof STUDIO_DRAFT_HASH_SCHEMA_ID;
  schema_version: typeof STUDIO_DRAFT_HASH_SCHEMA_VERSION;
  pin: typeof STUDIO_DRAFT_HASH_PIN;
  hashes: Record<string, string | null>;
}

export interface LegacyStudioDraftHashes {
  schema_id: typeof STUDIO_DRAFT_HASH_SCHEMA_ID;
  schema_version: typeof STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION;
  pin: typeof STUDIO_DRAFT_HASH_LEGACY_PIN;
  rehash_required: true;
  hashes: Record<string, string | null>;
}

export type StudioDraftHashes = CurrentStudioDraftHashes | LegacyStudioDraftHashes;

export function studioDraftDigestV2(value: unknown): string {
  let hash = 2166136261;
  for (const byte of new TextEncoder().encode(canonicalJsonV2(value))) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function studioDraftHashes(
  values: Record<string, unknown | null>
): CurrentStudioDraftHashes {
  return {
    schema_id: STUDIO_DRAFT_HASH_SCHEMA_ID,
    schema_version: STUDIO_DRAFT_HASH_SCHEMA_VERSION,
    pin: STUDIO_DRAFT_HASH_PIN,
    hashes: Object.fromEntries(
      Object.entries(values).map(([key, value]) => [
        key,
        value === null || value === undefined ? null : studioDraftDigestV2(value),
      ])
    ),
  };
}

export function admitStudioDraftHashes(value: unknown): StudioDraftHashes {
  if (!isRecord(value)) throw new Error('Studio draft hashes must be an object');
  const isEnvelope = ['schema_id', 'schema_version', 'pin', 'hashes'].some((key) => key in value);
  if (!isEnvelope) {
    return {
      schema_id: STUDIO_DRAFT_HASH_SCHEMA_ID,
      schema_version: STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION,
      pin: STUDIO_DRAFT_HASH_LEGACY_PIN,
      rehash_required: true,
      hashes: stringHashes(value),
    };
  }
  if (value.schema_id !== STUDIO_DRAFT_HASH_SCHEMA_ID) {
    throw new Error(`Unsupported Studio draft hash schema id: ${String(value.schema_id)}`);
  }
  if (value.schema_version === STUDIO_DRAFT_HASH_SCHEMA_VERSION) {
    if (value.pin !== STUDIO_DRAFT_HASH_PIN) {
      throw new Error(`Unsupported Studio draft hash pin: ${String(value.pin)}`);
    }
    const hashes = stringHashes(value.hashes);
    for (const digest of Object.values(hashes)) {
      if (digest !== null && !/^[0-9a-f]{8}$/.test(digest)) {
        throw new Error('Current Studio draft hash digests must be eight lowercase hex digits');
      }
    }
    return {
      schema_id: STUDIO_DRAFT_HASH_SCHEMA_ID,
      schema_version: STUDIO_DRAFT_HASH_SCHEMA_VERSION,
      pin: STUDIO_DRAFT_HASH_PIN,
      hashes,
    };
  }
  if (value.schema_version === STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION) {
    if (value.pin !== STUDIO_DRAFT_HASH_LEGACY_PIN) {
      throw new Error(`Unsupported Studio draft hash pin: ${String(value.pin)}`);
    }
    if (value.rehash_required !== true) {
      throw new Error('Legacy Studio draft hashes must declare rehash_required=true');
    }
    return {
      schema_id: STUDIO_DRAFT_HASH_SCHEMA_ID,
      schema_version: STUDIO_DRAFT_HASH_LEGACY_SCHEMA_VERSION,
      pin: STUDIO_DRAFT_HASH_LEGACY_PIN,
      rehash_required: true,
      hashes: stringHashes(value.hashes),
    };
  }
  throw new Error(
    `Unsupported Studio draft hash schema version: ${String(value.schema_version)}`
  );
}

function stringHashes(value: unknown): Record<string, string | null> {
  if (!isRecord(value)) throw new Error('Studio draft hash values must be an object');
  const hashes: Record<string, string | null> = {};
  for (const [key, digest] of Object.entries(value)) {
    if (typeof digest === 'string') {
      hashes[key] = digest;
      continue;
    }
    if (digest === null) {
      hashes[key] = null;
      continue;
    }
    throw new Error('Studio draft hash values must be strings or null');
  }
  return hashes;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}
