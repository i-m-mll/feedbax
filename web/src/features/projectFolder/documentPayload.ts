import type { Base, Document, JsonValue } from '../../generated/studioContracts';

/** The payload `name` reads as: a root document's own, a based document's as the server composed it. */
export function documentPayload(
  documents: Readonly<Record<string, Document>> | null | undefined,
  composed: Readonly<Record<string, JsonValue>> | undefined,
  name: string,
): JsonValue | undefined {
  const document = documents?.[name];
  if (!document) return undefined;
  return 'payload' in document ? document.payload : composed?.[name];
}

/**
 * The document `name` is based on, where it is based on one. A base kept unresolved when its spelling was later taken
 * reads no document.
 */
export function baseOf(documents: Readonly<Record<string, Document>> | null | undefined, name: string | null | undefined) {
  const document = name ? documents?.[name] : undefined;
  return document && 'base' in document && !document.base.unresolved ? document.base.document : undefined;
}

/** The document a step's base reads whole, unless that reference was kept unresolved. */
export function stepDocument(base: Base | false | null | undefined) {
  return base && 'document' in base && !base.pointer && !base.unresolved ? base.document : undefined;
}

/** `name`, its base, that document's base, and so on. */
export function baseChain(documents: Readonly<Record<string, Document>> | null | undefined, name: string): readonly string[] {
  const chain: string[] = [];
  for (let current: string | undefined = name; current !== undefined && !chain.includes(current);) {
    const document: Document | undefined = documents?.[current];
    if (!document) break;
    chain.push(current);
    current = baseOf(documents, current);
  }
  return chain;
}
