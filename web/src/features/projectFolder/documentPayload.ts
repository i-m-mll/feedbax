import type { Document, JsonValue } from '../../generated/studioContracts';

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

/** `name`, its base, that document's base, and so on. */
export function baseChain(documents: Readonly<Record<string, Document>> | null | undefined, name: string): readonly string[] {
  const chain: string[] = [];
  for (let current: string | undefined = name; current !== undefined && !chain.includes(current);) {
    const document: Document | undefined = documents?.[current];
    if (!document) break;
    chain.push(current);
    current = 'base' in document ? document.base.document : undefined;
  }
  return chain;
}
