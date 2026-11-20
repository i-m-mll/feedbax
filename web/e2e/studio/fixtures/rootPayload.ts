import type { Document, JsonValue } from '../../../src/generated/studioContracts';

/** The payload a root document holds; a based document holds differences instead. */
export function rootPayload(document: Document | undefined): JsonValue | undefined {
  return document && 'payload' in document ? document.payload : undefined;
}
