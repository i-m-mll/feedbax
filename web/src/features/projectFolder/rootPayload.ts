import type { Document, JsonValue } from '../../generated/studioContracts';

/** The payload a root document holds. A based document holds differences; its composed payload is the server's. */
export function rootPayload(document: Document | undefined): JsonValue | undefined {
  return document && 'payload' in document ? document.payload : undefined;
}
