import type { DocumentPresentation, ModelAddress, ModelProjection } from '@/generated/studioContracts';
import { addressKey } from './modelGraph';

/** The key under which the Service persists an element's layout; none when the Service has not spelled the element. */
export type LayoutKey = (address: ModelAddress) => string | undefined;

/** The Service's compact text for each element it projects, read by the element's runtime key. */
export function compactLayoutKey(texts: ModelProjection['address_texts'] | undefined): LayoutKey {
  return (address) => texts?.[addressKey(address)];
}

/** One Model document's persisted layout, read element by element under the key the Service persists for it. */
export function modelLayout(document: DocumentPresentation | undefined, key: LayoutKey) {
  const read = <T>(entries: Record<string, T> | undefined) => (address: ModelAddress): T | undefined => {
    const stored = key(address);
    return stored === undefined ? undefined : entries?.[stored];
  };
  return {
    position: read(document?.positions),
    flipped: read(document?.flipped),
    collapsed: read(document?.collapsed),
    waypoints: read(document?.waypoints),
  };
}
