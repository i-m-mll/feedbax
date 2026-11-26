import type { Document, JsonValue, StudioProjectMemberNames } from '@/generated/studioContracts';
import { documentPayload } from './documentPayload';
import { documentsOfRole } from './useProjectDocuments';

type Documents = Readonly<Record<string, Document>> | null | undefined;
type Composed = Readonly<Record<string, JsonValue>> | undefined;
type Labelled = { readonly label?: string | null };

/** An Analysis payload's operations and figures by key. */
function payloadElements(payload: JsonValue | undefined): Readonly<Record<string, Labelled>> {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return {};
  const section = (name: string) => {
    const members = payload[name];
    return members && typeof members === 'object' && !Array.isArray(members) ? members as Record<string, Labelled> : {};
  };
  return { ...section('operations'), ...section('views') };
}

/**
 * The keys of every operation and figure the project's Analysis documents hold, and their names: the snapshot's, and
 * the labels of elements an edit is still saving. A new element's key is none of the keys, and its name none of the
 * names.
 */
export function heldElements(documents: Documents, composed: Composed, names: StudioProjectMemberNames) {
  const elements = documentsOfRole({ documents }, 'analysis')
    .flatMap((name) => Object.entries(payloadElements(documentPayload(documents, composed, name))));
  return {
    keys: new Set(elements.map(([key]) => key)),
    names: new Set([
      ...Object.values(names.elements).flatMap((named) => Object.values(named)),
      ...elements.flatMap(([, { label }]) => label ? [label] : []),
    ]),
  };
}
