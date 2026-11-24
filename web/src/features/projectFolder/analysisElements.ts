import { shownName } from '@/features/pipeline/planView';
import type { Base, Document, JsonValue } from '@/generated/studioContracts';
import { documentPayload, stepDocument } from './documentPayload';
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

/** Every operation and figure key the project's Analysis documents hold; a new element's key is none of them. */
export function heldElementKeys(documents: Documents, composed: Composed): ReadonlySet<string> {
  return new Set(documentsOfRole({ documents }, 'analysis')
    .flatMap((name) => Object.keys(payloadElements(documentPayload(documents, composed, name)))));
}

/** The shown name of `element` of the Analysis document `step` reads. */
export function stepElementName(
  project: { readonly documents?: Documents; readonly steps?: Readonly<Record<string, { readonly base?: Base }>> | null },
  composed: Composed, step: string, element: string,
) {
  const document = stepDocument(project.steps?.[step]?.base);
  return shownName(element, document ? payloadElements(documentPayload(project.documents, composed, document))[element] : undefined);
}
