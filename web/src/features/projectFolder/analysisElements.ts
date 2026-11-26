import type { Base, Document, JsonValue } from '@/generated/studioContracts';
import { documentPayload, stepDocument } from './documentPayload';
import { documentsOfRole } from './useProjectDocuments';

type Documents = Readonly<Record<string, Document>> | null | undefined;
type Composed = Readonly<Record<string, JsonValue>> | undefined;
/** An Analysis payload's operations and figures by key. */
function payloadElements(payload: JsonValue | undefined): Readonly<Record<string, unknown>> {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return {};
  const section = (name: string) => {
    const members = payload[name];
    return members && typeof members === 'object' && !Array.isArray(members) ? members as Record<string, unknown> : {};
  };
  return { ...section('operations'), ...section('views') };
}

/** The keys of every operation and figure the project's Analysis documents hold, none of which a new key can be. */
export function heldElementKeys(documents: Documents, composed: Composed) {
  return new Set(documentsOfRole({ documents }, 'analysis')
    .flatMap((name) => Object.keys(payloadElements(documentPayload(documents, composed, name)))));
}

/** The label of operation or figure `element` of the Analysis document step `step` of `project` reads. */
export function elementLabel(
  project: { readonly documents?: Documents; readonly steps?: Readonly<Record<string, { readonly base: Base }>> | null },
  composed: Composed, step: string, element: string,
) {
  const base = project.steps?.[step]?.base;
  const document = base && stepDocument(base);
  const member = document === undefined ? undefined : payloadElements(documentPayload(project.documents, composed, document))[element];
  const label = member && typeof member === 'object' && 'label' in member ? member.label : undefined;
  if (typeof label !== 'string') throw new Error(`Step ${step} reads no element ${element}.`);
  return label;
}
