import { studioFormat } from '@/components/activity/studioVocabulary';
import { ModelAddressSchema, type ModelAddress, type ModelFitProblem, type StudioLoadedModelImportResult } from '@/generated/studioContracts';

type Element = ModelAddress['element'];

function elementText(element: Element): string {
  switch (element.kind) {
    case 'field': return [elementText(element.parent), ...element.path].filter(Boolean).join('.');
    case 'attachment': return `${elementText(element.parent)}.${element.id}`;
    case 'port': return 'id' in element.end ? `${element.end.id}.${element.end.port}` : element.end.port;
    case 'component': case 'wire': case 'physical_connection': return element.id;
    default: return '';
  }
}

/** A Model address the service wrote as its canonical text, as the element ids it passes through; any other text as it is. */
function addressText(address: string) {
  let parsed;
  try {
    parsed = ModelAddressSchema.safeParse(JSON.parse(address));
  } catch {
    return address;
  }
  if (!parsed.success) return address;
  return [...parsed.data.composites, elementText(parsed.data.element)].filter(Boolean).join('/');
}

function fitProblem(raw: ModelFitProblem) {
  const problem = { ...raw, address: addressText(raw.address) };
  switch (problem.kind) {
    case 'missing_slot': return studioFormat('fitMissing', problem);
    case 'unexpected_slot': return studioFormat('fitUnexpected', problem);
    case 'mismatched_slot': return studioFormat('fitMismatched', problem);
    default: return studioFormat('fitModel', problem).trim();
  }
}

/** Each reason an import was not taken, in the service's order. */
export function importProblems(result: Exclude<StudioLoadedModelImportResult, { status: 'imported' }>): readonly string[] {
  return result.status === 'does_not_fit' ? result.report.problems.map(fitProblem)
    : [studioFormat('unreadableArchive', result)];
}
