import { studioName, studioStep } from '@/components/activity/studioVocabulary';
import type { ReferenceChoice, ReferenceGroup } from '@/components/values/ReferenceField';
import type { PlanStepView } from '@/features/pipeline/planView';
import { viewGlyphs } from '@/features/projectWindow/ProjectWindow';
import { documentsOfRole } from './useProjectDocuments';

type DocumentKind = 'model' | 'task' | 'objective';

export const documentChoiceId = (document: string) => JSON.stringify(['document', document]);
export const stepChoiceId = (step: string) => JSON.stringify(['step', step]);

export function documentChoice(kind: DocumentKind, document: string): ReferenceChoice {
  const Glyph = viewGlyphs[kind];
  return { id: documentChoiceId(document), label: document, glyph: <Glyph /> };
}

export function stepChoice(step: Pick<PlanStepView, 'id' | 'label' | 'kind'>): ReferenceChoice {
  const { Icon } = studioStep(step.kind);
  return { id: stepChoiceId(step.id), label: step.label, glyph: <Icon />, pick: { kind: 'step', step: step.id } };
}

/** The project's documents of a kind, as the Documents group of a reference field. */
export function documentGroup(project: Parameters<typeof documentsOfRole>[0], kind: DocumentKind): ReferenceGroup {
  return { label: studioName('documents'), choices: documentsOfRole(project, kind).map((id) => documentChoice(kind, id)) };
}

/** Steps whose output a reference reads, as its Steps group. */
export function stepGroup(steps: readonly Pick<PlanStepView, 'id' | 'label' | 'kind'>[]): ReferenceGroup {
  return { label: studioName('steps'), choices: steps.map(stepChoice) };
}
