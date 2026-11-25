import { studioName, studioStep } from '@/components/activity/studioVocabulary';
import type { ReferenceChoice, ReferenceGroup } from '@/components/values/ReferenceField';
import { shownName, type PlanStepView } from '@/features/pipeline/planView';
import { viewGlyphs } from '@/features/projectWindow/ProjectWindow';
import { Package } from 'lucide-react';
import { documentsOfRole, roleOf, type DocumentRole } from './useProjectDocuments';

/** The roles of the documents an input reads. */
export type DocumentKind = Extract<DocumentRole, 'model' | 'loaded_model' | 'task' | 'objective'>;

const documentGlyphs = {
  model: viewGlyphs.model, loaded_model: Package, task: viewGlyphs.task, objective: viewGlyphs.objective,
} satisfies Record<DocumentKind, unknown>;

export const documentChoiceId = (document: string) => JSON.stringify(['document', document]);
const documentChoicePrefix = documentChoiceId('').slice(0, -3);

/** The document a choice of the Documents group names. */
export function chosenDocument(choice: ReferenceChoice): string | undefined {
  return choice.id.startsWith(documentChoicePrefix) ? (JSON.parse(choice.id) as [string, string])[1] : undefined;
}
export const stepChoiceId = (step: string) => JSON.stringify(['step', step]);

/** The project's document `document`, by its shown name. */
export function documentChoice(project: Project, kind: DocumentKind, document: string): ReferenceChoice {
  const Glyph = documentGlyphs[kind];
  return { id: documentChoiceId(document), label: shownName(document, project.documents?.[document]), glyph: <Glyph /> };
}

export function stepChoice(step: Pick<PlanStepView, 'id' | 'label' | 'kind'>): ReferenceChoice {
  const { Icon } = studioStep(step.kind);
  return { id: stepChoiceId(step.id), label: step.label, glyph: <Icon />, pick: { kind: 'step', step: step.id } };
}

type Project = Parameters<typeof documentsOfRole>[0];

/** The project's documents of the kinds, as the Documents group of a reference field. */
export function documentGroup(project: Project, kinds: DocumentKind | readonly DocumentKind[]): ReferenceGroup {
  return { label: studioName('documents'), choices: [kinds].flat().flatMap((kind) =>
    documentsOfRole(project, kind).map((id) => documentChoice(project, kind, id))) };
}

/** The kind among `kinds` of the project's document `document`. */
export function documentKind(project: Project, document: string, kinds: readonly DocumentKind[]): DocumentKind | undefined {
  const stored = project.documents?.[document];
  const role = stored && roleOf(stored);
  return kinds.find((kind) => kind === role);
}

/** Steps whose output a reference reads, as its Steps group. */
export function stepGroup(steps: readonly Pick<PlanStepView, 'id' | 'label' | 'kind'>[]): ReferenceGroup {
  return { label: studioName('steps'), choices: steps.map(stepChoice) };
}
