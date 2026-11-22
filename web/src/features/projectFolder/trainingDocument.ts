import type { Step } from '@/generated/studioContracts';

/** The Train payload keys a training document supplies, each read at its own pointer. */
export const trainingDocumentKeys = ['method', 'settings', 'training_config'] as const;

/** The training document a Train step reads, from its plan's document ids; none when its settings are inline. */
export function trainingDocumentId(ids?: Readonly<Record<string, string>>) {
  return ids?.['payload.settings'];
}

/** The document a Train step reads the form field at `field`, a JSON pointer, from; none when it holds it inline. */
export function trainFieldDocument(step: Step | undefined, field: string) {
  const key = field.split('/')[1];
  const reference = step && 'inline' in step.base && key ? step.base.inline[key] : undefined;
  return reference && typeof reference === 'object' && !Array.isArray(reference) && typeof reference.document === 'string'
    ? reference.document : undefined;
}

/** Whether a Train form field, by its JSON pointer, is one the training document holds. */
export function inTrainingDocument(field: string) {
  return trainingDocumentKeys.some((key) => field === `/${key}` || field.startsWith(`/${key}/`));
}
