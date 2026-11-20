/** The Train payload keys a training document supplies, each read at its own pointer. */
export const trainingDocumentKeys = ['method', 'settings', 'training_config'] as const;

/** The training document a Train step reads, from its plan's document ids; none when its settings are inline. */
export function trainingDocumentId(ids?: Readonly<Record<string, string>>) {
  return ids?.['payload.settings'];
}

/** Whether a Train form field, by its JSON pointer, is one the training document holds. */
export function inTrainingDocument(field: string) {
  return trainingDocumentKeys.some((key) => field === `/${key}` || field.startsWith(`/${key}/`));
}
