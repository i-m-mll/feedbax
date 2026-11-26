import { useId } from 'react';
import { catalogEntries, useAuthoringCatalog } from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { FieldRow } from '@/components/ui/FieldRow';
import { settingsFields } from '@/components/values/SettingsForm';
import { ValueField } from '@/components/values/ValueField';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { PlanView } from '@/features/pipeline/planView';
import type { JsonValue, Project, ProjectValueEdit, Step, StudioAuthoringCatalog } from '@/generated/studioContracts';

const inline = '/base/inline';

/** A Synthesize step's method and settings, when the step authors them inline. */
export function synthesisPayload(step: Step | null | undefined) {
  if (step?.kind !== 'synthesize' || !('inline' in step.base)) return undefined;
  const { method, settings } = step.base.inline;
  if (typeof method !== 'string') return undefined;
  const values = settings && typeof settings === 'object' && !Array.isArray(settings) ? settings : undefined;
  return { method, settings: values, hasSettings: settings !== undefined };
}

/** `plan` with each Synthesize step's method label, as the authoring catalog names it. */
export function withMethodLabels(plan: PlanView, project: Project, catalog: StudioAuthoringCatalog | undefined): PlanView {
  const labels = new Map(catalogEntries(catalog, 'synthesis_methods').map(({ key, label }) => [key, label]));
  if (!plan.steps.some(({ kind }) => kind === 'synthesize')) return plan;
  return { ...plan, steps: plan.steps.map((step) => {
    const method = step.kind === 'synthesize' ? synthesisPayload(project.steps?.[step.id])?.method : undefined;
    return method ? { ...step, method: labels.get(method) ?? method } : step;
  }) };
}

/**
 * The Method of a Synthesize step, chosen among the registered synthesis methods, and the chosen method's settings
 * from its schema. Choosing another method clears the settings, which belong to the method.
 */
export function SynthesisForm({ step, refusal, onEdit }: {
  step: Step;
  refusal?: string;
  /** Saves one change to the step under `name`; absent when the step cannot be edited here. */
  onEdit?: (name: string, patches: ProjectValueEdit[]) => void;
}) {
  const labelId = useId();
  const catalog = useAuthoringCatalog().data;
  const payload = synthesisPayload(step);
  if (!payload) return null;
  const methods = catalogEntries(catalog, 'synthesis_methods');
  const method = catalog?.registries.synthesis_methods?.entries
    .find((entry) => 'key' in entry && entry.key === payload.method);
  const schema = method && 'settings_schema' in method ? method.settings_schema : undefined;
  const settings: JsonValue = payload.settings ?? {};
  const write = onEdit && ((path: readonly string[], value: JsonValue) => {
    onEdit(humanizeLabel(path.at(-1) ?? ''), [
      { op: payload.hasSettings ? 'replace' : 'add', path: `${inline}/settings`, value: setIn(settings, path, value) },
    ]);
    return Promise.resolve(undefined);
  });
  return <section aria-label={studioName('definition')} className="grid gap-1">
    <FieldRow label={studioName('method')} labelId={labelId} below={refusal && <p role="alert">{refusal}</p>}>
      <CatalogPicker aria-labelledby={labelId} className="w-full" entries={methods} value={payload.method}
        disabled={!onEdit}
        onChoose={(key) => {
          if (key === payload.method) return;
          onEdit?.(studioName('method'), [
            { op: 'replace', path: `${inline}/method`, value: key },
            ...payload.hasSettings ? [{ op: 'remove', path: `${inline}/settings` } as const] : [],
          ]);
        }} />
    </FieldRow>
    {schema && settingsFields({ schema, value: settings, write, choose: write })
      .map((model) => <ValueField key={model.id} model={model} />)}
  </section>;
}
