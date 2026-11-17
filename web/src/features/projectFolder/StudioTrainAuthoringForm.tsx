import { useEffect, useId, useLayoutEffect, useRef, useSyncExternalStore } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiErrorMessage } from '@/api/request';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import {
  catalogEntries, projectTrainFields, readStudioTrainAuthoring, useAuthoringCatalog,
} from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { FieldRow } from '@/components/ui/FieldRow';
import { MenuItem } from '@/components/ui/Menu';
import { SettingsForm } from '@/components/values/SettingsForm';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { ValueField } from '@/components/values/ValueField';
import { pointer } from '@/components/values/adapters/jsonSchema';
import { fromFieldView, type ProjectedEdit } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { PlanStepView } from '@/features/pipeline/planView';
import { selectedRuns, type PipelineSelection } from '@/features/pipeline/selection';
import {
  StepReferenceSchema, type StepReference,
} from '@/generated/studioContracts';
import type {
  AuthoredFileAddress, JsonValue, StudioAuthoringField, StudioSavedTrainKey, StudioTrainAuthoringRequest, TrainFieldView,
} from '@/generated/studioContracts';
import { modelDocumentId, sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';
import { filesRemoval, sweepValueRuns, useRunRemoval } from './runRemoval';
import { checkpointStartEdits } from './createCheckpointTrain';
import { documentChoice, stepChoice, stepChoiceId, stepGroup } from './referenceChoices';
import { ReferenceField } from '@/components/values/ReferenceField';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';

interface Props {
  readonly editor: ProjectEditController;
  readonly stepId: string;
  readonly selection?: PipelineSelection;
  readonly run?: StudioSavedTrainKey;
  readonly fieldRows?: readonly string[];
  readonly step?: PlanStepView;
  readonly onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
  readonly focusFirstField?: boolean;
  readonly onFirstFieldFocused?: () => void;
  /** The project's Train steps, whose checkpoints the step can start from. */
  readonly trains?: readonly PlanStepView[];
  readonly picking?: ElementSelectionStore;
}

const settingsPrefix = '/settings';

function validateRecipe(
  authored: AuthoredProject, address: AuthoredFileAddress, documents: readonly string[],
): ProjectEditValidation {
  const bases: AuthoredFileAddress[] = [];
  let source = authored.steps?.[address.id]?.base;
  while (source && 'step' in source) {
    bases.push({ kind: 'step', id: source.step });
    source = authored.steps?.[source.step]?.base;
  }
  let edited: AuthoredProject | undefined;
  const sameStep = (a: AuthoredProject, b: AuthoredProject) => sameAuthoredFile(a, b, address);
  return (before, after) => {
    if (JSON.stringify(before.sweep) !== JSON.stringify(authored.sweep) ||
        documents.some((id) => JSON.stringify(before.documents?.[id]) !== JSON.stringify(authored.documents?.[id])) ||
        bases.some((base) => !sameAuthoredFile(before, authored, base))) {
      throw new ProjectEditConflictError();
    }
    if (!edited && sameStep(before, authored)) {
      edited = after;
    } else if (!edited ||
               !(sameStep(before, authored) && sameStep(after, edited) ||
                 sameStep(before, edited) && sameStep(after, authored))) {
      throw new ProjectEditConflictError();
    }
  };
}

/** The step's numeric fields as the server projects them, and the edit that applies one through the run confirmation. */
function useTrainFields(editor: ProjectEditController, stepId: string, run?: StudioSavedTrainKey, rows: readonly string[] = []) {
  const { saved, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const body = { revision, run, rows: run ? [...rows] : [] };
  const fields = useQuery({
    queryKey: ['train-fields', projectId, stepId, body],
    queryFn: ({ signal }) => projectTrainFields(projectId, stepId, body, signal),
    placeholderData: (previous, query) => query?.queryKey[1] === projectId && query.queryKey[2] === stepId ? previous : undefined,
  });
  const confirmRemoval = useRunRemoval();
  const editable = !fields.isPlaceholderData && !pending && !outside;
  async function edit(field: string, change: ProjectedEdit): Promise<string | undefined> {
    try {
      const recipe = await projectTrainFields(projectId, stepId, { ...body, field, edit: change }, editor.signal);
      if (recipe.problems.length) return recipe.problems[0].reason;
      if (recipe.files.length && await confirmRemoval(sweepValueRuns(recipe.removed_values))) {
        editor.editFiles(studioName('setField'), recipe.files, {
          reviewedRevision: revision,
          dependencies: recipe.dependencies.map((id) => ({ kind: 'document', id })),
          validate: validateRecipe(saved.project, { kind: 'step', id: stepId }, recipe.dependencies),
        });
        await editor.flush();
      }
      return undefined;
    } catch (error) {
      return apiErrorMessage(error);
    }
  }
  return { fields, edit: editable ? edit : undefined };
}

/** The step's fields a sweep can start from, as items of the menu at the end of its card's shape line. */
export function SweepFieldItems({ editor, stepId, onSwept }: {
  editor: ProjectEditController; stepId: string; onSwept(): void;
}) {
  const { fields, edit } = useTrainFields(editor, stepId);
  const sweepable = fields.data?.fields.filter((field) => field.can_sweep && !field.sweep) ?? [];
  return <>{sweepable.map((field) => <MenuItem key={field.id} disabled={!edit}
    onClick={() => { void edit?.(field.id, { kind: 'to_sweep' }).then((reason) => { if (!reason) onSwept(); }); }}>
    {field.label}
  </MenuItem>)}</>;
}

function choiceField(field: StudioAuthoringField, commit?: (value: JsonValue) => Promise<string | undefined>) {
  return rawField({
    id: field.id, label: field.label, value: field.value,
    type: field.control?.kind === 'choice' ? { kind: 'choice', options: field.control.options } : { kind: 'readOnly' },
    change: field.control ? commit : undefined,
  });
}

/** Where a Train step starts: its Model, or another Train step's last checkpoint. */
function CheckpointStart({ editor, stepId, step, trains, picking, editable }: {
  editor: ProjectEditController; stepId: string; step?: PlanStepView; trains: readonly PlanStepView[];
  picking?: ElementSelectionStore; editable: boolean;
}) {
  const { saved } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const confirmRemoval = useRunRemoval();
  const labelId = useId();
  const authored = saved.project.steps?.[stepId];
  const model = modelDocumentId(step?.documentIds);
  if (!authored || !model) return null;
  const source = StepReferenceSchema.safeParse(authored.inputs?.checkpoint);
  const sources = trains.filter(({ id }) => id !== stepId);
  const current = source.success ? sources.find(({ id }) => id === source.data.step) : undefined;
  const inherited = 'step' in authored.base &&
    !!trains.find(({ id }) => 'step' in authored.base && id === authored.base.step)?.inputs.some(({ id }) => id === 'checkpoint');
  const name = studioName('start');
  async function start(next: StepReference | null) {
    const files = [{ address: { kind: 'step' as const, id: stepId }, patches: checkpointStartEdits(authored!, inherited, next) }];
    if (!files[0].patches.length) return;
    const { revision } = editor.getSnapshot().saved;
    if (!await confirmRemoval(filesRemoval(saved.project_id, revision, files, editor.signal))) return;
    editor.editFiles(name, files, {
      reviewedRevision: revision,
      dependencies: next ? [{ kind: 'step', id: next.step }] : [{ kind: 'document', id: model! }],
    });
    await editor.flush();
  }
  return <FieldRow label={name} labelId={labelId}>
    <ReferenceField label={name} aria-labelledby={labelId} picking={picking}
      groups={[{ label: studioName('documents'), choices: [documentChoice('model', model)] }, stepGroup(sources)]}
      value={source.success ? current ? stepChoice(current) : { id: stepChoiceId(source.data.step), label: source.data.step }
        : documentChoice('model', model)}
      onChoose={editable ? (choice) => {
        const train = sources.find(({ id }) => stepChoiceId(id) === choice.id);
        void start(train ? { step: train.id, checkpoint: 'last' } : null);
      } : undefined} />
  </FieldRow>;
}

export function StudioTrainAuthoringForm(props: Props) {
  const { editor, stepId, onChooseSweepValue, focusFirstField, onFirstFieldFocused } = props;
  const selection = props.selection ?? { kind: 'steps', stepIds: [stepId] };
  const { saved, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const methodLabel = useId();
  const definition = useQuery({
    queryKey: ['train-authoring', projectId, stepId, revision],
    queryFn: ({ signal }) => readStudioTrainAuthoring(projectId, stepId, { revision }, signal),
    placeholderData: (previous) => previous?.project_id === projectId && previous.step_id === stepId ? previous : undefined,
  });
  const catalog = useAuthoringCatalog();
  const numeric = useTrainFields(editor, stepId, props.run, props.fieldRows);
  const active = useRef<AbortController | null>(null);
  const command = useRef<AbortController | null>(null);
  const region = useRef<HTMLElement | null>(null);
  const shown = definition.data;
  const current = shown?.revision === revision ? shown : null;

  useEffect(() => {
    const lifetime = new AbortController();
    active.current = lifetime;
    return () => lifetime.abort();
  }, [editor, stepId]);

  useLayoutEffect(() => {
    if (!focusFirstField || !current || numeric.fields.isFetching) return;
    const first = region.current?.querySelector<HTMLElement>('input:not([readonly]):not(:disabled)');
    if (!first) return;
    first.focus();
    onFirstFieldFocused?.();
  }, [focusFirstField, current, onFirstFieldFocused, numeric.fields.isFetching]);

  async function apply(change: NonNullable<StudioTrainAuthoringRequest['change']>, label: string): Promise<string | undefined> {
    const lifetime = active.current;
    if (!current || !lifetime || lifetime.signal.aborted || pending || outside || command.current === lifetime) {
      return undefined;
    }
    command.current = lifetime;
    try {
      const recipe = await readStudioTrainAuthoring(projectId, stepId, { revision, change }, editor.signal);
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) throw new ProjectEditConflictError();
      if (!recipe.patches.length) return undefined;
      const address = { kind: 'step', id: stepId } as const;
      editor.edit(address, label, recipe.patches, validateRecipe(latest.saved.project, address, recipe.documents));
      await editor.flush();
      return undefined;
    } catch (error) {
      return apiErrorMessage(error);
    } finally {
      if (command.current === lifetime) command.current = null;
    }
  }

  const projected = numeric.fields.data?.fields ?? [];
  const members = (field: TrainFieldView) => new Set(props.step && field.sweep && selection.kind !== 'steps'
    ? selectedRuns(props.step, selection).map((row) => row.values[field.sweep!.id]).filter(Boolean) : []);
  const fieldModel = (field: TrainFieldView) => fromFieldView(field, {
    id: JSON.stringify([projectId, stepId, props.fieldRows, field.id]),
    commit: numeric.edit && ((edit) => numeric.edit!(field.id, edit)),
    rowSelection: !!props.fieldRows?.length,
    selectedMembers: members(field),
    chooseMember: field.sweep && onChooseSweepValue
      ? (valueId, extend) => onChooseSweepValue(field.sweep!.id, valueId, extend) : undefined,
  });
  const fields = current?.fields ?? [];
  const methodField = fields.find(({ id }) => id === '/method');
  const methods = catalogEntries(catalog.data, 'training_methods');
  const method = catalog.data?.registries.training_methods?.entries
    .find((entry) => 'key' in entry && entry.key === methodField?.value);
  const settingsSchema = method && 'settings_schema' in method ? method.settings_schema : undefined;
  const choices = fields.filter(({ id, control }) => id !== '/method' && id !== settingsPrefix && control?.kind !== 'number');
  const steps = projected.filter(({ id }) => !id.startsWith(settingsPrefix + '/'));
  const byPath = new Map(projected.map((field) => [field.id, field]));
  const editable = !!current && !pending && !outside;

  return (
    <StudioRegion state={studioReadState('definition', shown && numeric.fields.data,
      definition.error ?? numeric.fields.error ?? catalog.error, () => {
        void definition.refetch();
        void numeric.fields.refetch();
        void catalog.refetch();
      })}>
      <section ref={region} aria-label={studioName('definition')} className="grid gap-1">
        {props.trains && <CheckpointStart editor={editor} stepId={stepId} step={props.step} trains={props.trains}
          picking={props.picking} editable={editable} />}
        {methodField && <FieldRow label={methodField.label} labelId={methodLabel}>
          <CatalogPicker aria-labelledby={methodLabel} className="w-full" entries={methods}
            value={typeof methodField.value === 'string' ? methodField.value : null}
            disabled={!editable || !methodField.control}
            onChoose={(key) => { void apply({ field: '/method', value: key }, methodField.label); }} />
        </FieldRow>}
        {choices.map((field) => <ValueField key={field.id} model={choiceField(field,
          editable ? (value) => apply({ field: field.id, value }, field.label) : undefined)} />)}
        {steps.map((field) => { const model = fieldModel(field); return <ValueField key={model.id} model={model} />; })}
        {settingsSchema && current?.settings && <SettingsForm schema={settingsSchema} value={current.settings}
          projected={(path) => {
            const field = byPath.get(settingsPrefix + pointer(path));
            return field && fieldModel(field);
          }}
          write={editable ? (path, value) => apply({ field: settingsPrefix + pointer(path), value }, humanizeLabel(path.at(-1) ?? '')) : undefined} />}
      </section>
    </StudioRegion>
  );
}
