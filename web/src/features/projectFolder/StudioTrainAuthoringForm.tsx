import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiErrorMessage, isRefusal } from '@/api/request';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import {
  catalogEntries, prepareTrainStart, projectTrainFields, readStudioTrainAuthoring, useAuthoringCatalog,
} from '@/api/studioAuthoring';
import { studioName, studioSweptParameterMissing } from '@/components/activity/studioVocabulary';
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
  AuthoredFileAddress, JsonValue, StudioAuthoringField, StudioClassField, StudioSavedTrainKey, StudioTrainAuthoringRequest,
  TrainFieldRequest, TrainFieldView,
} from '@/generated/studioContracts';
import type { FieldCommit, FieldKindOffer } from '@/components/values/fieldModel';
import type { ClassProjection } from '@/components/values/adapters/jsonSchema';
import { modelDocumentId, sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';
import { filesRemoval, sweepValueRuns, useRunRemoval } from './runRemoval';
import { inTrainingDocument } from './trainingDocument';
import { documentChoice, stepChoice, stepChoiceId, stepGroup } from './referenceChoices';
import { ReferenceField } from '@/components/values/ReferenceField';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';

interface Props {
  readonly editor: ProjectEditController;
  readonly stepId: string;
  readonly selection?: PipelineSelection;
  readonly run?: StudioSavedTrainKey;
  /** The plan of the saved revision is still loading, so `run` is not yet known and the fields wait for it. */
  readonly runPending?: boolean;
  readonly fieldRows?: readonly string[];
  /** The selected sweep values, when the selection is values: the rows are then stored by them where they are exactly their runs. */
  readonly fieldAxes?: Readonly<Record<string, readonly string[]>>;
  readonly step?: PlanStepView;
  readonly onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
  /** The project's Train steps, whose checkpoints the step can start from. */
  readonly trains?: readonly PlanStepView[];
  readonly picking?: ElementSelectionStore;
  /** The form is the Training view's: it shows the training document's fields. */
  readonly trainingDocument?: boolean;
  /** The training document's users an edit is made for, the step among them; without them, the step alone. */
  readonly chosen?: readonly string[];
  /** Takes the reason the service refuses to read the step's definition, which belongs on the step, and none once it reads. */
  readonly onRefused?: (reason: string | undefined) => void;
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

type RunSelection = Pick<TrainFieldRequest, 'run' | 'rows' | 'axes' | 'chosen_steps'>;

/**
 * The runs an edit is made for: the named run, and the selected rows and sweep values when they are not the whole step
 * (rows that are every run of the step are the step, so that an edit there reaches runs the edit itself adds); and the
 * steps an edit of a document is made for, none being the step alone.
 */
function runSelection(
  run?: StudioSavedTrainKey, rows: readonly string[] = [], axes: Readonly<Record<string, readonly string[]>> = {},
  chosen: readonly string[] = [], step?: PlanStepView,
): RunSelection {
  if (!run) return {};
  if (step && new Set(rows).size === step.runs.length) rows = [];
  return {
    run, rows: [...rows],
    axes: rows.length ? Object.fromEntries(Object.entries(axes).map(([axis, values]) => [axis, [...values]])) : {},
    chosen_steps: [...chosen],
  };
}

function useTrainFieldsQuery(
  editor: ProjectEditController, stepId: string | undefined, selection: RunSelection = {}, runPending = false,
) {
  const { project_id: projectId, revision } = useSyncExternalStore(editor.subscribe, editor.getSnapshot).saved;
  const body = { revision, ...selection };
  return { body, fields: useQuery({
    queryKey: ['train-fields', projectId, stepId, body],
    queryFn: ({ signal }) => projectTrainFields(projectId, stepId!, body, signal),
    enabled: stepId !== undefined && !runPending,
    placeholderData: (previous, query) => query?.queryKey[1] === projectId && query.queryKey[2] === stepId ? previous : undefined,
  }) };
}

/** A Train step's numeric fields' labels, by the dotted step-local path a sweep exception patches. */
export function useTrainFieldLabels(editor: ProjectEditController, stepId: string | undefined): ReadonlyMap<string, string> {
  const { data } = useTrainFieldsQuery(editor, stepId).fields;
  return useMemo(() => new Map((data?.fields ?? []).map(({ id, label }) => [jsonPointerPath(id), label])), [data]);
}

/** A Train step's class-valued fields as the server projects them for the whole step. */
export function useTrainClassFields(editor: ProjectEditController, stepId: string | undefined): readonly StudioClassField[] {
  const { project_id: projectId, revision } = useSyncExternalStore(editor.subscribe, editor.getSnapshot).saved;
  const { data } = useQuery({
    queryKey: ['train-authoring', projectId, stepId, revision, {}],
    queryFn: ({ signal }) => readStudioTrainAuthoring(projectId, stepId!, { revision }, signal),
    enabled: stepId !== undefined,
  });
  return data?.classes ?? [];
}

export function jsonPointerPath(id: string) {
  return id.split('/').slice(1).map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~')).join('.');
}

/** The step's numeric fields as the server projects them, and the edit that applies one through the run confirmation. */
function useTrainFields(
  editor: ProjectEditController, stepId: string, selection: RunSelection = {}, runPending = false,
) {
  const { saved, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const { body, fields } = useTrainFieldsQuery(editor, stepId, selection, runPending);
  const confirmRemoval = useRunRemoval();
  const editable = !fields.isPlaceholderData && !pending && !outside;
  async function edit(field: string, change: ProjectedEdit): Promise<string | undefined> {
    try {
      const { problems = [], files = [], dependencies = [], removed_values: removedValues } = await projectTrainFields(projectId, stepId, { ...body, field, edit: change }, editor.signal);
      if (problems[0]) return problems[0].reason;
      if (files.length && await confirmRemoval(sweepValueRuns(removedValues))) {
        editor.editFiles(studioName('setField'), files, {
          reviewedRevision: revision,
          dependencies: dependencies.map((id) => ({ kind: 'document', id })),
          validate: validateRecipe(saved.project, { kind: 'step', id: stepId }, dependencies),
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
  const sweepable = fields.data?.fields?.filter((field) => field.can_sweep && !field.sweep) ?? [];
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
  const [reason, setReason] = useState<string>();
  const authored = saved.project.steps?.[stepId];
  const model = modelDocumentId(step?.documentIds);
  if (!authored || !model) return null;
  const source = StepReferenceSchema.safeParse(authored.inputs?.checkpoint);
  const sources = trains.filter(({ id }) => id !== stepId);
  const current = source.success ? sources.find(({ id }) => id === source.data.step) : undefined;
  const name = studioName('start');
  async function start(next: StepReference | null) {
    const { revision } = editor.getSnapshot().saved;
    const { files } = await prepareTrainStart(saved.project_id, stepId, { revision, source: next }, editor.signal);
    if (!files.length || !await confirmRemoval(filesRemoval(saved.project_id, revision, files, editor.signal))) return;
    const based = 'step' in authored!.base ? [authored!.base.step] : [];
    editor.editFiles(name, files, {
      reviewedRevision: revision,
      dependencies: [...new Set([...based, ...next ? [next.step] : []])].map((id) => ({ kind: 'step' as const, id })),
    });
    await editor.flush();
  }
  return <FieldRow label={name} labelId={labelId} below={reason && <p role="alert">{reason}</p>}>
    <ReferenceField label={name} aria-labelledby={labelId} picking={picking}
      groups={[{ label: studioName('documents'), choices: [documentChoice('model', model)] }, stepGroup(sources)]}
      value={source.success ? current ? stepChoice(current) : { id: stepChoiceId(source.data.step), label: source.data.step }
        : documentChoice('model', model)}
      onChoose={editable ? (choice) => {
        const train = sources.find(({ id }) => stepChoiceId(id) === choice.id);
        setReason(undefined);
        start(train ? { step: train.id, checkpoint: 'last' } : null).catch((error: unknown) => setReason(apiErrorMessage(error)));
      } : undefined} />
  </FieldRow>;
}

export function StudioTrainAuthoringForm(props: Props) {
  const { editor, stepId, onChooseSweepValue } = props;
  const selection = props.selection ?? { kind: 'steps', stepIds: [stepId] };
  const { saved, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const methodLabel = useId();
  const [methodReason, setMethodReason] = useState<string>();
  const scope = runSelection(props.run, props.fieldRows, props.fieldAxes, props.chosen, props.step);
  const shows = (id: string) => !props.trainingDocument || inTrainingDocument(id);
  const definition = useQuery({
    queryKey: ['train-authoring', projectId, stepId, revision, scope],
    queryFn: ({ signal }) => readStudioTrainAuthoring(projectId, stepId, { revision, ...scope }, signal),
    enabled: !props.runPending,
    placeholderData: (previous) => previous?.project_id === projectId && previous.step_id === stepId ? previous : undefined,
  });
  const catalog = useAuthoringCatalog();
  const confirmRemoval = useRunRemoval();
  const numeric = useTrainFields(editor, stepId, scope, props.runPending);
  const active = useRef<AbortController | null>(null);
  const command = useRef<AbortController | null>(null);
  const shown = definition.data;
  const current = shown?.revision === revision ? shown : null;
  const refused = [definition.error, numeric.fields.error].find(isRefusal);
  const refusal = refused && apiErrorMessage(refused);
  const onRefused = useRef(props.onRefused);
  onRefused.current = props.onRefused;
  useEffect(() => { onRefused.current?.(refusal); }, [refusal]);
  useEffect(() => () => onRefused.current?.(undefined), []);

  useEffect(() => {
    const lifetime = new AbortController();
    active.current = lifetime;
    return () => lifetime.abort();
  }, [editor, stepId]);

  /** Makes one change or class sweep edit for the selected runs, confirming the runs it removes. */
  async function apply(change: Pick<StudioTrainAuthoringRequest, 'change' | 'class_sweep'>, label: string): Promise<string | undefined> {
    const lifetime = active.current;
    if (!current || !lifetime || lifetime.signal.aborted || pending || outside || command.current === lifetime) {
      return undefined;
    }
    command.current = lifetime;
    try {
      const recipe = await readStudioTrainAuthoring(projectId, stepId, { revision, ...scope, ...change }, editor.signal);
      if (recipe.refusal) return studioSweptParameterMissing(recipe.refusal);
      if (!recipe.files.length || !await confirmRemoval(sweepValueRuns(recipe.removed_values))) return undefined;
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) throw new ProjectEditConflictError();
      editor.editFiles(label, recipe.files, {
        reviewedRevision: revision,
        dependencies: recipe.documents.map((id) => ({ kind: 'document', id })),
        validate: validateRecipe(latest.saved.project, { kind: 'step', id: stepId }, recipe.documents),
      });
      await editor.flush();
      return undefined;
    } catch (error) {
      return apiErrorMessage(error);
    } finally {
      if (command.current === lifetime) command.current = null;
    }
  }

  const projected = numeric.fields.data?.fields ?? [];
  const members = (sweepId?: string) => new Set(props.step && sweepId && selection.kind !== 'steps'
    ? selectedRuns(props.step, selection).map((row) => row.values[sweepId]).filter(Boolean) : []);
  const fieldModel = (field: TrainFieldView) => fromFieldView(field, {
    id: JSON.stringify([projectId, stepId, props.fieldRows, field.id]),
    commit: numeric.edit && ((edit) => numeric.edit!(field.id, edit)),
    rowSelection: !!props.fieldRows?.length,
    selectedMembers: members(field.sweep?.id),
    chooseMember: field.sweep && onChooseSweepValue
      ? (valueId, extend) => onChooseSweepValue(field.sweep!.id, valueId, extend) : undefined,
  });
  const fields = shown?.fields ?? [];
  const methodField = fields.find(({ id }) => id === '/method');
  const methods = catalogEntries(catalog.data, 'training_methods');
  const method = catalog.data?.registries.training_methods?.entries
    .find((entry) => 'key' in entry && entry.key === methodField?.value);
  const settingsSchema = method && 'settings_schema' in method ? method.settings_schema : undefined;
  const choices = fields.filter(({ id, control }) => id !== '/method' && id !== settingsPrefix && control?.kind !== 'number' && shows(id));
  const steps = projected.filter(({ id }) => !id.startsWith(settingsPrefix + '/') && shows(id));
  const byPath = new Map(projected.map((field) => [field.id, field]));
  const editable = !!current && !pending && !outside;
  const mixed = shown?.mixed ?? [];
  /** The selected runs hold different values at this pointer, or at an object holding it. */
  const isMixed = (at: string) => mixed.some((differs) => at === differs || at.startsWith(differs + '/'));
  const classFields = new Map((shown?.classes ?? []).map((field) => [field.id, field]));
  const rowSelection = !!scope.rows?.length;
  /** A class field's kind, members and edits. Its sweep is changed only with the whole step selected. */
  function classProjection(field: StudioClassField): ClassProjection {
    const { sweep } = field;
    const kind = sweep ? 'sweep' : 'value';
    const keeps = sweep ? field.selected_value ?? sweep.members[0]?.id : undefined;
    const picked = members(sweep?.id);
    const kinds: FieldKindOffer[] = (['value', 'sweep'] as const).map((offer) => ({
      kind: offer,
      enabled: editable && !(rowSelection && offer !== kind || offer === 'sweep' && !sweep && !field.can_sweep ||
        offer === 'value' && !!sweep && !keeps),
    }));
    const commit: FieldCommit | undefined = editable && !(sweep && rowSelection) ? (edit) => {
      switch (edit.kind) {
        case 'set': return apply({ change: { field: field.id, value: edit.value } }, field.label);
        case 'to_sweep': return apply({ class_sweep: { field: field.id, edit: { kind: 'to_sweep' } } }, field.label);
        case 'to_value': return !edit.value_id ? Promise.resolve(undefined) : apply({ class_sweep: { field: field.id, edit: { kind: 'to_value', value_id: edit.value_id } } }, field.label);
        case 'add_member': case 'remove_member': return apply({ class_sweep: { field: field.id, edit } }, field.label);
        default: return Promise.resolve(undefined);
      }
    } : undefined;
    return { classes: field.classes, field: {
      kind, kinds, keeps, commit,
      members: sweep ? {
        values: sweep.members.map(({ id, label, value }) => ({ id, text: label, option: value, selected: picked.has(id) })),
        choose: onChooseSweepValue && ((id, extend) => onChooseSweepValue(sweep.id, id, extend)),
      } : undefined,
    } };
  }

  if (refusal !== undefined) return null;
  return (
    <StudioRegion state={studioReadState('definition', shown && numeric.fields.data,
      definition.error ?? numeric.fields.error ?? catalog.error, () => {
        void definition.refetch();
        void numeric.fields.refetch();
        void catalog.refetch();
      })}>
      <section aria-label={studioName('definition')} className="grid gap-1">
        {props.trains && !props.trainingDocument && <CheckpointStart editor={editor} stepId={stepId} step={props.step} trains={props.trains}
          picking={props.picking} editable={editable} />}
        {methodField && <div data-field={methodField.id} className="contents"><FieldRow label={methodField.label} labelId={methodLabel}
          below={methodReason && <p role="alert">{methodReason}</p>}>
          <CatalogPicker aria-labelledby={methodLabel} className="w-full" entries={methods}
            value={typeof methodField.value === 'string' && !isMixed('/method') ? methodField.value : null}
            placeholder={isMixed('/method') ? studioName('mixed') : undefined}
            disabled={!editable || !methodField.control}
            onChoose={(key) => { void apply({ change: { field: '/method', value: key } }, methodField.label).then(setMethodReason); }} />
        </FieldRow></div>}
        {choices.map((field) => <div key={field.id} data-field={field.id} className="contents"><ValueField model={choiceField(field,
          editable ? (value) => apply({ change: { field: field.id, value } }, field.label) : undefined)} /></div>)}
        {steps.map((field) => { const model = fieldModel(field); return <div key={model.id} data-field={field.id} className="contents">
          <ValueField model={model} />
        </div>; })}
        {settingsSchema && shown?.settings && <SettingsForm schema={settingsSchema} value={shown.settings}
          mixed={(path) => isMixed(settingsPrefix + pointer(path))}
          projected={(path) => {
            const field = byPath.get(settingsPrefix + pointer(path));
            return field && fieldModel(field);
          }}
          write={editable ? (path, value) => apply({ change: { field: settingsPrefix + pointer(path), value } }, humanizeLabel(path.at(-1) ?? '')) : undefined}
          choose={editable ? (path, key) => apply({ change: { field: settingsPrefix + pointer(path), value: key } }, humanizeLabel(path.at(-1) ?? '')) : undefined}
          classed={(path) => {
            const field = classFields.get(settingsPrefix + pointer(path));
            return field && classProjection(field);
          }} />}
      </section>
    </StudioRegion>
  );
}
