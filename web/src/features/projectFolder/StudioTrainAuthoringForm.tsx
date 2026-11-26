import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useQuery } from '@tanstack/react-query';
import { isRefusal } from '@/api/request';
import { shownError } from '@/api/shownErrors';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import {
  catalogEntries, projectTrainFields, readStudioTrainAuthoring, useAuthoringCatalog,
} from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { labelOf } from './memberLabels';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { FieldRow } from '@/components/ui/FieldRow';
import { MenuItem, MenuSubmenu } from '@/components/ui/Menu';
import { settingsFields } from '@/components/values/SettingsForm';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { FieldChange, ValueField } from '@/components/values/ValueField';
import { pointer } from '@/components/values/adapters/jsonSchema';
import { fromFieldView, type ProjectedEdit } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { PlanStepView } from '@/features/pipeline/planView';
import { selectedRuns, type PipelineSelection } from '@/features/pipeline/selection';
import type {
  AuthoredFileAddress, FieldResetEdit, JsonValue, StudioAuthoringField, StudioClassField, StudioSavedTrainKey, StudioTrainAuthoringRequest,
  TrainFieldRequest, TrainFieldView,
} from '@/generated/studioContracts';
import type { ChangeMark, FieldCommit, FieldKindOffer, FieldModel } from '@/components/values/fieldModel';
import { reasonOf } from '@/components/values/fieldModel';
import { baseOf } from './documentPayload';
import type { ClassProjection } from '@/components/values/adapters/jsonSchema';
import { sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { projectRead, ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';
import { filesRemoval, sweepValueRuns, useRunRemoval } from './runRemoval';
import { sendWithStore, useAskStore } from './storeChoice';
import { editedElement } from '@/components/ui/ChoiceMenu';
import { inTrainingDocument, trainFieldDocument } from './trainingDocument';

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
  /** The form is the Training view's: it shows the training document's fields. */
  readonly trainingDocument?: boolean;
  /** The training document's users an edit is made for, the step among them; without them, the step alone. */
  readonly chosen?: readonly string[];
  /** Takes the reason the service refuses to read the step's definition, which belongs on the step, and none once it reads. */
  readonly onRefused?: (reason: string | undefined) => void;
  /** The anchors of fields another editor draws with `StudioStepField`, which the form leaves out. */
  readonly omitted?: readonly string[];
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
  return (before, _after, direction) => {
    // Replaying an acknowledged gesture uses the saved revision. The writer can omit defaults from its optimistic form.
    if ((direction === 'edit' && !sameAuthoredFile(before, authored, address)) ||
        JSON.stringify(before.sweep) !== JSON.stringify(authored.sweep) ||
        documents.some((id) => JSON.stringify(before.documents?.[id]) !== JSON.stringify(authored.documents?.[id])) ||
        bases.some((base) => !sameAuthoredFile(before, authored, base))) {
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
    ...projectRead(editor, revision, (signal) => projectTrainFields(projectId, stepId!, body, signal)),
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
  const { outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { body, fields } = useTrainFieldsQuery(editor, stepId, selection, runPending);
  const confirmRemoval = useRunRemoval();
  const askStore = useAskStore();
  function edit(field: string, change: ProjectedEdit | FieldResetEdit): Promise<string | undefined> {
    const from = document.activeElement;
    return editor.command(async (saved) => {
      const answer = await sendWithStore(askStore, () => editedElement(from, from), () => editor.flush(),
        (store_choice) => projectTrainFields(
          saved.project_id, stepId, { ...body, revision: saved.revision, field, edit: change, store_choice }, editor.signal),
        (projected) => projected.store_choices);
      if (!answer) return undefined;
      const { problems = [], files = [], created = [], dependencies = [], removed_values: removedValues } = answer;
      if (problems[0]) return problems[0].reason;
      if ((files.length || created.length) && await confirmRemoval(sweepValueRuns(removedValues))) {
        const stored = editor.storeEdit(studioName('setField'), { files, created }, {
          reviewedRevision: saved.revision,
          dependencies: dependencies.map((id) => ({ kind: 'document', id })),
          validate: validateRecipe(saved.project, { kind: 'step', id: stepId }, dependencies),
        });
        await editor.flush();
        const settled = await stored;
        if (settled.status === 'refused') return shownError(settled.error);
      }
      return undefined;
    }).catch(shownError);
  }
  return { fields, edit: outside ? undefined : edit };
}

/** A field a sweep can start from, or a group of them as the Inspector's form groups them. */
export type SweepTarget =
  | { readonly kind: 'field'; readonly label: string; readonly anchor: string }
  | { readonly kind: 'group'; readonly label: string; readonly members: readonly SweepTarget[] };

/** The targets among `model` and its parts: a group with one target is that target. */
function sweepTargets(model: FieldModel): SweepTarget[] {
  if (model.parts?.length && model.type.kind !== 'vector' || model.type.kind === 'group' || model.type.kind === 'list') {
    const members = (model.parts ?? []).flatMap(sweepTargets);
    return members.length > 1 ? [{ kind: 'group', label: model.label, members }] : members;
  }
  const offered = model.kind !== 'sweep' && model.kinds.some(({ kind, enabled }) => kind === 'sweep' && enabled);
  return offered && model.anchor ? [{ kind: 'field', label: model.label, anchor: model.anchor }] : [];
}

export function SweepTargetItem({ target, onChoose }: { target: SweepTarget; onChoose(anchor: string): void }) {
  return target.kind === 'field'
    ? <MenuItem onClick={() => onChoose(target.anchor)}>{target.label}</MenuItem>
    : <MenuSubmenu label={target.label}>
      {target.members.map((member) => <SweepTargetItem key={member.label} target={member} onChoose={onChoose} />)}
    </MenuSubmenu>;
}

/**
 * The step's fields a sweep can start from, as items of the menu at the end of its card's shape line, nested as the
 * Inspector's form nests them. Choosing one goes to the field; the sweep is made by the text it commits.
 */
export function SweepFieldItems({ editor, stepId, onChoose }: {
  editor: ProjectEditController; stepId: string; onChoose(anchor: string): void;
}) {
  const form = useTrainForm({ editor, stepId });
  return <>{form.fields.flatMap(sweepTargets).map((target) =>
    <SweepTargetItem key={target.label} target={target} onChoose={onChoose} />)}</>;
}

function choiceField(field: StudioAuthoringField, commit?: (value: JsonValue) => Promise<string | undefined>, changed?: ChangeMark): FieldModel {
  return {
    ...rawField({
      id: field.id, label: field.label, value: field.value, changed,
      type: field.control?.kind === 'choice' ? { kind: 'choice', options: field.control.options } : { kind: 'readOnly' },
      change: field.control ? commit : undefined,
    }),
    anchor: field.id,
  };
}

/**
 * A Train or Evaluate step's form for the runs `props` selects: its definition, and its fields in the order and
 * nesting the Inspector draws them, each with its anchor. The Add Sweep menu reads the same fields for the whole step.
 */
function useTrainForm(props: Props) {
  const { editor, stepId, onChooseSweepValue } = props;
  const selection = props.selection ?? { kind: 'steps', stepIds: [stepId] };
  const { saved, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  /** The changed mark of the field at `id`, a pointer in the definition: the base of the document it is read from. */
  const changeMark = (id: string, reset?: () => Promise<string | undefined>): ChangeMark | undefined => {
    const base = baseOf(saved.project.documents, trainFieldDocument(saved.project.steps?.[stepId], id));
    return base === undefined ? undefined : { from: { kind: 'base', name: labelOf(saved.project.documents, base) }, reset };
  };
  const { project_id: projectId, revision } = saved;
  const methodLabel = useId();
  const methodChangeId = useId();
  const [methodReason, setMethodReason] = useState<string>();
  const scope = runSelection(props.run, props.fieldRows, props.fieldAxes, props.chosen, props.step);
  const shows = (id: string) => !props.trainingDocument || inTrainingDocument(id);
  const definition = useQuery({
    queryKey: ['train-authoring', projectId, stepId, revision, scope],
    ...projectRead(editor, revision, (signal) => readStudioTrainAuthoring(projectId, stepId, { revision, ...scope }, signal)),
    enabled: !props.runPending,
    placeholderData: (previous) => previous?.project_id === projectId && previous.step_id === stepId ? previous : undefined,
  });
  const catalog = useAuthoringCatalog();
  const confirmRemoval = useRunRemoval();
  const askStore = useAskStore();
  const numeric = useTrainFields(editor, stepId, scope, props.runPending);
  const active = useRef<AbortController | null>(null);
  const shown = definition.data;
  const refused = [definition.error, numeric.fields.error].find(isRefusal);
  const refusal = refused && shownError(refused);
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
  function apply(change: Pick<StudioTrainAuthoringRequest, 'change' | 'class_sweep' | 'reset'>, label: string): Promise<string | undefined> {
    const lifetime = active.current;
    if (!lifetime) return Promise.resolve(undefined);
    const from = document.activeElement;
    return editor.command(async (saved) => {
      if (lifetime.signal.aborted) return undefined;
      const { revision } = saved;
      const recipe = await sendWithStore(askStore, () => editedElement(from, from), () => editor.flush(),
        (store_choice) => readStudioTrainAuthoring(saved.project_id, stepId, { revision, ...scope, ...change, store_choice }, editor.signal),
        (answer) => answer.store_choices);
      if (!recipe) return undefined;
      if (!recipe.files.length && !recipe.created?.length ||
          !await confirmRemoval(sweepValueRuns(recipe.removed_values))) return undefined;
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) throw new ProjectEditConflictError();
      const stored = editor.storeEdit(label, recipe, {
        reviewedRevision: revision,
        dependencies: recipe.documents.map((id) => ({ kind: 'document', id })),
        validate: validateRecipe(latest.saved.project, { kind: 'step', id: stepId }, recipe.documents),
      });
      await editor.flush();
      const settled = await stored;
      if (settled.status === 'refused') return shownError(settled.error);
      return undefined;
    }).catch(shownError);
  }

  const projected = numeric.fields.data?.fields ?? [];
  const members = (sweepId?: string) => new Set(props.step && sweepId && selection.kind !== 'steps'
    ? selectedRuns(props.step, selection).map((row) => row.values[sweepId]).filter(Boolean) : []);
  const fieldModel = (field: TrainFieldView): FieldModel => ({ ...fromFieldView(field, numeric.fields.data!, {
    id: JSON.stringify([projectId, stepId, props.fieldRows, field.id]),
    commit: numeric.edit && ((edit) => numeric.edit!(field.id, edit)),
    changed: field.changed ? changeMark(field.id, numeric.edit && (() => numeric.edit!(field.id, { kind: 'reset' }))) : undefined,
    rowSelection: !!props.fieldRows?.length,
    selectedMembers: members(field.sweep?.id),
    chooseMember: field.sweep && onChooseSweepValue
      ? (valueId, extend) => onChooseSweepValue(field.sweep!.id, valueId, extend) : undefined,
  }), anchor: field.id });
  const definedFields = shown?.fields ?? [];
  const methodField = definedFields.find(({ id }) => id === '/method');
  const methods = catalogEntries(catalog.data, 'training_methods');
  const method = catalog.data?.registries.training_methods?.entries
    .find((entry) => 'key' in entry && entry.key === methodField?.value);
  const settingsSchema = method && 'settings_schema' in method ? method.settings_schema : undefined;
  const choices = definedFields.filter(({ id, control }) => id !== '/method' && id !== settingsPrefix && control?.kind !== 'number' && shows(id));
  const steps = projected.filter(({ id }) => !id.startsWith(settingsPrefix + '/') && shows(id));
  const byPath = new Map(projected.map((field) => [field.id, field]));
  const editable = !!shown && !outside;
  const mixed = shown?.mixed ?? [];
  /** The selected runs hold different values at this pointer, or at an object holding it. */
  const isMixed = (at: string) => mixed.some((differs) => at === differs || at.startsWith(differs + '/'));
  const classFields = new Map((shown?.classes ?? []).map((field) => [field.id, field]));
  /** The changed mark of the value at `id`, a pointer in the definition, when a difference of its document sets it. */
  const changedAt = (id: string): ChangeMark | undefined => shown?.changed?.includes(id)
    ? changeMark(id, editable ? () => apply({ reset: id }, studioName('reset')) : undefined) : undefined;
  const methodChange = changedAt('/method');
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

  const fields: FieldModel[] = [
    ...choices.map((field) => choiceField(field,
      editable ? (value) => apply({ change: { field: field.id, value } }, field.label) : undefined, changedAt(field.id))),
    ...steps.map(fieldModel),
    ...settingsSchema && shown?.settings ? settingsFields({
      schema: settingsSchema, value: shown.settings,
      anchor: (path) => settingsPrefix + pointer(path),
      mixed: (path) => isMixed(settingsPrefix + pointer(path)),
      changed: (path) => changedAt(settingsPrefix + pointer(path)),
      projected: (path) => {
        const field = byPath.get(settingsPrefix + pointer(path));
        return field && fieldModel(field);
      },
      write: editable ? (path, value) => apply({ change: { field: settingsPrefix + pointer(path), value } }, humanizeLabel(path.at(-1) ?? '')) : undefined,
      choose: editable ? (path, key) => apply({ change: { field: settingsPrefix + pointer(path), value: key } }, humanizeLabel(path.at(-1) ?? '')) : undefined,
      classed: (path) => {
        const field = classFields.get(settingsPrefix + pointer(path));
        return field && classProjection(field);
      },
    }) : [],
  ];
  return {
    shown, refusal, editable, methodField, methods, methodChange, methodReason, setMethodReason, methodLabel, methodChangeId,
    isMixed, apply, fields, definition, numeric, catalog,
  };
}

export function StudioTrainAuthoringForm(props: Props) {
  const { editor, stepId } = props;
  const {
    shown, refusal, editable, methodField, methods, methodChange, methodReason, setMethodReason, methodLabel, methodChangeId,
    isMixed, apply, fields, definition, numeric, catalog,
  } = useTrainForm(props);
  if (refusal !== undefined) return null;
  return (
    <StudioRegion state={studioReadState('definition', shown && numeric.fields.data,
      definition.error ?? numeric.fields.error ?? catalog.error, () => {
        void definition.refetch();
        void numeric.fields.refetch();
        void catalog.refetch();
      })}>
      <section aria-label={studioName('definition')} className="grid gap-1">
        {methodField && <div data-field={methodField.id} className="contents"><FieldRow label={methodField.label} labelId={methodLabel}
          below={methodReason && <p role="alert">{methodReason}</p>}>
          <CatalogPicker aria-labelledby={methodLabel} aria-describedby={methodChange ? methodChangeId : undefined} className="w-full" entries={methods}
            trailing={methodChange && <FieldChange change={methodChange} id={methodChangeId}
              onReset={() => { void methodChange!.reset!().then((refusal) => setMethodReason(reasonOf(refusal))); }} />}
            value={typeof methodField.value === 'string' && !isMixed('/method') ? methodField.value : null}
            placeholder={isMixed('/method') ? studioName('mixed') : undefined}
            disabled={!editable || !methodField.control}
            onChoose={(key) => { void apply({ change: { field: '/method', value: key } }, methodField.label).then(setMethodReason); }} />
        </FieldRow></div>}
        {fields.filter(({ anchor }) => !anchor || !props.omitted?.includes(anchor))
          .map((model) => <ValueField key={model.id} model={model} />)}
      </section>
    </StudioRegion>
  );
}

/** One field of the step's form, at `anchor`, drawn inside another editor under its own `label`. */
export function StudioStepField({ anchor, label, ...props }: Props & { anchor: string; label: string }) {
  const model = useTrainForm({ ...props, onRefused: undefined }).fields.find((field) => field.anchor === anchor);
  return model ? <ValueField model={{ ...model, label }} /> : null;
}
