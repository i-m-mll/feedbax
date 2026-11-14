import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiErrorMessage } from '@/api/request';
import { projectTrainFields, readStudioTrainAuthoring } from '@/api/studioAuthoring';
import { studioFormat, studioName, studioUnit } from '@/components/activity/studioVocabulary';
import {
  StudioDefinitionForm,
  StudioDocumentPicker,
  type FieldEdit,
  type StudioDefinitionField,
  type StudioFieldCommand,
} from '@/components/pipeline/StudioDefinitionForm';
import { StudioSweepPicker } from '@/components/pipeline/StudioSweepOverview';
import { StudioFieldControl } from '@/components/values/StudioFieldControl';
import type { PlanStepView } from '@/features/pipeline/planView';
import { selectedRuns, type PipelineSelection } from '@/features/pipeline/selection';
import type {
  AuthoredFileAddress, StudioReachTargetAuthoring, StudioReachTargetEdit, StudioSavedTrainKey, StudioTrainAuthoring,
  StudioTrainAuthoringRequest, TrainFieldRequest,
} from '@/generated/studioContracts';
import { authoredPointer, sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';

interface Props {
  readonly editor: ProjectEditController;
  readonly stepId: string;
  readonly selection?: PipelineSelection;
  readonly run?: StudioSavedTrainKey;
  readonly fieldRows?: readonly string[];
  readonly step?: PlanStepView;
  readonly component?: string;
  readonly picker?: () => void;
  readonly onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
  readonly renderAdditionalFields?: (projected: ReadonlySet<string>) => ReactNode;
  readonly view?: 'model' | 'task' | 'objective' | 'training';
  readonly documentPicker?: boolean;
  readonly taskBinding?: boolean;
  readonly focusFirstField?: boolean;
  readonly onFirstFieldFocused?: () => void;
  readonly onDetachedEditError?: (message: string) => void;
}

const viewPrefixes = {
  model: ['/graph'],
  task: ['/task'],
  objective: ['/objective'],
  training: ['/training_config', '/method_ref', '/method_payload'],
} as const;

function formFields(projection: StudioTrainAuthoring, { view, component, step, selection, picker }: Props): StudioDefinitionField[] {
  const componentPrefix = component ? authoredPointer(['graph', 'nodes', component, 'params']) : '/graph/nodes';
  const selected = step && selection?.kind !== 'steps' ? selectedRuns(step, selection ?? null) : [];
  return projection.fields.filter(({ id, sweep }) => {
    if (component) return id.startsWith(`${componentPrefix}/`);
    if (id.startsWith(`${componentPrefix}/`)) return Boolean(picker || (!view && sweep));
    return !view || viewPrefixes[view].some((path) => id === path || id.startsWith(`${path}/`));
  }).map(({ id, label, value, source, control, sweep, can_sweep }) => {
    const node = !component && id.startsWith(`${componentPrefix}/`) ? id.split('/')[3].replace(/~1/g, '/').replace(/~0/g, '~') : null;
    const canEdit = !selection || selection.kind === 'steps' &&
      (!id.startsWith('/objective/') || selection.stepIds.length === 1);
    const actions: FieldEdit['kind'][] = control && !sweep && canEdit ? ['set'] : [];
    if (can_sweep && (!selection || selection.kind === 'steps')) actions.push('toSweep');
    let projected: StudioDefinitionField['sweep'];
    if (sweep?.kind === 'list') {
      if (sweep.editable) actions.push('setSweepValue', 'nameSweepValue', 'addValue', 'toField');
      projected = { kind: 'list', id: sweep.id, label, values: sweep.values.map((item) => ({
        id: item.id, label: item.label ?? undefined, value: item.value,
        source: sweep.id, selected: selected.some((run) => run.values[sweep.id] === item.id),
      })) };
    } else if (sweep?.kind === 'range') {
      if (sweep.can_list) actions.push('toList');
      projected = { kind: 'range', id: sweep.id, label, first: sweep.rule.start, last: sweep.rule.stop,
        count: sweep.rule.count, scale: sweep.rule.scale === 'log' ? 'logarithmic' : 'linear' };
    } else if (sweep?.kind === 'sampler') {
      if (sweep.can_list) actions.push('toList');
      const { distribution, count, seed } = sweep.rule;
      projected = { kind: 'sampler', id: sweep.id, label, count, seed,
        distribution: typeof distribution === 'string' ? distribution : distribution.family };
    }
    return {
      id, label: node ? `${node} · ${label}` : label, actions, sweep: projected,
      control: control?.kind === 'number'
        ? { kind: 'number', min: control.min ?? undefined, max: control.max ?? undefined, step: control.step ?? undefined }
        : control ?? undefined,
      resolved: [{ id, value, source: source?.name }],
    };
  });
}

function changeRequest(field: string, edit: FieldEdit): StudioTrainAuthoringRequest['change'] {
  switch (edit.kind) {
    case 'set':
      return typeof edit.value === 'number' || typeof edit.value === 'string' ? { field, value: edit.value } : undefined;
    case 'toSweep':
      return { field, edit: { kind: edit.kind } };
    case 'addValue':
    case 'toList':
      return { field, edit: { kind: edit.kind, sweep_id: edit.sweepId } };
    case 'toField':
      return { field, edit: { kind: edit.kind, sweep_id: edit.sweepId, value_id: edit.keepValueId } };
    case 'nameSweepValue':
      return { field, edit: { kind: edit.kind, sweep_id: edit.sweepId, value_id: edit.valueId, label: edit.label } };
    case 'setSweepValue':
      return typeof edit.value === 'number'
        ? { field, edit: { kind: edit.kind, sweep_id: edit.sweepId, value_id: edit.valueId, value: edit.value } }
        : undefined;
  }
}

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

function ReachTargetForm({ target, stepId, disabled, onEdit }: {
  readonly target: StudioReachTargetAuthoring;
  readonly stepId: string;
  readonly disabled: boolean;
  readonly onEdit: (change: StudioReachTargetEdit) => void;
}) {
  const [input, setInput] = useState(target.input ?? '');
  const [x, setX] = useState(target.position?.x.toString() ?? '');
  const [y, setY] = useState(target.position?.y.toString() ?? '');
  const position = { x: Number(x), y: Number(y) };
  const complete = Boolean(input && x && y && Number.isFinite(position.x) && Number.isFinite(position.y));
  const changed = input !== target.input || position.x !== target.position?.x || position.y !== target.position?.y;
  const control = 'rounded-shell-control border border-shell-divider bg-shell-chrome px-2 py-1';
  return <form onSubmit={(event) => {
    event.preventDefault();
    if (!disabled && complete && changed) onEdit({ kind: 'bindReachTarget', input, position });
  }}>
    <fieldset disabled={disabled} className="flex flex-wrap items-end gap-3 px-3 py-2 text-sm">
      <legend className="px-3 font-medium">{studioName('reachGoal')} · {stepId}</legend>
      <label className="grid gap-1">
        {studioName('modelInput')}
        <select className={control} value={input} onChange={(event) => setInput(event.target.value)} required>
          <option value="" disabled />
          {target.inputs.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <label className="grid gap-1">
        {studioFormat('withUnit', { name: studioName('goalX'), unit: studioUnit('meter') })}
        <input className={`${control} w-24`} type="number" step="any" value={x} onChange={(event) => setX(event.target.value)} required />
      </label>
      <label className="grid gap-1">
        {studioFormat('withUnit', { name: studioName('goalY'), unit: studioUnit('meter') })}
        <input className={`${control} w-24`} type="number" step="any" value={y} onChange={(event) => setY(event.target.value)} required />
      </label>
      <button type="submit" disabled={!complete || !changed} className={`${control} disabled:opacity-50`}>
        {studioName('bindGoal')}
      </button>
    </fieldset>
  </form>;
}

export function StudioTrainAuthoringForm(props: Props) {
  const { editor, stepId, picker, documentPicker, taskBinding, onChooseSweepValue, renderAdditionalFields, focusFirstField, onFirstFieldFocused } = props;
  const selection = props.selection ?? { kind: 'steps', stepIds: [stepId] };
  const { saved, pending, outside, error: editorError } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const [projection, setProjection] = useState<StudioTrainAuthoring | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editError, setEditError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const active = useRef<AbortController | null>(null);
  const command = useRef<AbortController | null>(null);
  const fieldSet = useRef<HTMLFieldSetElement | null>(null);
  const numericRegion = useRef<HTMLDivElement | null>(null);
  const shown = projection?.project_id === projectId && projection.step_id === stepId ? projection : null;
  const current = shown?.revision === revision ? shown : null;
  const disabled = !current || editing || pending > 0 || outside !== null;
  const showNumeric = !props.component && !picker && !documentPicker && !taskBinding && (!props.view || props.view === 'training');
  const numericBody = { revision, run: props.run };
  const numeric = useQuery({
    queryKey: ['train-fields', projectId, stepId, numericBody],
    queryFn: ({ signal }) => projectTrainFields(projectId, stepId, numericBody, signal),
    enabled: showNumeric,
    retry: false,
    placeholderData: (previous, query) => query?.queryKey[1] === projectId && query.queryKey[2] === stepId ? previous : undefined,
  });

  async function editNumeric(field: string, edit: NonNullable<TrainFieldRequest['edit']>) {
    try {
      const recipe = await projectTrainFields(projectId, stepId, { ...numericBody, field, edit, rows: [...props.fieldRows ?? []] }, editor.signal);
      if (recipe.problems.length) return recipe.problems[0].reason;
      if (recipe.files.length) {
        editor.editFiles('Set Field', recipe.files, {
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

  useLayoutEffect(() => {
    if (!focusFirstField || !current || disabled || showNumeric && numeric.isFetching) return;
    const first = fieldSet.current?.querySelector<HTMLElement>('input:enabled, select:enabled') ??
      numericRegion.current?.querySelector<HTMLElement>('input:not([readonly])');
    if (!first) return;
    first.focus();
    onFirstFieldFocused?.();
  }, [focusFirstField, current, disabled, onFirstFieldFocused, showNumeric, numeric.isFetching]);

  useEffect(() => {
    const lifetime = new AbortController();
    active.current = lifetime;
    setEditError(null);
    setEditing(false);
    return () => lifetime.abort();
  }, [editor, stepId]);

  useEffect(() => {
    const request = new AbortController();
    setLoadError(null);
    setEditError(null);
    void readStudioTrainAuthoring(projectId, stepId, { revision }, request.signal).then(
      (result) => { if (!request.signal.aborted) setProjection(result); },
      (error) => { if (!request.signal.aborted) setLoadError(apiErrorMessage(error)); },
    );
    return () => request.abort();
  }, [projectId, stepId, revision]);

  async function apply(change: StudioTrainAuthoringRequest['change']) {
    const lifetime = active.current;
    if (!current || !lifetime || lifetime.signal.aborted || pending || outside ||
        command.current === lifetime || !change) return;
    command.current = lifetime;
    setEditing(true);
    setEditError(null);
    try {
      const recipe = await readStudioTrainAuthoring(
        projectId, stepId, { revision, change }, editor.signal,
      );
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) {
        throw new ProjectEditConflictError();
      }
      const label = 'field' in change
        ? recipe.fields.find(({ id }) => id === change.field)?.label
        : studioName('bindGoal');
      if (!label) throw new Error('The Train edit has no matching field.');
      if (!recipe.patches.length) return;
      const address = { kind: 'step', id: stepId } as const;
      editor.edit(address, label, recipe.patches, validateRecipe(latest.saved.project, address, recipe.documents));
      await editor.flush();
      if (!lifetime.signal.aborted) picker?.();
    } catch (error) {
      if (!lifetime.signal.aborted) setEditError(apiErrorMessage(error));
      else if (!editor.signal.aborted) props.onDetachedEditError?.(apiErrorMessage(error));
    } finally {
      if (command.current === lifetime) command.current = null;
      if (!lifetime.signal.aborted) setEditing(false);
    }
  }

  const error = editError ?? loadError ?? (editorError ? apiErrorMessage(editorError) : null);
  if (taskBinding) return <>
    {error && <p role="alert">{error}</p>}
    {shown?.reach_target && <ReachTargetForm key={shown.revision}
      target={shown.reach_target} stepId={stepId} disabled={disabled} onEdit={(change) => { void apply(change); }} />}
  </>;
  const shared = showNumeric ? numeric.data?.fields ?? [] : [];
  const projected = new Set(shared.map(({ id }) => id));
  const fields = shown ? formFields(shown, picker ? { ...props, component: undefined, selection: undefined } : props)
    .filter(({ id }) => !projected.has(id)) : [];
  const onEdit = ({ fieldId, edit }: StudioFieldCommand) => { void apply(changeRequest(fieldId, edit)); };
  if (documentPicker) return <>
    {fields[0] && <fieldset ref={fieldSet} disabled={disabled} className="studio-window-slot__document">
      <StudioDocumentPicker field={fields[0]} selection={selection} onEdit={onEdit} />
    </fieldset>}
    {error && <p className="min-w-0 truncate text-xs" role="alert">{error}</p>}
    {!shown && !error && <span className="sr-only" aria-busy="true">{studioName('definition')}</span>}
  </>;
  return (
    <>
      {error && !(numeric.data && loadError === error) && <p role="alert">{error}</p>}
      {showNumeric && numeric.isError && <p role="alert">{apiErrorMessage(numeric.error)}</p>}
      {shared.length > 0 && <div ref={numericRegion} className="grid gap-2 px-3 py-2">
        {shared.map((field) => <StudioFieldControl key={JSON.stringify([projectId, stepId, props.fieldRows, field.id])} field={field}
          commit={!numeric.isPlaceholderData && !pending && !outside ? (edit) => editNumeric(field.id, edit) : undefined}
          rowSelection={!!props.fieldRows?.length}
          multipleRows={(props.fieldRows?.length ?? 0) > 1}
          selectedValues={new Set(props.step && field.sweep && selection.kind !== 'steps'
            ? selectedRuns(props.step, selection).map((row) => row.values[field.sweep!.id]).filter(Boolean) : [])}
          onChooseValue={field.sweep && onChooseSweepValue
            ? (valueId, extend) => onChooseSweepValue(field.sweep!.id, valueId, extend) : undefined} />)}
      </div>}
      {shown
        ? (
          <fieldset ref={fieldSet} disabled={disabled}>
            {picker ? <StudioSweepPicker fields={fields} onChoose={(fieldId) => {
              const value = fields.find(({ id }) => id === fieldId)?.resolved[0]?.value;
              if (value !== undefined) void apply(changeRequest(fieldId, { kind: 'toSweep', value }));
            }} /> : <StudioDefinitionForm fields={fields} selection={selection} onChooseSweepValue={onChooseSweepValue}
              onEdit={onEdit} />}
          </fieldset>
        )
        : !error && !numeric.data && <div aria-busy="true" aria-label={studioName('definition')} />}
      {shown && renderAdditionalFields?.(new Set(fields.map(({ id }) => id)))}
    </>
  );
}
