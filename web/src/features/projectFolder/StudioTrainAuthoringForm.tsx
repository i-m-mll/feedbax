import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { apiErrorMessage } from '@/api/request';
import { readStudioTrainAuthoring } from '@/api/studioAuthoring';
import {
  StudioDefinitionForm,
  type FieldEdit,
  type StudioDefinitionField,
  type StudioFieldCommand,
} from '@/components/pipeline/StudioDefinitionForm';
import { StudioSweepPicker } from '@/components/pipeline/StudioSweepOverview';
import type { PlanStepView } from '@/features/pipeline/planView';
import { selectedRuns, type PipelineSelection } from '@/features/pipeline/selection';
import type { AuthoredFileAddress, StudioTrainAuthoring, StudioTrainAuthoringRequest } from '@/generated/studioContracts';
import { authoredPointer, sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';

interface Props {
  readonly editor: ProjectEditController;
  readonly stepId: string;
  readonly selection?: PipelineSelection;
  readonly step?: PlanStepView;
  readonly component?: string;
  readonly picker?: () => void;
  readonly onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
  readonly renderAdditionalFields?: (projected: ReadonlySet<string>) => ReactNode;
  readonly view?: 'model' | 'task' | 'objective' | 'training';
  readonly focusFirstField?: boolean;
  readonly onFirstFieldFocused?: () => void;
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
    const actions: FieldEdit['kind'][] = control && !sweep && (!selection || selection.kind === 'steps') ? ['set'] : [];
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
      projected = { kind: 'sampler', id: sweep.id, label, ...sweep.rule };
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

export function StudioTrainAuthoringForm(props: Props) {
  const { editor, stepId, picker, onChooseSweepValue, renderAdditionalFields, focusFirstField, onFirstFieldFocused } = props;
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
  const shown = projection?.project_id === projectId && projection.step_id === stepId ? projection : null;
  const current = shown?.revision === revision ? shown : null;
  const disabled = !current || editing || pending > 0 || outside !== null;

  useLayoutEffect(() => {
    if (!focusFirstField || !current || disabled) return;
    const first = fieldSet.current?.querySelector<HTMLElement>('input:enabled, select:enabled');
    if (!first) return;
    first.focus();
    onFirstFieldFocused?.();
  }, [focusFirstField, current, disabled, onFirstFieldFocused]);

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

  async function apply({ fieldId, edit }: StudioFieldCommand) {
    const lifetime = active.current;
    const change = changeRequest(fieldId, edit);
    if (!current || !lifetime || lifetime.signal.aborted || pending || outside ||
        command.current === lifetime || !change) return;
    command.current = lifetime;
    setEditing(true);
    setEditError(null);
    try {
      const recipe = await readStudioTrainAuthoring(
        projectId, stepId, { revision, change }, lifetime.signal,
      );
      lifetime.signal.throwIfAborted();
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) {
        throw new ProjectEditConflictError();
      }
      const field = recipe.fields.find(({ id }) => id === fieldId);
      if (!field) throw new Error('The Train edit has no matching field.');
      if (!recipe.patches.length) return;
      const address = { kind: 'step', id: stepId } as const;
      editor.edit(address, field.label, recipe.patches, validateRecipe(latest.saved.project, address, recipe.documents));
      await editor.flush();
      lifetime.signal.throwIfAborted();
      picker?.();
    } catch (error) {
      if (!lifetime.signal.aborted) setEditError(apiErrorMessage(error));
    } finally {
      if (command.current === lifetime) command.current = null;
      if (!lifetime.signal.aborted) setEditing(false);
    }
  }

  const error = editError ?? loadError ?? (editorError ? apiErrorMessage(editorError) : null);
  const fields = shown ? formFields(shown, picker ? { ...props, component: undefined, selection: undefined } : props) : [];
  return (
    <>
      {error && <p role="alert">{error}</p>}
      {shown
        ? (
          <fieldset ref={fieldSet} disabled={disabled}>
            {picker ? <StudioSweepPicker fields={fields} onChoose={(fieldId) => {
              const value = fields.find(({ id }) => id === fieldId)?.resolved[0]?.value;
              if (value !== undefined) void apply({ fieldId, selection, edit: { kind: 'toSweep', value } });
            }} /> : <StudioDefinitionForm fields={fields} selection={selection} onChooseSweepValue={onChooseSweepValue}
              onEdit={(change) => { void apply(change); }} />}
          </fieldset>
        )
        : !error && <p role="status">Loading Train definition</p>}
      {shown && renderAdditionalFields?.(new Set(fields.map(({ id }) => id)))}
    </>
  );
}
