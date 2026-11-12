import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';
import { apiErrorMessage } from '@/api/request';
import { readStudioTrainAuthoring } from '@/api/studioAuthoring';
import {
  StudioDefinitionForm,
  type StudioDefinitionField,
  type StudioFieldCommand,
} from '@/components/pipeline/StudioDefinitionForm';
import type { AuthoredFileAddress, StudioTrainAuthoring } from '@/generated/studioContracts';
import { sameAuthoredFile, type AuthoredProject } from './authoredEdits';
import { ProjectEditConflictError, type ProjectEditController, type ProjectEditValidation } from './projectEditController';

interface Props {
  readonly editor: ProjectEditController;
  readonly stepId: string;
  readonly focusFirstField?: boolean;
  readonly onFirstFieldFocused?: () => void;
}

function formFields(projection: StudioTrainAuthoring): StudioDefinitionField[] {
  return projection.fields.map(({ id, label, value, source, control }) => ({
    id,
    label,
    control: control?.kind === 'number'
      ? { kind: 'number', min: control.min ?? undefined, max: control.max ?? undefined, step: control.step ?? undefined }
      : control ?? undefined,
    resolved: [{ id, value, source: source?.name }],
    actions: control ? ['set'] : [],
  }));
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

export function StudioTrainAuthoringForm({ editor, stepId, focusFirstField, onFirstFieldFocused }: Props) {
  const { saved, pending, outside, error: editorError } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { project_id: projectId, revision } = saved;
  const [projection, setProjection] = useState<StudioTrainAuthoring | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editError, setEditError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const active = useRef<AbortController | null>(null);
  const command = useRef<AbortController | null>(null);
  const fields = useRef<HTMLFieldSetElement | null>(null);
  const current = projection?.project_id === projectId && projection.step_id === stepId &&
    projection.revision === revision ? projection : null;
  const disabled = editing || pending > 0 || outside !== null;

  useLayoutEffect(() => {
    if (!focusFirstField || !current || disabled) return;
    const first = fields.current?.querySelector<HTMLElement>('input:enabled, select:enabled');
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
    if (!current || !lifetime || lifetime.signal.aborted || pending || outside ||
        command.current === lifetime || edit.kind !== 'set' ||
        (typeof edit.value !== 'number' && typeof edit.value !== 'string')) return;
    command.current = lifetime;
    setEditing(true);
    setEditError(null);
    try {
      const recipe = await readStudioTrainAuthoring(
        projectId, stepId, { revision, change: { field: fieldId, value: edit.value } }, lifetime.signal,
      );
      lifetime.signal.throwIfAborted();
      const latest = editor.getSnapshot();
      if (latest.saved.revision !== revision || latest.pending || latest.outside) {
        throw new ProjectEditConflictError();
      }
      const field = recipe.fields.find(({ id }) => id === fieldId);
      if (!field) throw new Error('The Train edit has no matching field.');
      const address = { kind: 'step', id: stepId } as const;
      editor.edit(address, field.label, recipe.patches, validateRecipe(latest.saved.project, address, recipe.documents));
      await editor.flush();
    } catch (error) {
      if (!lifetime.signal.aborted) setEditError(apiErrorMessage(error));
    } finally {
      if (command.current === lifetime) command.current = null;
      if (!lifetime.signal.aborted) setEditing(false);
    }
  }

  const error = editError ?? loadError ?? (editorError ? apiErrorMessage(editorError) : null);
  return (
    <>
      {error && <p role="alert">{error}</p>}
      {current
        ? (
          <fieldset ref={fields} disabled={disabled}>
            <StudioDefinitionForm
              fields={formFields(current)} selection={{ kind: 'steps', stepIds: [stepId] }}
              onEdit={(change) => { void apply(change); }}
            />
          </fieldset>
        )
        : !error && <p role="status">Loading Train definition</p>}
    </>
  );
}
