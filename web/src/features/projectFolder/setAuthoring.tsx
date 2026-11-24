import { useQuery } from '@tanstack/react-query';
import { prepareSetEdit, readSetsOffered } from '@/api/studioAuthoring';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import type { BarControl } from '@/components/ui/CommandBar';
import { MenuGroup, MenuGroupLabel, MenuItem, MenuRadioGroup, MenuRadioItem, MenuSeparator } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { shownName, type PlanRunView, type PlanStepView } from '@/features/pipeline/planView';
import { selectSweepValue, selectedRuns, type PipelineSelectionState } from '@/features/pipeline/selection';
import type { DocumentPickerModel } from '@/features/projectWindow/DocumentPicker';
import { bindStudioCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import type { Project, StudioSetEditRequest, StudioSetEdit } from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import type { ProjectEditController } from './projectEditController';
import { unvalued } from './projectWindowPlan';
import type { Refusal } from './refusals';
import { documentTreeOf } from './useProjectDocuments';

export const setSchema = 'feedbax.set_document.v1';

type SetCommand = StudioSetEditRequest['command'];
type Perform = (work: () => Promise<void>, refuse?: (reason: string) => Refusal | undefined) => Promise<void> | void;

export interface SetAuthoringInputs {
  readonly editor: ProjectEditController;
  readonly project: Project;
  /** Whether its commands and controls apply; an edit made while the workspace saves waits for it. */
  readonly applies: boolean;
  readonly perform: Perform;
  readonly state: () => PipelineSelectionState;
  readonly select: (state: PipelineSelectionState) => void;
}

/** The authored axis of a step's sweep, when the step authors it. */
function authoredAxis(project: Project, step: string, axis: string) {
  return project.steps?.[step]?.sweep?.axes?.[axis] ?? undefined;
}

/** The row key a run's id names. */
function runKey(run: PlanRunView) {
  const [variant, step, row] = JSON.parse(run.id) as [string, string, string];
  return { variant, step, row };
}

export function setAuthoring({ editor, project, applies, perform, state, select }: SetAuthoringInputs) {
  const refuseStep = (step: string) => (reason: string): Refusal => ({ kind: 'step', step, reason });

  async function store(name: string, command: SetCommand): Promise<StudioSetEdit | undefined> {
    const { saved } = editor.getSnapshot();
    const edit = await prepareSetEdit(saved.project_id, { revision: saved.revision, command }, editor.signal);
    const stored = await editor.storeEdit(name, { files: edit.files, created: edit.created }, { reviewedRevision: edit.revision });
    if (stored.status === 'refused') throw stored.error;
    await editor.flush();
    return edit;
  }

  /**
   * Gives each selected run of `step` a copy, and selects the copies. Runs outside the selection are chosen first.
   */
  function duplicateRun(step: PlanStepView, chosen?: readonly PlanRunView[]) {
    const current = state();
    const selected = selectedRuns(step, current.selection);
    const ids = new Set(selected.map(({ id }) => id));
    const inSelection = !!selected.length && (!chosen || chosen.every(({ id }) => ids.has(id)));
    const runs = inSelection ? selected : chosen ?? [];
    const terms = inSelection && current.selection?.kind === 'values' &&
      !current.selection.terms.some(({ valueIds }) => valueIds.includes(unvalued)) ? current.selection.terms : [];
    const whole = inSelection && current.selection?.kind === 'steps';
    return bindStudioCommand('duplicateRun', () => {
      if (!inSelection && runs.length) select({ selection: { kind: 'rows', stepId: step.id, runIds: runs.map(({ id }) => id) }, focus: null });
      void perform(async () => {
        const edit = await store(studioAccessibleName('duplicateRun', step.label), {
          kind: 'duplicate_run', run: runKey(runs[0]),
          rows: whole ? [] : runs.map((run) => runKey(run).row),
          axes: Object.fromEntries(terms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])),
        });
        if (edit?.selected) select(selectSweepValue(state(), step.id, edit.selected.axis, edit.selected.value));
      }, (reason) => ({ kind: 'runs', reasons: Object.fromEntries(runs.map(({ id }) => [id, reason])) }));
    }, {
      subject: step.label, instance: JSON.stringify([step.id, chosen?.map(({ id }) => id) ?? null]),
      enabled: applies && runs.length > 0 && (step.kind === 'train' || step.kind === 'evaluate'),
    });
  }

  /** Save Set and Inline Set on a step's authored axis. */
  function axisCommands(step: string, axis: string): StudioCommand[] {
    const read = authoredAxis(project, step, axis);
    if (!read) return [];
    const command = (id: 'saveSet' | 'inlineSet', kind: 'save_set' | 'inline_set') => bindStudioCommand(id, () => {
      void perform(async () => { await store(studioAccessibleName(id, axis), { kind, step, axis }); },
        (reason) => ({ kind: 'axis', step, axis, reason }));
    }, { subject: axis, instance: JSON.stringify([step, axis]), enabled: applies });
    return 'set' in read ? [command('inlineSet', 'inline_set')] : 'values' in read ? [command('saveSet', 'save_set')] : [];
  }

  /** The set an axis reads, as the document picker shows a view's document; none for an inline axis. */
  function axisPicker(step: string, axis: string): DocumentPickerModel | undefined {
    const read = authoredAxis(project, step, axis);
    if (!read || !('set' in read)) return undefined;
    const document = read.set;
    const kind = studioName('setDocument');
    const point = (next: string) => editor.editFiles(studioAccessibleName('select', kind), [{
      address: { kind: 'step', id: step },
      patches: [{ op: 'replace', path: authoredPointer(['sweep', 'axes', axis, 'set']), value: next }],
    }], { reviewedRevision: editor.getSnapshot().saved.revision, dependencies: [{ kind: 'document', id: next }] });
    return {
      label: 'setDocument', document, name: shownName(document, project.documents?.[document]),
      documents: documentTreeOf(project, [setSchema]),
      choose: applies ? (next) => {
        void perform(async () => {
          const stored = await point(next);
          if (stored.status === 'refused') throw stored.error;
          await editor.flush();
        }, refuseStep(step));
      } : undefined,
      duplicate: applies ? async () => {
        let copy: string | undefined;
        await perform(async () => {
          const result = await editor.memberCommand(studioAccessibleName('duplicate', kind),
            { kind: 'duplicate', family: 'documents', member: document }, editor.getSnapshot().saved,
            async ({ member }) => [{
              address: { kind: 'step', id: step },
              patches: [{ op: 'replace', path: authoredPointer(['sweep', 'axes', axis, 'set']), value: member }],
            }]);
          copy = result.member;
        }, refuseStep(step));
        return copy;
      } : undefined,
      rename: applies ? (label) => {
        void perform(async () => {
          await editor.memberCommand(studioAccessibleName('rename', kind),
            { kind: 'rename', family: 'documents', member: document, label }, editor.getSnapshot().saved);
        }, refuseStep(step));
      } : undefined,
      changeKey: applies ? {
        commit: (key) => {
          void perform(async () => {
            await editor.memberCommand(studioAccessibleName('changeKey', kind),
              { kind: 'change_key', family: 'documents', member: document, key }, editor.getSnapshot().saved);
          }, refuseStep(step));
        },
        held: new Set(Object.keys(project.documents ?? {})),
      } : undefined,
      detach: applies && 'base' in (project.documents?.[document] ?? {}) ? () => {
        void perform(async () => {
          await editor.memberCommand(studioAccessibleName('detach', kind),
            { kind: 'detach', member: document }, editor.getSnapshot().saved);
        }, refuseStep(step));
      } : undefined,
    };
  }

  /** Adds a sweep reading `set` to the step. */
  function applySet(step: string, set: string, done: () => void) {
    void perform(async () => {
      await store(studioAccessibleName('newSweep', set), { kind: 'apply_set', step, set });
      done();
    }, refuseStep(step));
  }

  /** The Cross or Zip of a step's axes, in its Runs header; absent while the step has fewer than two axes. */
  function combination(step: string): BarControl | undefined {
    const sweep = project.steps?.[step]?.sweep;
    const value = sweep?.combination ?? 'cross';
    if (Object.keys(sweep?.axes ?? {}).length < 2 || typeof value !== 'string') return undefined;
    const change = (next: string) => {
      if (next === value) return;
      void perform(async () => {
        const stored = await editor.editFiles(studioAccessibleName('combination', studioName(next === 'zip' ? 'zip' : 'cross')), [{
          address: { kind: 'step', id: step },
          patches: [{ op: sweep?.combination == null ? 'add' : 'replace', path: '/sweep/combination', value: next }],
        }], { reviewedRevision: editor.getSnapshot().saved.revision });
        if (stored.status === 'refused') throw stored.error;
        await editor.flush();
      }, refuseStep(step));
    };
    const items = (['cross', 'zip'] as const).map((id) => ({ value: id, label: studioName(id) }));
    return {
      id: 'combination',
      bar: <Select aria-label={studioName('combination')} value={value} disabled={!applies} onValueChange={change} items={items} />,
      menu: <MenuGroup>
        <MenuGroupLabel>{studioName('combination')}</MenuGroupLabel>
        <MenuRadioGroup value={value} onValueChange={(next: string) => change(next)} disabled={!applies}>
          {items.map(({ value: id, label }) => <MenuRadioItem key={id} value={id}>{label}</MenuRadioItem>)}
        </MenuRadioGroup>
      </MenuGroup>,
    };
  }

  return { duplicateRun, axisCommands, axisPicker, applySet, combination };
}

export type SetAuthoring = ReturnType<typeof setAuthoring>;

/** The sets offered to a step, below the fields of its Add Sweep menu. */
export function OfferedSets({ editor, step, choose }: {
  editor: ProjectEditController; step: string; choose: (set: string) => void;
}) {
  const { saved } = editor.getSnapshot();
  const offered = useQuery({
    queryKey: ['sets-offered', saved.project_id, saved.revision, step],
    queryFn: ({ signal }) => readSetsOffered(saved.project_id, { revision: saved.revision, step }, signal),
  });
  if (!offered.data?.length) return null;
  return <>
    <MenuSeparator />
    {offered.data.map((set) => <MenuItem key={set} onClick={() => choose(set)}>{set}</MenuItem>)}
  </>;
}
