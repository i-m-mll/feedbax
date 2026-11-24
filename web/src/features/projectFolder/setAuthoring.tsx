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
import { filesRemoval, type ConfirmRunRemoval } from './runRemoval';
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
  readonly confirmRemoval: ConfirmRunRemoval;
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

export function setAuthoring({ editor, project, applies, perform, state, select, confirmRemoval }: SetAuthoringInputs) {
  const refuseStep = (step: string) => (reason: string): Refusal => ({ kind: 'step', step, reason });

  /** Stores the edit `command` makes; one that removes runs first asks when some of them cost a confirmation. */
  async function store(name: string, command: SetCommand, removes = false): Promise<StudioSetEdit | undefined> {
    const { saved } = editor.getSnapshot();
    const edit = await prepareSetEdit(saved.project_id, { revision: saved.revision, command }, editor.signal);
    if (removes && !await confirmRemoval(filesRemoval(saved.project_id, edit.revision, edit.files, editor.signal))) return undefined;
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

  const refuseAxis = (step: string, axis: string) => (reason: string): Refusal => ({ kind: 'axis', step, axis, reason });

  /**
   * Save Set and Inline Set on a step's authored axis, and Add Entry on an axis whose Set has no entries (`empty`), which
   * Duplicate Run cannot copy one of.
   */
  function axisCommands(step: string, axis: string, empty = false): StudioCommand[] {
    const read = authoredAxis(project, step, axis);
    if (!read) return [];
    const command = (id: 'saveSet' | 'inlineSet' | 'addEntry', edit: SetCommand) => bindStudioCommand(id, () => {
      void perform(async () => {
        const stored = await store(studioAccessibleName(id, axis), edit);
        if (stored?.selected) select(selectSweepValue(state(), step, stored.selected.axis, stored.selected.value));
      }, refuseAxis(step, axis));
    }, { subject: axis, instance: JSON.stringify([step, axis]), enabled: applies });
    if ('values' in read) return [command('saveSet', { kind: 'save_set', step, axis })];
    if (!('set' in read)) return [];
    return [
      ...empty ? [command('addEntry', { kind: 'add_set_entry', step, axis })] : [],
      command('inlineSet', { kind: 'inline_set', step, axis }),
    ];
  }

  /**
   * Removes `value` from a step's authored axis of named values, or its entry from the Set the axis reads. The last
   * entry of a required Set is removed too, leaving the Set empty with its Problem. None removes the runs an optional
   * axis keeps, or a value of another kind of axis.
   */
  function removeValue(step: string, axis: string, value: string, label: string): StudioCommand | undefined {
    const read = authoredAxis(project, step, axis);
    if (value === unvalued || !read || !('set' in read || 'values' in read)) return undefined;
    const name = 'set' in read ? 'removeEntry' : 'removeValue';
    return bindStudioCommand('removeValue', () => {
      void perform(async () => {
        await store(studioAccessibleName(name, label), { kind: 'remove_value', step, axis, value }, true);
      }, refuseAxis(step, axis));
    }, { name, subject: label, instance: JSON.stringify([step, axis, value]), enabled: applies });
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
      changeKey: applies ? (key) => {
        void perform(async () => {
          await editor.memberCommand(studioAccessibleName('changeKey', kind),
            { kind: 'change_key', family: 'documents', member: document, key }, editor.getSnapshot().saved);
        }, refuseStep(step));
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

  return { duplicateRun, axisCommands, removeValue, axisPicker, applySet, combination };
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
