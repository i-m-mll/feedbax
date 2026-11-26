import { useQuery } from '@tanstack/react-query';
import { shownError } from '@/api/shownErrors';
import { prepareSetEdit, readSetsOffered } from '@/api/studioAuthoring';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import type { BarControl } from '@/components/ui/CommandBar';
import { MenuGroup, MenuGroupLabel, MenuItem, MenuRadioGroup, MenuRadioItem, MenuSeparator } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import type { PlanRunView, PlanStepView } from '@/features/pipeline/planView';
import { selectSweepValue, selectedRuns, type PipelineSelectionState } from '@/features/pipeline/selection';
import type { DocumentPickerModel } from '@/features/projectWindow/DocumentPicker';
import { bindStudioCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  setDocumentSchemaId, type Project, type SetDocument, type SetVariableOrigin, type StudioSetEditRequest, type StudioSetEdit,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { documentPayload } from './documentPayload';
import type { ProjectEditController } from './projectEditController';
import { unvalued } from './projectWindowPlan';
import type { Refusal } from './refusals';
import { sendWithStore, type AskStore } from './storeChoice';
import { editedElement } from '@/components/ui/ChoiceMenu';
import { filesRemoval, type ConfirmRunRemoval } from './runRemoval';
import type { EntryTask, TaskSetVariable, WeightEdit } from './task/taskModel';
import { labelOf } from './memberLabels';
import { documentTreeOf } from './useProjectDocuments';


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
  /** Asks where an edit that reaches beyond the selection is written. */
  readonly ask: AskStore;
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

export function setAuthoring({ editor, project, applies, perform, state, select, confirmRemoval, ask }: SetAuthoringInputs) {
  const refuseStep = (step: string) => (reason: string): Refusal => ({ kind: 'step', step, reason });

  /** Stores the edit `command` makes; one that removes runs first asks when some of them cost a confirmation. */
  async function store(name: string, command: SetCommand, removes = false): Promise<StudioSetEdit | undefined> {
    const { saved } = editor.getSnapshot();
    const from = document.activeElement;
    const edit = await sendWithStore(ask, () => editedElement(from, from), () => editor.flush(),
      (store_choice) => prepareSetEdit(saved.project_id, { revision: saved.revision, command, store_choice }, editor.signal),
      (answer) => answer.store_choices);
    if (!edit) return undefined;
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
   * Save Set and Inline Set on a step's authored axis, which convert between its values and the Model differences of a
   * Set and so apply only while the step reads a Model document (`model`); Add Entry on an axis whose Set has no
   * entries (`empty`), which Duplicate Run cannot copy one of; and To Field on a Train or Evaluate Set axis, which moves the Set
   * into the run's trials.
   */
  function axisCommands(step: string, axis: string, { model, empty }: { model: boolean; empty: boolean }): StudioCommand[] {
    const read = authoredAxis(project, step, axis);
    if (!read) return [];
    const command = (id: 'saveSet' | 'inlineSet' | 'addEntry' | 'toField', edit: SetCommand, enabled = applies) => bindStudioCommand(id, () => {
      void perform(async () => {
        const stored = await store(studioAccessibleName(id, axis), edit);
        if (stored?.selected) select(selectSweepValue(state(), step, stored.selected.axis, stored.selected.value));
      }, refuseAxis(step, axis));
    }, { subject: axis, instance: JSON.stringify([step, axis]), enabled });
    if ('values' in read) return [command('saveSet', { kind: 'save_set', step, axis }, applies && model)];
    if (!('set' in read)) return [];
    const kind = project.steps?.[step]?.kind;
    return [
      ...empty ? [command('addEntry', { kind: 'add_set_entry', step, axis })] : [],
      command('inlineSet', { kind: 'inline_set', step, axis }, applies && model),
      ...(kind === 'train' || kind === 'evaluate') ? [command('toField', { kind: 'set_to_field', step, axis })] : [],
    ];
  }

  /** Moves a Set the step's trials apply back to the sweep axis `axis`, whose runs it was moved out of. */
  function toSweep(step: string, axis: string): StudioCommand {
    return bindStudioCommand('toSweep', () => {
      void perform(async () => {
        await store(studioAccessibleName('toSweep', axis), { kind: 'set_to_sweep', step, axis });
      }, refuseStep(step));
    }, { subject: axis, instance: JSON.stringify([step, axis]), enabled: applies });
  }

  /**
   * Sets one weight of the Set the step's trials apply from the sweep axis `axis`: of an entry, or of None with a null
   * entry, and of its `side` where entries co-occur. Its refusal lands on the weight.
   */
  function fieldWeight(step: string, axis: string): TaskSetVariable['weigh'] {
    if (!applies) return undefined;
    return (entry, side, weight) => editor.command(async (): Promise<WeightEdit> => {
      const stored = await store(studioAccessibleName('weight', axis), {
        kind: 'set_field_weight', step, axis, entry, weight, ...side ? { part: side } : {},
      });
      return { kind: stored ? 'stored' : 'kept' };
    }).catch((error: unknown) => ({ kind: 'refused', reason: shownError(error) }));
  }

  /** The entry `entry` of the Set `set` that the step's axis `axis` reads, with its own Task variables and links. */
  function entryTask({ step, axis, set, entry }: Omit<SetVariableOrigin, 'variable'>): EntryTask | undefined {
    const stored = (documentPayload(project.documents, editor.getSnapshot().saved.composed, set) as SetDocument | undefined)?.entries[entry];
    if (!stored) return undefined;
    return {
      key: JSON.stringify([step, axis, entry]),
      variables: stored.task_variables ?? {},
      links: stored.task_links ?? {},
      write: applies ? (name, change) => editor.command(async () => {
        await store(name, { kind: 'set_entry_task', step, axis, entry, ...change });
        return undefined;
      }).catch((error: unknown) => shownError(error)) : undefined,
    };
  }

  /** The one entry of a Set its sweep authors that the selected runs of `step` read; none unless exactly one is. */
  function selectedEntry(step: PlanStepView): EntryTask | undefined {
    const runs = selectedRuns(step, state().selection);
    const chosen = Object.entries(project.steps?.[step.id]?.sweep?.axes ?? {}).flatMap(([axis, read]) => {
      if (!read || !('set' in read)) return [];
      const entries = new Set(runs.map((run) => run.values[axis]));
      const [entry] = entries;
      return entries.size === 1 && entry !== undefined && entry !== unvalued ? [{ step: step.id, axis, set: read.set, entry }] : [];
    });
    return chosen.length === 1 ? entryTask(chosen[0]) : undefined;
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
    const composes = documentPayload(project.documents, editor.getSnapshot().saved.composed, document) !== undefined;
    const point = (next: string) => editor.editFiles(studioAccessibleName('select', kind), [{
      address: { kind: 'step', id: step },
      patches: [{ op: 'replace', path: authoredPointer(['sweep', 'axes', axis, 'set']), value: next }],
    }], { reviewedRevision: editor.getSnapshot().saved.revision, dependencies: [{ kind: 'document', id: next }] });
    return {
      label: 'setDocument', document, name: labelOf(project.documents, document),
      documents: documentTreeOf(project, [setDocumentSchemaId]),
      choose: applies ? (next) => {
        void perform(async () => {
          const stored = await point(next);
          if (stored.status === 'refused') throw stored.error;
          await editor.flush();
        }, refuseStep(step));
      } : undefined,
      duplicate: applies && composes ? async () => {
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
      detach: applies && composes && 'base' in (project.documents?.[document] ?? {}) ? () => {
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

  return { duplicateRun, axisCommands, toSweep, fieldWeight, entryTask, selectedEntry, removeValue, axisPicker, applySet, combination };
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
    {offered.data.map((set) => <MenuItem key={set} onClick={() => choose(set)}>{labelOf(saved.project.documents, set)}</MenuItem>)}
  </>;
}
