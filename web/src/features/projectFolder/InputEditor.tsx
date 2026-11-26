import { Plus, X } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { studioAccessibleName, studioFormat, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { Button, IconButton } from '@/components/ui/Button';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { StatusMark } from '@/components/ui/StatusMark';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice, type ReferenceGroup } from '@/components/values/ReferenceField';
import { outputName, type PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { InputBinding, InputRole, StepReference, StudioSavedTrainKey } from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';
import {
  selectionCheckpoints, type CheckpointKind, type CheckpointRange, type EditableCheckpointRule, type SelectionCheckpoint,
} from './evaluateAuthoring';
import {
  chosenDocument, documentChoice, documentGroup, documentKind, stepChoice, stepChoiceId, stepGroup, type DocumentKind,
} from './referenceChoices';
import type { DocumentNames } from './memberNames';
import { savedRunId } from './projectWindowPlan';
import { changedSelections, stepSelections, type SelectionProblems } from './selectionEdits';
import { CheckpointEditor } from './StudioEvaluateAuthoring';
import { outputChoice, outputChoiceId, type CardOutput } from './AnalyzeReferences';

/** What a gesture on an input asks for: its step selections changed, a document read instead, or nothing bound. */
export type InputGesture =
  | { kind: 'select'; binding: InputBinding }
  | { kind: 'document'; document: string; role: DocumentKind }
  | { kind: 'absent' };

/** The binding by which an input reads the project's document `document` of kind `role`. */
export function documentBinding(role: DocumentKind, document: string): InputBinding {
  const source = { document };
  switch (role) {
    case 'model': return { kind: 'initial_model', source };
    case 'loaded_model': return { kind: 'loaded_model', source };
    case 'task': return { kind: 'task', source };
    case 'objective': return { kind: 'objective', source };
  }
}

/** What the caller draws on selection `index`: its Runs control, and the plan's problems on the selection. */
export interface SelectionParts {
  readonly runs?: ReactNode;
  /** The selection's problems; its Runs control draws the `rule` one. */
  readonly problems?: SelectionProblems;
}

/** The Checkpoint of the input's selections, when the caller edits it itself; each group holds one rule. */
export interface CheckpointControl {
  readonly groups: readonly SelectionCheckpoint[];
  readonly disabled: boolean;
  readonly range?: (selections: readonly number[]) => CheckpointRange | undefined;
  readonly readOnly?: boolean;
  /** Why the last edit of the Checkpoint was refused. */
  readonly refusal?: string;
  onSave(rule: EditableCheckpointRule, selections: readonly number[]): void;
  readonly onToSweep?: () => void;
}

const allKinds: readonly CheckpointKind[] = ['last', 'at', 'list', 'every'];
const singleKinds: readonly CheckpointKind[] = ['last', 'at'];

/**
 * A selection whose runs keep runs of the step from running: the Cannot Run mark, and a link to the step it reads,
 * named by the run when it is one.
 */
function BlockedMark({ blockedBy, steps, onReveal }: {
  blockedBy: readonly StudioSavedTrainKey[];
  steps: readonly PlanStepView[];
  onReveal?: (step: string) => void;
}) {
  const [first] = blockedBy;
  const source = steps.find(({ id }) => id === first.step);
  if (!source) throw new Error(`The plan has no step ${first.step}.`);
  const run = () => {
    const row = source.runs.find(({ id }) => id === savedRunId(first));
    if (!row) throw new Error(`The plan has no run ${savedRunId(first)} of ${first.step}.`);
    return row.label;
  };
  const label = blockedBy.length === 1 ? studioFormat('stepRun', { step: source.label, run: run() }) : source.label;
  return <span className="inline-flex shrink-0 items-center gap-1.5">
    <StatusMark status="cannot_run" label={studioRunStatus('cannot_run')} />
    {onReveal ? <Button variant="bare" size="content" className="underline-offset-2 hover:underline"
      onClick={() => onReveal(first.step)}>{label}</Button> : <span>{label}</span>}
  </span>;
}

function documentOf(binding: InputBinding | null | undefined) {
  const source = (binding as { source?: { document?: unknown } } | null | undefined)?.source?.document;
  return typeof source === 'string' ? source : undefined;
}

/** A reference kept at a spelling another member later took: it names no choice, so picking that member repairs it. */
function unresolvedChoice(spelling: string): ReferenceChoice {
  return { id: `unresolved:${spelling}`, label: spelling };
}

/**
 * Any step input, drawn from the planner's entry for it: a header, with the add control where it takes several
 * selections; a row per selection with its picker; under each its Runs and, for a checkpoint product, its
 * Checkpoint. Add makes an empty row, and nothing is written until its step is chosen.
 */
export function InputEditor({ label, role, binding, self, steps, project, documentNames, outputs = [], absent, picking, refusal, selection,
  checkpoint, onReveal, onChange }: {
  label: string;
  role: InputRole;
  binding: InputBinding | null | undefined;
  /** The step the input belongs to, which it cannot read. */
  self: string;
  steps: readonly PlanStepView[];
  project: AuthoredProject;
  documentNames: DocumentNames;
  /** Other Analyze cards' operation outputs, which an input that takes them can read. */
  outputs?: readonly CardOutput[];
  /** What the step reads when the input is unbound, offered as the choice that unbinds it. */
  absent?: ReferenceChoice;
  picking?: ElementSelectionStore;
  /** Why the last edit of the input was refused. */
  refusal?: string;
  selection?: (index: number, reference: StepReference) => SelectionParts;
  checkpoint?: CheckpointControl;
  /** Selects a step a blocked selection links to. */
  onReveal?: (step: string) => void;
  onChange?: (gesture: InputGesture) => void;
}) {
  const labelId = useId();
  const [adding, setAdding] = useState(false);
  const source = (step: string) => role.producers.find(({ kind }) => steps.some(({ id, kind: candidate }) =>
    id === step && candidate === kind));
  const producers = steps.filter(({ id, kind }) => id !== self && role.producers.some((entry) => entry.kind === kind));
  const offered = role.producers.some((entry) => entry.operation_outputs) ? outputs : [];
  const stepGroups: ReferenceGroup[] = [
    ...role.producers.length ? [stepGroup(producers)] : [],
    ...offered.length ? [{ label: studioName('operations'), choices: offered.map(outputChoice) }] : [],
  ];
  const documentGroups = role.documents.length ? [documentGroup(project, documentNames, role.documents)] : [];
  const selections = stepSelections(binding);
  const document = documentOf(binding);
  const reads = (choice: ReferenceChoice): Pick<StepReference, 'step' | 'element' | 'product'> | undefined => {
    const step = producers.find(({ id }) => stepChoiceId(id) === choice.id);
    if (step) return { step: step.id };
    const output = offered.find((item) => outputChoiceId(item) === choice.id);
    return output && { step: output.step, element: output.element, product: output.product };
  };
  const shown = (reference: StepReference): ReferenceChoice => {
    if (reference.unresolved) return unresolvedChoice(reference.element != null && reference.product != null
      ? outputName(reference.step, reference.element, reference.product) : reference.step);
    if (reference.element != null && reference.product != null) {
      const output = { step: reference.step, element: reference.element, product: reference.product };
      return outputChoice(offered.find((item) => outputChoiceId(item) === outputChoiceId(output))
        ?? { ...output, name: outputName(output.step, output.element, output.product) });
    }
    const step = steps.find(({ id }) => id === reference.step);
    return step ? stepChoice(step) : { id: stepChoiceId(reference.step), label: reference.step };
  };
  const readsCheckpoint = ({ step }: StepReference) => source(step)?.product === 'checkpoint';
  const added = (reference: Pick<StepReference, 'step' | 'element' | 'product'>): StepReference => readsCheckpoint(reference)
    ? { ...reference, checkpoint: selections?.find(readsCheckpoint)?.checkpoint ?? 'last' } : reference;
  const choose = (index: number | 'new', choice: ReferenceChoice) => {
    if (!onChange) return;
    if (absent && choice.id === absent.id) return onChange({ kind: 'absent' });
    const document = chosenDocument(choice);
    const chosen = document !== undefined && documentKind(project, document, role.documents);
    if (chosen) return onChange({ kind: 'document', document, role: chosen });
    const reference = reads(choice);
    if (!reference) return;
    const current = selections ?? [];
    if (index === 'new') {
      setAdding(false);
      return onChange({ kind: 'select', binding: changedSelections(current, { kind: 'add', reference: added(reference) }) });
    }
    if (!selections) return onChange({ kind: 'select', binding: added(reference) });
    onChange({ kind: 'select', binding: changedSelections(current, { kind: 'replace', index, reference: added(reference) }) });
  };
  const groups = (only: boolean): ReferenceGroup[] => [
    ...absent ? [{ label: studioName('documents'), choices: [absent] }] : [],
    ...only || !role.gathers ? documentGroups : [],
    ...stepGroups,
  ];
  const checkpointSelections = (selections ?? []).flatMap((reference, index) =>
    readsCheckpoint(reference) ? [{ reference, index }] : []);
  const rules: readonly SelectionCheckpoint[] = checkpoint?.groups ?? (
    selectionCheckpoints(checkpointSelections.map(({ reference }) => reference)) ?? []
  ).map((group) => ({ ...group, selections: group.selections.map((index) => checkpointSelections[index].index) }));
  const sharedRule = selections && selections.length > 1 && rules.length === 1 && rules[0].selections.length === selections.length
    ? rules[0] : undefined;
  const rows: { value: ReferenceChoice | null; reference?: StepReference }[] = selections
    ? selections.map((reference) => ({ value: shown(reference), reference }))
    : role.gathers && !document ? []
      : [{ value: document !== undefined
        ? documentChoice(documentNames, documentKind(project, document, role.documents) ?? role.documents[0] ?? 'model', document)
        : absent ?? null }];
  const parts = rows.map(({ reference }, index) => reference ? selection?.(index, reference) : undefined);
  function checkpointEditor({ rule, selections: covered }: SelectionCheckpoint, step?: string) {
    const planned = covered.map((index) => parts[index]?.problems?.checkpoint).find((problem) => problem !== undefined);
    return <CheckpointEditor key={JSON.stringify(covered)} initial={rule} step={step}
      kinds={role.checkpoint_sets ? allKinds : singleKinds}
      range={checkpoint?.range?.(covered)} disabled={checkpoint ? checkpoint.disabled : !onChange}
      readOnly={checkpoint?.readOnly}
      problem={checkpoint?.refusal ?? planned}
      onSave={(next) => {
        if (checkpoint) return checkpoint.onSave(next, covered);
        if (!selections || !onChange) return;
        const changed = selections.map((reference, index) => covered.includes(index) ? { ...reference, checkpoint: next } : reference);
        onChange({ kind: 'select', binding: changed.length === 1 ? changed[0] : changed });
      }}
      onToSweep={checkpoint?.onToSweep} />;
  }
  const addable = onChange && role.gathers && document === undefined;
  /** A refusal with no picker to land on: the input's own, at its label's end. */
  const unplaced = !rows.length && !adding && refusal !== undefined;
  const removable = (count: number) => onChange && (role.gathers ? count > 1 : !role.required && !absent);
  return <div data-field="input" className="contents"><FieldRow label={label} labelId={labelId}
    actions={(unplaced || addable) && <>
      {unplaced && <FieldProblem reason={refusal} id={`${labelId}-problem`} />}
      {addable && <IconButton size="mark" label={studioAccessibleName('add', label)}
        icon={<Plus aria-hidden="true" />} disabled={adding} onClick={() => setAdding(true)} />}
    </>}
    below={sharedRule && checkpointEditor(sharedRule)}>
    <ul aria-labelledby={labelId} className="grid min-w-0 gap-1">
      {rows.map(({ value, reference }, index) => {
        const problems = parts[index]?.problems;
        const own = !sharedRule ? rules.find(({ selections: covered }) => covered[0] === index) : undefined;
        return <li key={index} className="grid min-w-0 gap-1">
          <div className="flex min-w-0 items-center gap-1">
            <ReferenceField className="flex-1" label={label} aria-labelledby={labelId} picking={picking}
              groups={groups(rows.length === 1)} value={value}
              problem={(index === 0 ? refusal : undefined) ?? problems?.picker}
              onChoose={onChange && ((choice) => { if (choice.id !== value?.id) choose(index, choice); })} />
            {problems?.blockedBy.length ? <BlockedMark blockedBy={problems.blockedBy} steps={steps} onReveal={onReveal} /> : null}
            {removable(rows.length) && reference && <IconButton size="mark"
              label={studioAccessibleName('remove', value?.label)} icon={<X aria-hidden="true" />}
              onClick={() => onChange!(selections!.length === 1 ? { kind: 'absent' }
                : { kind: 'select', binding: changedSelections(selections!, { kind: 'remove', index }) })} />}
          </div>
          {own && checkpointEditor(own, rules.length > 1 && reference ? shown(reference).label : undefined)}
          {parts[index]?.runs}
        </li>;
      })}
      {adding && <li className="flex min-w-0 items-center gap-1">
        <ReferenceField className="flex-1" label={label} aria-labelledby={labelId} picking={picking}
          groups={stepGroups} value={null} problem={rows.length ? undefined : refusal}
          onChoose={(choice) => choose('new', choice)} />
        <IconButton size="mark" label={studioAccessibleName('remove', label)} icon={<X aria-hidden="true" />}
          onClick={() => setAdding(false)} />
      </li>}
    </ul>
  </FieldRow></div>;
}
