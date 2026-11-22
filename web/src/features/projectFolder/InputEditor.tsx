import { Plus, X } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice, type ReferenceGroup } from '@/components/values/ReferenceField';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { InputBinding, InputRole, StepReference } from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';
import {
  selectionCheckpoints, type CheckpointKind, type CheckpointRange, type EditableCheckpointRule, type SelectionCheckpoint,
} from './evaluateAuthoring';
import { documentChoice, documentChoiceId, documentGroup, stepChoice, stepChoiceId, stepGroup } from './referenceChoices';
import { changedSelections, stepSelections } from './selectionEdits';
import { CheckpointEditor } from './StudioEvaluateAuthoring';
import { outputChoice, outputChoiceId, type CardOutput } from './AnalyzeReferences';

/** What a gesture on an input asks for: its step selections changed, a document read instead, or nothing bound. */
export type InputGesture =
  | { kind: 'select'; binding: InputBinding }
  | { kind: 'document'; document: string }
  | { kind: 'absent' };

/** What the caller draws on selection `index`: its Runs control, why the plan refuses it, its blocked mark. */
export interface SelectionParts {
  readonly runs?: ReactNode;
  readonly problem?: string;
  readonly mark?: ReactNode;
}

/** The Checkpoint of the input's selections, when the caller edits it itself; each group holds one rule. */
export interface CheckpointControl {
  readonly groups: readonly SelectionCheckpoint[];
  readonly disabled: boolean;
  readonly range?: (selections: readonly number[]) => CheckpointRange | undefined;
  readonly readOnly?: boolean;
  readonly refusal?: string;
  onSave(rule: EditableCheckpointRule, selections: readonly number[]): void;
  readonly onToSweep?: () => void;
}

const allKinds: readonly CheckpointKind[] = ['last', 'at', 'list', 'every'];
const singleKinds: readonly CheckpointKind[] = ['last', 'at'];

function documentOf(binding: InputBinding | null | undefined) {
  const source = (binding as { source?: { document?: unknown } } | null | undefined)?.source?.document;
  return typeof source === 'string' ? source : undefined;
}

/**
 * Any step input, drawn from the planner's entry for it: a header, with the add control where it takes several
 * selections; a row per selection with its picker; under each its Runs and, for a checkpoint product, its
 * Checkpoint. Add makes an empty row, and nothing is written until its step is chosen.
 */
export function InputEditor({ label, role, binding, self, steps, project, outputs = [], absent, picking, refusal, selection,
  checkpoint, onChange }: {
  label: string;
  role: InputRole;
  binding: InputBinding | null | undefined;
  /** The step the input belongs to, which it cannot read. */
  self: string;
  steps: readonly PlanStepView[];
  project: AuthoredProject;
  /** Other Analyze cards' operation outputs, which an input that takes them can read. */
  outputs?: readonly CardOutput[];
  /** What the step reads when the input is unbound, offered as the choice that unbinds it. */
  absent?: ReferenceChoice;
  picking?: ElementSelectionStore;
  /** Why the last edit of the input was refused. */
  refusal?: string;
  selection?: (index: number, reference: StepReference) => SelectionParts;
  checkpoint?: CheckpointControl;
  onChange?: (gesture: InputGesture) => void;
}) {
  const labelId = useId();
  const [adding, setAdding] = useState(false);
  const producer = role.producer;
  const producers = producer ? steps.filter(({ id, kind }) => kind === producer.kind && id !== self) : [];
  const offered = producer?.operation_outputs ? outputs : [];
  const stepGroups: ReferenceGroup[] = [
    ...producer ? [stepGroup(producers)] : [],
    ...offered.length ? [{ label: studioName('operations'), choices: offered.map(outputChoice) }] : [],
  ];
  const documentGroups = role.documents.map((kind) => documentGroup(project, kind));
  const selections = stepSelections(binding);
  const document = documentOf(binding);
  const reads = (choice: ReferenceChoice): Pick<StepReference, 'step' | 'element' | 'product'> | undefined => {
    const step = producers.find(({ id }) => stepChoiceId(id) === choice.id);
    if (step) return { step: step.id };
    const output = offered.find((item) => outputChoiceId(item) === choice.id);
    return output && { step: output.step, element: output.element, product: output.product };
  };
  const shown = (reference: StepReference): ReferenceChoice => {
    if (reference.element != null && reference.product != null) {
      const output = { step: reference.step, element: reference.element, product: reference.product };
      return outputChoice(offered.find((item) => outputChoiceId(item) === outputChoiceId(output))
        ?? { ...output, name: `${output.step}/${output.element}.${output.product}` });
    }
    const step = steps.find(({ id }) => id === reference.step);
    return step ? stepChoice(step) : { id: stepChoiceId(reference.step), label: reference.step };
  };
  const readsCheckpoint = producer?.product === 'checkpoint';
  const added = (reference: Pick<StepReference, 'step' | 'element' | 'product'>): StepReference => readsCheckpoint
    ? { ...reference, checkpoint: selections?.[0]?.checkpoint ?? 'last' } : reference;
  const choose = (index: number | 'new', choice: ReferenceChoice) => {
    if (!onChange) return;
    if (absent && choice.id === absent.id) return onChange({ kind: 'absent' });
    if (choice.id === documentChoiceId(choice.label)) return onChange({ kind: 'document', document: choice.label });
    const reference = reads(choice);
    if (!reference) return;
    const current = selections ?? [];
    if (index === 'new') {
      setAdding(false);
      return onChange({ kind: 'select', binding: changedSelections(current, { kind: 'add', reference: added(reference) }) });
    }
    if (!selections) return onChange({ kind: 'select', binding: added(reference) });
    onChange({ kind: 'select', binding: changedSelections(current, { kind: 'replace', index, reference }) });
  };
  const groups = (only: boolean): ReferenceGroup[] => [
    ...absent ? [{ label: studioName('documents'), choices: [absent] }] : [],
    ...only || !role.gathers ? documentGroups : [],
    ...stepGroups,
  ];
  const rules: readonly SelectionCheckpoint[] = !readsCheckpoint || !selections ? []
    : checkpoint?.groups ?? selectionCheckpoints(selections) ?? [];
  const sharedRule = selections && selections.length > 1 && rules.length === 1 && rules[0].selections.length === selections.length
    ? rules[0] : undefined;
  function checkpointEditor({ rule, selections: covered }: SelectionCheckpoint, step?: string) {
    return <CheckpointEditor key={JSON.stringify(covered)} initial={rule} step={step}
      kinds={role.checkpoint_sets ? allKinds : singleKinds}
      range={checkpoint?.range?.(covered)} disabled={checkpoint ? checkpoint.disabled : !onChange}
      readOnly={checkpoint?.readOnly}
      refusal={checkpoint?.refusal}
      onSave={(next) => {
        if (checkpoint) return checkpoint.onSave(next, covered);
        if (!selections || !onChange) return;
        const changed = selections.map((reference, index) => covered.includes(index) ? { ...reference, checkpoint: next } : reference);
        onChange({ kind: 'select', binding: changed.length === 1 ? changed[0] : changed });
      }}
      onToSweep={checkpoint?.onToSweep} />;
  }
  const removable = (count: number) => onChange && (role.gathers ? count > 1 : !role.required && !absent);
  const rows: { value: ReferenceChoice | null; reference?: StepReference }[] = selections
    ? selections.map((reference) => ({ value: shown(reference), reference }))
    : role.gathers && !document ? []
      : [{ value: document !== undefined ? documentChoice(role.documents[0] ?? 'model', document) : absent ?? null }];
  return <FieldRow label={label} labelId={labelId}
    actions={onChange && role.gathers && document === undefined && <IconButton size="mark" label={studioAccessibleName('add', label)}
      icon={<Plus aria-hidden="true" />} disabled={adding} onClick={() => setAdding(true)} />}
    below={(refusal || sharedRule) && <>
      {refusal && <p role="alert">{refusal}</p>}
      {sharedRule && checkpointEditor(sharedRule)}
    </>}>
    <ul aria-labelledby={labelId} className="grid min-w-0 gap-1">
      {rows.map(({ value, reference }, index) => {
        const parts = reference && selection?.(index, reference);
        const own = !sharedRule ? rules.find(({ selections: covered }) => covered[0] === index) : undefined;
        return <li key={index} className="grid min-w-0 gap-1">
          <div className="flex min-w-0 items-center gap-1">
            <ReferenceField className="flex-1" label={label} aria-labelledby={labelId} picking={picking}
              groups={groups(rows.length === 1)} value={value}
              onChoose={onChange && ((choice) => { if (choice.id !== value?.id) choose(index, choice); })} />
            {parts?.mark}
            {removable(rows.length) && reference && <IconButton size="mark"
              label={studioAccessibleName('remove', value?.label)} icon={<X aria-hidden="true" />}
              onClick={() => onChange!(selections!.length === 1 ? { kind: 'absent' }
                : { kind: 'select', binding: changedSelections(selections!, { kind: 'remove', index }) })} />}
          </div>
          {parts?.problem && <p role="alert">{parts.problem}</p>}
          {own && checkpointEditor(own, rules.length > 1 && reference ? shown(reference).label : undefined)}
          {parts?.runs}
        </li>;
      })}
      {adding && <li className="flex min-w-0 items-center gap-1">
        <ReferenceField className="flex-1" label={label} aria-labelledby={labelId} picking={picking}
          groups={stepGroups} value={null} onChoose={(choice) => choose('new', choice)} />
        <IconButton size="mark" label={studioAccessibleName('remove', label)} icon={<X aria-hidden="true" />}
          onClick={() => setAdding(false)} />
      </li>}
    </ul>
  </FieldRow>;
}
