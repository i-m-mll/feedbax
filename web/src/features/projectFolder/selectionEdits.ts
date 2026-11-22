import {
  StepReferenceSchema, type InputBinding, type JsonValue, type ProjectValueEdit, type StepReference,
  type Step, type StepKind, type StudioProjectWindowMapProblem,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';

/** The step selections an input binds: one reference is a list of one; a binding of anything else has none. */
export function stepSelections(binding: InputBinding | null | undefined): StepReference[] | undefined {
  if (Array.isArray(binding)) return binding;
  const reference = StepReferenceSchema.safeParse(binding);
  return reference.success ? [reference.data] : undefined;
}

export type SelectionChange =
  | { kind: 'add'; reference: StepReference }
  /** `reference` names a step, or an operation's output in an Analyze card. */
  | { kind: 'replace'; index: number; reference: Pick<StepReference, 'step' | 'element' | 'product'> }
  | { kind: 'remove'; index: number };

/**
 * The binding after `change`. A selection whose step changes keeps its checkpoint, when it still reads a step's
 * output, and drops its restriction, which names the old step's runs. A list of one is written as its one reference.
 */
export function changedSelections(selections: readonly StepReference[], change: SelectionChange): InputBinding {
  const next = change.kind === 'add' ? [...selections, change.reference]
    : change.kind === 'remove' ? selections.filter((_, index) => index !== change.index)
      : selections.map((reference, index) => index !== change.index ? reference : {
        ...change.reference,
        ...(reference.checkpoint != null && change.reference.element == null && { checkpoint: reference.checkpoint }),
      });
  return next.length === 1 ? next[0] : next;
}

/** The edit that binds `role` of a step whose inputs are `inputs` to `binding`. */
export function bindingEdit(
  inputs: Readonly<Record<string, unknown>> | null | undefined, role: string, binding: InputBinding,
): ProjectValueEdit {
  return {
    op: inputs && role in inputs ? 'replace' : 'add', path: authoredPointer(['inputs', role]), value: binding as JsonValue,
  };
}

/**
 * The plan's refusal of selection `index` of `role`: a problem of the step at or under `inputs.<role>.<index>`, or
 * `inputs.<role>` when the role holds one reference rather than a list.
 */
export function selectionProblem(
  problems: readonly StudioProjectWindowMapProblem[], step: string, variant: string | null, role: string, index: number,
  listed: boolean,
): string | undefined {
  const path = listed ? `inputs.${role}.${index}` : `inputs.${role}`;
  return problems.find((problem) => problem.step === step && (problem.variant === null || problem.variant === variant) &&
    (problem.path === path || problem.path.startsWith(path + '.')))?.detail;
}

/**
 * What a wire from step `source` to step `target` adds: a selection of `source` on the one input of `target` that
 * takes a list of `source`'s kind. A target that has no such input, or already reads `source` there, takes none.
 */
export function connection(
  steps: Readonly<Record<string, Step | null>> | null | undefined, source: string, target: string,
): { role: string; edits: ProjectValueEdit[]; reads: string[] } | undefined {
  const from = steps?.[source];
  const to = steps?.[target];
  if (!from || !to || source === target || from.kind !== producerKind[to.kind]) return undefined;
  const roles = Object.entries(to.inputs ?? {}).filter(([role, binding]) =>
    (to.kind !== 'evaluate' || role === 'subject') &&
    stepSelections(binding)?.every(({ step }) => steps[step]?.kind === from.kind));
  if (roles.length !== 1) return undefined;
  const [[role, binding]] = roles;
  const selections = stepSelections(binding)!;
  if (selections.some(({ step }) => step === source)) return undefined;
  const reference = to.kind === 'evaluate' ? { step: source, checkpoint: selections[0].checkpoint ?? 'last' } : { step: source };
  const next = changedSelections(selections, { kind: 'add', reference });
  return {
    role,
    edits: [bindingEdit(to.inputs, role, next), ...to.kind === 'evaluate' ? subjectAxisEdits(to) : []],
    reads: [...new Set(stepSelections(next)!.map(({ step }) => step))],
  };
}

/** The kind of step whose runs an input of a step of each kind gathers as a list. */
const producerKind: Partial<Record<StepKind, StepKind>> = { evaluate: 'train', analyze: 'evaluate' };

/** The edits that fan an Evaluate step out over the members of its subject, when no axis does yet. */
export function subjectAxisEdits(step: Step): ProjectValueEdit[] {
  const axes = step.sweep?.axes;
  if (Object.values(axes ?? {}).some((axis) => axis && 'input' in axis && axis.input === 'subject')) return [];
  const axis = { input: 'subject' };
  if (!step.sweep) return [{ op: 'add', path: '/sweep', value: { axes: { subject: axis } } }];
  if (!axes) return [{ op: 'add', path: '/sweep/axes', value: { subject: axis } }];
  return [{ op: 'subject' in axes ? 'replace' : 'add', path: '/sweep/axes/subject', value: axis }];
}
