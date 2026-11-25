import {
  StepReferenceSchema, type InputBinding, type JsonValue, type ProjectValueEdit, type StepReference,
  type SelectedAxisValue, type SelectedAxisValueProblem, type Step, type StepKind, type StudioProjectWindowMapProblem,
  type StudioSavedTrainKey, type SubsetTerms, type TextSpan,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { savedRunId, storedClauses } from './projectWindowPlan';

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

/** A plan's problem on a run rule, and the characters of the saved rule it is about. */
export interface RuleProblem {
  readonly reason: string;
  readonly span?: TextSpan;
  /** The selected axis value the rule names that no longer exists, and the axis's values that do. */
  readonly value?: SelectedAxisValueProblem;
}

const ruleProblem = (problem: StudioProjectWindowMapProblem): RuleProblem =>
  ({ reason: problem.detail, ...problem.span && { span: problem.span }, ...problem.selection_value && { value: problem.selection_value } });

/**
 * `terms` with the missing value of `value`'s clause and axis replaced by `chosen`, or dropped where that clause already
 * wants `chosen`.
 */
export function replacedAxisValue(terms: SubsetTerms, value: SelectedAxisValueProblem, chosen: SelectedAxisValue): SubsetTerms {
  const clauses = terms.clauses ?? [];
  const clause = clauses[value.clause_index];
  if (!clause?.[value.axis]) throw new Error(`Selection has no clause ${value.clause_index} naming ${value.axis}.`);
  const values = clause[value.axis];
  const kept = values.flatMap((selected) => selected.id !== value.missing.id ? [selected]
    : values.some(({ id }) => id === chosen.id) ? [] : [chosen]);
  return { ...terms, clauses: storedClauses(clauses.map((each, index) =>
    index === value.clause_index ? { ...each, [value.axis]: kept } : each)) };
}

/** A selection's problems, each on the field its path names. */
export interface SelectionProblems {
  /** A problem of the selection that names neither its Runs nor its Checkpoint, on its picker. */
  readonly picker?: string;
  /** A problem at the selection's `where`, on its Runs field, with the term of the rule it is about. */
  readonly rule?: RuleProblem;
  /** A problem at the selection's `checkpoint`, or the role's when the role lists selections. */
  readonly checkpoint?: string;
  /** The runs the selection reads that cannot run, and keep runs of the step from running. */
  readonly blockedBy: readonly StudioSavedTrainKey[];
}

/** A problem's path in the planner's dotted form; the service reports some as JSON Pointers. */
const dotted = (path: string) => path.startsWith('/') ? path.slice(1).split('/').join('.') : path;

/**
 * The problems the plan reports on selection `index` of `role` of `step`, placed by path: under `inputs.<role>.<index>`,
 * or `inputs.<role>` when the role holds one reference rather than a list.
 */
export function selectionProblems(
  problems: readonly StudioProjectWindowMapProblem[], step: string, variant: string | null, role: string, index: number,
  listed: boolean,
): SelectionProblems {
  const own = listed ? `inputs.${role}.${index}` : `inputs.${role}`;
  const placed: { picker?: string; rule?: RuleProblem; checkpoint?: string; blockedBy: StudioSavedTrainKey[] } = { blockedBy: [] };
  for (const problem of problems) {
    if (problem.step !== step || (problem.variant !== null && problem.variant !== variant)) continue;
    const path = dotted(problem.path);
    const rest = path === own || path.startsWith(own + '.') ? path.slice(own.length)
      : listed && (path === `inputs.${role}.checkpoint` || path.startsWith(`inputs.${role}.checkpoint.`)) ? path.slice(`inputs.${role}`.length)
        : listed && index === 0 && path === `inputs.${role}` ? '' : undefined;
    if (rest === undefined) continue;
    if (problem.code === 'blocked_input') {
      if (problem.blocked_by && !placed.blockedBy.some((key) => savedRunId(key) === savedRunId(problem.blocked_by!))) {
        placed.blockedBy.push(problem.blocked_by);
      }
      continue;
    }
    if (rest === '.where' || rest.startsWith('.where.')) {
      placed.rule ??= ruleProblem(problem);
      continue;
    }
    placed[rest === '.checkpoint' || rest.startsWith('.checkpoint.') ? 'checkpoint' : 'picker'] ??= problem.detail;
  }
  return placed;
}

/** The problem the plan reports on the rule of `step`'s sweep exception `id`. */
export function exceptionRuleProblem(
  problems: readonly StudioProjectWindowMapProblem[], step: string, variant: string | null, id: string,
): RuleProblem | undefined {
  const own = `sweep.exceptions.${id}.where`;
  const problem = problems.find((problem) => problem.step === step && (problem.variant === null || problem.variant === variant)
    && (dotted(problem.path) === own || dotted(problem.path).startsWith(own + '.')));
  return problem && ruleProblem(problem);
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
