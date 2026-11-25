import type { StudioResultState, StudioRunStatusId } from "@/components/activity/studioVocabulary";
import type {
  ReceiptValidity, StepKind, StudioActivityJob, StudioDefinitionChange, StudioProjectWindowMapProblem, StudioSavedTrainChange,
} from "@/generated/studioContracts";

export interface RunAddress {
  readonly stepId: string;
  readonly runId: string;
}

export interface PlanRunEdge {
  readonly from: RunAddress;
  readonly to: RunAddress;
}

export interface PlanSweepValueView {
  readonly id: string;
  /** The value's full name, which names its axis. */
  readonly label: string;
  /** The value's name where its axis is already named, as under a sweep's title. */
  readonly valueLabel: string;
}

export interface PlanSweepView {
  readonly id: string;
  readonly label: string;
  readonly values: readonly PlanSweepValueView[];
}

/** A retained result of a run whose validity the backend reports. */
export interface PlanResultView {
  readonly requestId: string;
  readonly jobId: string;
  readonly validity: ReceiptValidity;
}

export interface PlanRunView {
  readonly id: string;
  readonly label: string;
  readonly values: Readonly<Record<string, string>>;
  readonly status: StudioRunStatusId;
  /** The status with the results' validity folded in; what the run's mark draws. */
  readonly state: StudioResultState;
  readonly results: readonly PlanResultView[];
  /** The problems that keep the run from running; each names its place. */
  readonly problems?: readonly StudioProjectWindowMapProblem[];
  /** The definition change that left the run's retained result out of date. */
  readonly change?: StudioDefinitionChange | StudioSavedTrainChange;
  /** The run of another step this run reads, which cannot run, and so keeps this one from running. */
  readonly blockedBy?: { readonly stepId: string; readonly label: string };
  /** What the run's attempts report, which names nothing in the definition. */
  readonly reason?: string;
  readonly openingReason?: string;
  readonly openingErrors?: readonly string[];
  readonly progress?: NonNullable<StudioActivityJob["progress"]>;
  readonly collectionProgress?: NonNullable<StudioActivityJob["collection_progress"]>;
  readonly etaSeconds?: number;
  readonly sizeBytes?: number;
  readonly variant?: string;
}

export interface PlanInputView {
  readonly id: string;
  readonly label: string;
  readonly sourceStepId?: string;
  readonly subset?: {
    readonly terms: readonly string[];
    readonly count: number;
    readonly total: number;
    /** Whether the members are a lowest or highest term's ranks. */
    readonly ranked?: boolean;
    readonly memberGroups?: readonly {
      label?: string;
      members: readonly { id: string; label: string }[];
    }[];
  };
}

export interface PlanStepView {
  /** The step's key. */
  readonly id: string;
  /** The step's shown name, `shownName` of its key. */
  readonly label: string;
  readonly kind: StepKind;
  /** The label of the method a Synthesize step determines its model by. */
  readonly method?: string;
  readonly group?: string | null;
  readonly sweeps: readonly PlanSweepView[];
  readonly runs: readonly PlanRunView[];
  readonly inputs: readonly PlanInputView[];
  readonly documentIds?: Readonly<Record<string, string>>;
  readonly documentUses?: readonly string[];
  readonly statusCounts: Readonly<Partial<Record<StudioResultState, number>>>;
  /** How a run rule reading the step writes each of its axes' values, by axis and value id. */
  readonly ruleAxes?: Readonly<Record<string, Readonly<Record<string, string>>>>;
  /** Why the last gesture on the step was refused; its problem mark carries it. */
  readonly refusal?: string;
  /** A step being added: drawn from the gesture until the plan that has it is read, and taking no gestures. */
  readonly adding?: boolean;
}

// Presentation input for the regions. IDs, values, status and edges are supplied
// by the plan reader. These types do not describe an HTTP response.
export interface PlanView {
  readonly steps: readonly PlanStepView[];
  readonly edges: readonly PlanRunEdge[];
}

/** What Studio calls a document, step, operation or figure: its label, or its key when it has none. */
export function shownName(key: string, labelled?: { readonly label?: string | null } | null) {
  return labelled?.label ?? key;
}

/** An output's name: its card (when another card's), its operation's shown name, and its product. */
export function outputName(...parts: readonly string[]) {
  return parts.join(' · ');
}

const settled = new Set<StudioRunStatusId>(["succeeded", "stale"]);

export function runResultState(
  status: StudioRunStatusId, results: readonly PlanResultView[], checking: ReadonlySet<string>,
): StudioResultState {
  if (!settled.has(status) || !results.length) return status;
  if (results.some(({ requestId }) => checking.has(requestId))) return "checking";
  const states = new Set(results.map(({ validity }) => validity.state));
  return states.has("stale") ? "stale" : states.has("unchecked") ? "unchecked" : status;
}

/** Whether Check can run on a result: the backend lists the checks it would perform. */
export function resultCheckable({ validity }: PlanResultView) {
  return validity.state !== "current" && (validity.checks?.length ?? 0) > 0;
}

/** The code differences an Accept must name: every one the result's current reasons list. */
export function resultCodeDifferences({ validity }: PlanResultView) {
  if (validity.state === "current") return [];
  return (validity.reasons ?? []).flatMap((reason) => "basis" in reason ? reason.differences : []);
}
