import type { StudioRunStatusId } from "@/components/activity/studioVocabulary";
import type { StudioActivityJob } from "@/generated/studioContracts";

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
  readonly label: string;
}

export interface PlanSweepView {
  readonly id: string;
  readonly label: string;
  readonly values: readonly PlanSweepValueView[];
}

export interface PlanRunView {
  readonly id: string;
  readonly label: string;
  readonly values: Readonly<Record<string, string>>;
  readonly status: StudioRunStatusId;
  readonly reason?: string;
  readonly progress?: NonNullable<StudioActivityJob["progress"]>;
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
    readonly members?: readonly { id: string; label: string }[];
  };
}

export interface PlanStepView {
  readonly id: string;
  readonly label: string;
  readonly kind: "train" | "evaluate" | "analyze" | "report";
  readonly group?: string | null;
  readonly sweeps: readonly PlanSweepView[];
  readonly runs: readonly PlanRunView[];
  readonly inputs: readonly PlanInputView[];
  readonly documentIds?: Readonly<Record<string, string>>;
  readonly documentUses?: readonly string[];
  readonly statusCounts: Readonly<Partial<Record<StudioRunStatusId, number>>>;
}

// Presentation input for the regions. IDs, values, status and edges are supplied
// by the plan reader. These types do not describe an HTTP response.
export interface PlanView {
  readonly steps: readonly PlanStepView[];
  readonly edges: readonly PlanRunEdge[];
}
