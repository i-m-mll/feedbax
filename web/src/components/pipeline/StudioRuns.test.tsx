// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { PlanStepView, PlanView } from "@/features/pipeline/planView";
import { emptyPipelineSelection, selectStep, type PipelineSelection, type PipelineSelectionState } from "@/features/pipeline/selection";
import { bindStudioCommand } from "@/features/projectWindow/studioCommands";
import { StudioRunControls, StudioRuns, StudioStepInspector } from "./StudioRuns";

const train: PlanStepView = {
  id: "train",
  kind: "train",
  label: "Train arm",
  inputs: [],
  sweeps: [
    { id: "rate", label: "Rate", values: [{ id: "low", label: "0.001" }, { id: "high", label: "0.003" }] },
    { id: "seed", label: "Seed", values: [{ id: "one", label: "1" }, { id: "two", label: "2" }] },
  ],
  runs: [
    { id: "a", label: "A", values: { rate: "low", seed: "one" }, status: "succeeded", state: "succeeded", results: [] },
    { id: "b", label: "B", values: { rate: "high", seed: "one" }, status: "succeeded", state: "succeeded", results: [] },
    { id: "c", label: "C", values: { rate: "high", seed: "two" }, status: "stale", state: "stale", results: [], reason: "Learning rate 0.001 → 0.003" },
  ],
  statusCounts: { succeeded: 2, stale: 1 },
};
const evaluate: PlanStepView = {
  id: "eval", kind: "evaluate", label: "Test reach", inputs: [], sweeps: [], statusCounts: { not_run: 1 },
  runs: [{ id: "a", label: "Straight reach", values: {}, status: "not_run", state: "not_run", results: [] }],
};
const plan: PlanView = {
  steps: [train, evaluate],
  edges: [{ from: { stepId: train.id, runId: "b" }, to: { stepId: evaluate.id, runId: "a" } }],
};
const result = { kind: "result", missing: [] } as const;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function showRuns(step: PlanStepView, run?: (selection: PipelineSelection | null) => void) {
  vi.spyOn(Element.prototype, "clientHeight", "get").mockReturnValue(360);
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(44);
  function TestRuns() {
    const [state, setState] = useState<PipelineSelectionState>(() => selectStep(emptyPipelineSelection, step.id));
    const [pivot, setPivot] = useState(false);
    const selection = { step, state, onSelectionChange: setState };
    return <>
      <StudioRunControls commands={run ? [bindStudioCommand("run", () => run(state.selection))] : undefined}
        pivotCommand={step.sweeps.length > 1 ? bindStudioCommand("pivot", () => setPivot(!pivot)) : undefined}
        pivotActive={pivot} />
      <StudioRuns plan={plan} region={result} pivotActive={pivot} {...selection} />
      <StudioStepInspector plan={plan} {...selection} onChooseStep={(stepId) => setState(selectStep(state, stepId))} />
    </>;
  }
  return render(<TestRuns />);
}

it("identifies runs by ordinal and the fields that differ, and runs the selected rows", () => {
  const perform = vi.fn();
  showRuns(train, perform);
  const table = screen.getByRole("table", { name: "Runs" });
  expect(within(table).getAllByRole("columnheader").map(({ textContent }) => textContent)).toEqual(["Rate", "Seed", "Status", "Progress"]);
  expect(within(table).getAllByRole("rowheader").map((header) => header.querySelector("[data-run-ordinal]")?.textContent))
    .toEqual(["1", "2", "3"]);
  expect(within(table).queryByRole("cell", { name: "B" })).toBeNull();
  expect(screen.queryByRole("group", { name: "Rate" })).toBeNull();

  fireEvent.click(within(table).getByRole("button", { name: /^\d+ B$/ }));
  fireEvent.click(within(table).getByRole("button", { name: /^\d+ C$/ }), { metaKey: true });
  expect(within(table).getByRole("button", { name: /^\d+ C$/ }).getAttribute("aria-current")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "Run" }));
  expect(perform).toHaveBeenLastCalledWith({ kind: "rows", stepId: "train", runIds: ["b", "c"] });
  expect(within(table).queryByText("Learning rate 0.001 → 0.003")).toBeNull();
  expect(screen.getByText("Learning rate 0.001 → 0.003")).toBeDefined();
  expect(screen.getByRole("button", { name: "Select Used By Test reach 1 of 1" })).toBeDefined();
});

it("puts the opening stage in the status tooltip and keeps provider error lines in the inspector", () => {
  const summary = "Acquiring: RunPod HTTP 400: Placement refused";
  const line = "The GPU had no slots with this CUDA version.";
  showRuns({
    ...train,
    runs: [
      { ...train.runs[0], status: "starting", openingReason: "Opening machine: connecting" },
      { ...train.runs[1], status: "failed", openingReason: summary, reason: summary, openingErrors: [line] },
    ],
    statusCounts: { starting: 1, failed: 1 },
  });
  const table = within(screen.getByRole("table", { name: "Runs" }));
  expect(table.queryByText("Opening machine: connecting")).toBeNull();
  expect(table.getAllByRole("cell").filter((cell) => cell.getAttribute("aria-description")).map((cell) => cell.querySelector<HTMLElement>("[data-app-tip]")?.dataset.appTip))
    .toEqual(["Opening machine: connecting", summary]);
  expect(table.queryByText(line)).toBeNull();
  fireEvent.click(table.getByRole("button", { name: /^\d+ B$/ }));
  expect(screen.getByText(line)).toBeDefined();
});

it("virtualizes an authoritative 1,000-row result and keeps its full accessible count", () => {
  const runs = Array.from({ length: 1000 }, (_, index) => ({ ...train.runs[0], id: String(index), label: `Reach ${index}` }));
  showRuns({ ...train, runs, statusCounts: { succeeded: runs.length } });
  const table = screen.getByRole("table", { name: "Runs" });
  expect(table.getAttribute("aria-rowcount")).toBe("1001");
  expect(within(table).getByRole("button", { name: /^\d+ Reach 0$/ })).toBeDefined();
  expect(within(table).getAllByRole("row").length).toBeLessThan(50);
  expect(within(table).queryByRole("button", { name: "Reach 999" })).toBeNull();
  const viewport = table.parentElement!;
  Object.defineProperties(viewport, {
    scrollHeight: { value: runs.length * 44 },
    scrollTo: { value: ({ top }: ScrollToOptions) => fireEvent.scroll(viewport, { target: { scrollTop: top } }) },
  });
  fireEvent.scroll(viewport, { target: { scrollTop: 44 * 500 } });
  const middle = within(table).getByRole("button", { name: /^\d+ Reach 500$/ });
  expect(within(table).queryByRole("button", { name: "Reach 0" })).toBeNull();
  middle.focus();
  fireEvent.keyDown(middle, { key: "End" });
  expect(document.activeElement).toBe(within(table).getByRole("button", { name: /^\d+ Reach 999$/ }));
});

it("names pivot cells from both axes and retains explicit rows without coordinates", () => {
  showRuns({ ...train, runs: [...train.runs, { id: "explicit", label: "Custom reach", values: {}, status: "not_run", state: "not_run", results: [] }] });
  fireEvent.click(screen.getByRole("button", { name: "Pivot" }));
  const pivot = within(screen.getByRole("table", { name: "Pivot" }));
  const run = pivot.getByRole("button", { name: "Select Rate 0.003 · Seed 1 · Done" });
  fireEvent.click(run);
  expect(run.getAttribute("aria-pressed")).toBe("true");
  const rows = within(screen.getByRole("table", { name: "Runs" }));
  expect(rows.getByRole("button", { name: /^\d+ Custom reach$/ })).toBeDefined();
});

it("draws a Results differ reason as one row per change", () => {
  const receipt = { object_id: "content:a", sha256: "a".repeat(64), size_bytes: 1 };
  const run = {
    ...train.runs[0], status: "succeeded" as const, state: "stale" as const,
    results: [{ requestId: "r", jobId: "j", validity: { receipt, state: "stale" as const, reasons: [{
      kind: "code" as const, basis: "computation" as const, differences: [],
      changes: [
        { kind: "driver" as const, previous: { name: "train", revision: 1 }, present: { name: "train", revision: 2 } },
        { kind: "steps" as const, previous: 2, present: 6 },
      ],
    }] } }],
  };
  render(<StudioStepInspector plan={plan} step={{ ...train, runs: [run] }} onChooseStep={() => {}}
    state={{ ...selectStep(emptyPipelineSelection, "train"), selection: { kind: "rows", stepId: "train", runIds: [run.id] } }} />);
  const reasons = within(screen.getByRole("list", { name: "Reason" }));
  expect(reasons.getByText("Results differ")).toBeDefined();
  expect(reasons.getAllByRole("listitem").slice(1).map(({ textContent }) => textContent))
    .toEqual(["train revision 1 → 2", "Updates 2 → 6"]);
});

it("titles the inspector by step and kind, then by the chosen run's values, and reveals a run's problem from its row", () => {
  const reveal = vi.fn();
  const problem = {
    code: "cannot_run", path: "/trial_documents/objective/terms", detail: "Objective has no enabled terms.",
    variant: "default", step: "train", row: "b", place: { part: "objective", document: "objective" },
  } as const;
  const rows = (runIds: string[]) => ({ ...selectStep(emptyPipelineSelection, "train"), selection: { kind: "rows" as const, stepId: "train", runIds } });
  const step = { ...train, runs: train.runs.map((run) => run.id === "b" ? { ...run, problems: [problem] } : run) };
  const inspect = (shown: PlanStepView, state: PipelineSelectionState) =>
    <StudioStepInspector plan={plan} step={shown} state={state} onReveal={reveal} onChooseStep={() => {}} />;
  const { rerender } = render(inspect(step, selectStep(emptyPipelineSelection, "train")));
  const inspector = within(document.body);
  expect(inspector.getByRole("heading").textContent).toBe("Train arm");
  expect(inspector.getByText("Train")).toBeDefined();
  expect(inspector.queryByRole("button", { name: /Objective has no enabled terms/ })).toBeNull();
  rerender(inspect(step, rows(["b"])));
  expect(inspector.getByRole("heading").textContent).toBe("Train arm · Rate 0.003 · Seed 1");
  fireEvent.click(inspector.getByRole("button", { name: /^Objective · Objective has no enabled terms/ }));
  expect(reveal).toHaveBeenCalledWith(step.runs[1], expect.objectContaining({ place: problem.place, problem }));
  const change = { path: "payload.seed", place: { part: "step", field: "seed" }, before: 0, after: 1 } as const;
  const stale = { ...train, runs: train.runs.map((run) => run.id === "b" ? { ...run, status: "stale" as const, change } : run) };
  rerender(inspect(stale, rows(["b"])));
  fireEvent.click(inspector.getByRole("button", { name: /^Seed · 0 → 1$/ }));
  expect(reveal).toHaveBeenLastCalledWith(stale.runs[1], { place: change.place, detail: "0 → 1" });
  rerender(inspect({ ...evaluate, runs: [{ ...evaluate.runs[0], label: "default" }] },
    { ...selectStep(emptyPipelineSelection, "eval"), selection: { kind: "rows", stepId: "eval", runIds: ["a"] } }));
  expect(screen.getByRole("heading").textContent).toBe("Test reach");
});
