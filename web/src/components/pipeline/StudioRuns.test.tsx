// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { PlanStepView, PlanView } from "@/features/pipeline/planView";
import { emptyPipelineSelection, selectStep, type PipelineSelectionState } from "@/features/pipeline/selection";
import { StudioRuns, type StudioRunCommand } from "./StudioRuns";

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
    { id: "a", label: "A", values: { rate: "low", seed: "one" }, status: "succeeded" },
    { id: "b", label: "B", values: { rate: "high", seed: "one" }, status: "succeeded" },
    { id: "c", label: "C", values: { rate: "high", seed: "two" }, status: "stale", reason: "Learning rate 0.001 → 0.003" },
  ],
  statusCounts: { succeeded: 2, stale: 1 },
};
const evaluate: PlanStepView = {
  id: "eval", kind: "evaluate", label: "Test reach", inputs: [], sweeps: [], statusCounts: { not_run: 1 },
  runs: [{ id: "a", label: "Straight reach", values: {}, status: "not_run" }],
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

function showRuns(step: PlanStepView, commands: readonly StudioRunCommand[] = []) {
  vi.spyOn(Element.prototype, "clientHeight", "get").mockReturnValue(360);
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(44);
  function TestRuns() {
    const [state, setState] = useState<PipelineSelectionState>(() => selectStep(emptyPipelineSelection, step.id));
    return <StudioRuns plan={plan} step={step} region={result} commands={commands} state={state} onSelectionChange={setState} />;
  }
  return render(<TestRuns />);
}

it("preserves the sweep and row handles through the same Run command", () => {
  const perform = vi.fn();
  showRuns(train, [{ name: "run", enabled: true, perform }]);
  const table = screen.getByRole("table", { name: "Runs" });
  const rate = within(screen.getByRole("group", { name: "Rate" })).getByRole("button", { name: "0.003" });
  fireEvent.click(rate);
  expect(screen.getByText("Rate 0.003 · 2 of 3")).toBeDefined();
  expect(within(table).getByRole("row", { name: /B 0.003/ }).getAttribute("data-selected")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "Run" }));
  expect(perform).toHaveBeenLastCalledWith({ kind: "values", stepId: "train", terms: [{ sweepId: "rate", valueIds: ["high"] }] });

  fireEvent.click(within(table).getByRole("button", { name: "B" }));
  fireEvent.click(within(table).getByRole("button", { name: "C" }), { metaKey: true });
  expect(within(table).getByRole("button", { name: "C" }).getAttribute("aria-current")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "Run" }));
  expect(perform).toHaveBeenLastCalledWith({ kind: "rows", stepId: "train", runIds: ["b", "c"] });
  expect(screen.getByText("Learning rate 0.001 → 0.003")).toBeDefined();
  expect(within(screen.getByRole("region", { name: "Used By" })).getByRole("button", { name: "Test reach 1 of 1" })).toBeDefined();
});

it("virtualizes an authoritative 1,000-row result and keeps its full accessible count", () => {
  const runs = Array.from({ length: 1000 }, (_, index) => ({ ...train.runs[0], id: String(index), label: `Reach ${index}` }));
  showRuns({ ...train, runs, statusCounts: { succeeded: runs.length } });
  const table = screen.getByRole("table", { name: "Runs" });
  expect(table.getAttribute("aria-rowcount")).toBe("1001");
  expect(within(table).getByRole("button", { name: "Reach 0" })).toBeDefined();
  expect(within(table).getAllByRole("row").length).toBeLessThan(50);
  expect(within(table).queryByRole("button", { name: "Reach 999" })).toBeNull();
  fireEvent.scroll(table.parentElement!, { target: { scrollTop: 44 * 500 } });
  expect(within(table).getByRole("button", { name: "Reach 500" })).toBeDefined();
  expect(within(table).queryByRole("button", { name: "Reach 0" })).toBeNull();
});
