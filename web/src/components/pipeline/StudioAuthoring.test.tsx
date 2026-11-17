// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioDefinitionForm, type StudioDefinitionField } from "./StudioDefinitionForm";

const field: StudioDefinitionField = {
  id: "rate-field", label: "Learning rate", control: { kind: "number", min: 0, max: 1 },
  resolved: [{ id: "current", value: 0.001, source: "Training settings" }], actions: ["set", "toSweep"],
};
const selection = { kind: "steps", stepIds: ["train"] } as const;

afterEach(cleanup);

it("keeps the current typed value when turning the field into a sweep, then edits and keeps values by ID", () => {
  const edit = vi.fn();
  const choose = vi.fn();
  const view = render(<StudioDefinitionForm fields={[field]} selection={selection} onEdit={edit} />);
  fireEvent.click(screen.getByRole("button", { name: "To Sweep Learning rate" }));
  expect(edit).toHaveBeenLastCalledWith({ fieldId: "rate-field", selection, edit: { kind: "toSweep", value: 0.001 } });
  const swept: StudioDefinitionField = {
    ...field,
    actions: ["setSweepValue", "addValue", "toField"],
    sweep: { kind: "list", id: "learning-rate", label: "Rate", values: [
      { id: "stable-a", value: 0.001, source: "Base rate" },
      { id: "stable-b", value: 0.003, source: "Fast rate", selected: true },
    ] },
  };
  view.rerender(<StudioDefinitionForm fields={[swept]} selection={selection} onEdit={edit} onChooseSweepValue={choose} />);
  const pick = screen.getByRole("button", { name: "Select Learning rate 0.003" });
  expect(pick.getAttribute("aria-description")).toBe("Fast rate");
  expect(pick.getAttribute("aria-pressed")).toBe("true");
  fireEvent.click(pick);
  expect(choose).toHaveBeenLastCalledWith("learning-rate", "stable-b", false);
  fireEvent.click(pick, { shiftKey: true });
  expect(choose).toHaveBeenLastCalledWith("learning-rate", "stable-b", true);
  fireEvent.click(screen.getByRole("button", { name: "Edit Learning rate 0.003" }));
  expect(screen.getByText("Fast rate")).toBeDefined();
  const value = screen.getByRole("spinbutton", { name: "Edit Learning rate 0.003" });
  fireEvent.change(value, { target: { value: "0.005125" } });
  fireEvent.blur(value);
  expect(edit).toHaveBeenLastCalledWith({ fieldId: "rate-field", selection, edit: { kind: "setSweepValue", sweepId: "learning-rate", valueId: "stable-b", value: 0.005125 } });
  fireEvent.click(screen.getByRole("button", { name: "To Field Learning rate 0.003" }));
  expect(edit).toHaveBeenLastCalledWith({ fieldId: "rate-field", selection, edit: { kind: "toField", sweepId: "learning-rate", keepValueId: "stable-b" } });
});

it("displays the authored range and sends a rule conversion instead of expanding it", () => {
  const edit = vi.fn();
  render(<StudioDefinitionForm fields={[{
    ...field, actions: ["toList"], sweep: { kind: "range", id: "rate", label: "Rate", first: 0.001125, last: 0.015125, count: 20, scale: "linear" },
  }]} selection={selection} onEdit={edit} />);
  expect(screen.getByText("0.001125")).toBeDefined();
  expect(screen.getByText("0.015125")).toBeDefined();
  expect(screen.queryByRole("spinbutton")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "To List" }));
  expect(edit).toHaveBeenLastCalledWith({ fieldId: "rate-field", selection, edit: { kind: "toList", sweepId: "rate" } });
});
