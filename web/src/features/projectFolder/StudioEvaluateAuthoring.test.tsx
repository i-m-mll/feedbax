// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { TrialsEditor } from "./StudioEvaluateAuthoring";

afterEach(cleanup);

// Catches a refused Trials edit printed as a sentence under the field instead of marked on the control it refuses.
it("marks a refused Trials edit on the kind or on the count, never as a line under the field", () => {
  const reason = "Count exceeds the sampler's limit.";
  const { rerender } = render(<TrialsEditor variables={[]} initial={{ kind: "every_condition" }} domain={{ kind: "fixed", conditions: ["a"] }}
    disabled={false} refusal={reason} onSave={() => {}} />);
  expect(screen.getByRole("img", { name: "Problem" }).getAttribute("data-app-tip")).toBe(reason);
  expect(screen.getByRole("combobox", { name: "Trials" }).getAttribute("aria-describedby")).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
  rerender(<TrialsEditor variables={[]} initial={{ kind: "sample", count: 20_000 }} domain={{ kind: "sampler" }}
    disabled={false} refusal={reason} onSave={() => {}} />);
  expect(screen.getByRole("img", { name: "Problem" }).getAttribute("data-app-tip")).toBe(reason);
  expect(screen.getByRole("textbox", { name: "Count" }).getAttribute("aria-invalid")).toBe("true");
  expect(screen.queryByRole("alert")).toBeNull();
});
