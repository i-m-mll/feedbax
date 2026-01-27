# Web UI Notes (2026-01-27)

## Implemented (ready to test)
- Curved vs elbow edges: added a toggle in the canvas (top-right). Curved = bezier, Elbow = right-angle.
- Collapsed nodes keep edges visible: handles remain (hidden) so edges still render.
- Disallow multiple outputs into the same input: connection rejected if the target input already has a wire.
- Folder icon now opens the “open project” menu (previously not wired).

## Answers to your questions

### 1) CompositeLoss tree + eager models + where-functions
- The “Composite loss tree” string is currently a placeholder.
- We should show the loss term tree and make it editable (add/remove terms, weights, params).
- For eager models / where-functions, two viable approaches:
  - Replace where-functions with explicit selectors (e.g., node.port or StatePath strings).
  - Prefer explicit probe/metric nodes in the graph, and bind loss terms to those outputs.
- Recommendation: probe/metric nodes for UI clarity (but either works).

### 2) Training in right sidebar vs bottom shelf
- Training/analysis/validation likely belong in a separate “lower shelf.”
- Proposed layout:
  - Top shelf: graph canvas + component/properties sidebars
  - Bottom shelf: tabs for Validation / Training / Analysis
  - Both can be open, with the bottom as a draggable drawer.

### 3) Labels like “mechanics”, “network”, “feedback”
- Those are node IDs/names (auto-generated), not categories.
- We can add inline renaming if desired.

### 4) Task component / component catalog coverage
- Tasks are not implemented yet in the UI.
- Current registry only includes a minimal subset (SimpleStagedNetwork, Mechanics, FeedbackChannel, Constant, Gain).
- Task/Intervention/Signal catalogs are not wired yet.

### 5) Collapsed node hides ports -> edges disappear
- Fixed: handles remain present, so edges stay visible.

### 6) Multiple outputs to one input + type checking
- “One input can only accept one wire” is now enforced.
- Type checking: I could not find public Collimator port-typing docs. Similar systems (e.g., Simulink) use port constraints for compile-time compatibility.
- Suggested approach:
  - Define per-port dtype (optional shape) in ComponentDefinition.
  - Enforce compatibility in isValidConnection.
  - Optionally color-code port types.

### 7) Open button / save flow
- Folder icon now opens the project list.
- Save is manual (Save icon). Data stored in a single backend directory (`~/.feedbax/web/graphs`).
- No path picker yet; can add “Save As” later if needed.

## Decisions needed to proceed
1) Two-shelf layout now? (Yes/No)
2) Loss tree editor: bind loss terms to graph probes or to explicit state/port selectors?
3) Type system: simple (float/vector/state/key) or explicit with shapes?

---

# Update (2026-01-27)

## GitHub issues created
- Probe/metric nodes as loss targets: https://github.com/i-m-mll/feedbax/issues/44
- Port constraints / type checking: https://github.com/i-m-mll/feedbax/issues/45

(Each issue includes a short “related opportunities” section pointing to similar changes elsewhere in the repo.)

## UI fixes shipped in this pass
- **Inline node renaming** in the node header (click the name to edit). This also ensures there is always spacing between the name and the type label.
- **Collapsed nodes keep edges visible** (handles remain alive but hidden).
- **Edge style toggle**: Curved (bezier) vs Elbow (right‑angle).
- **One input = one wire** enforced at connect time.
- **Folder icon** now opens the project list.

## Two‑shelf layout
I understand the requirement: either shelf can be collapsed independently (i.e., a NAND on “is collapsed”). I have not yet implemented this layout shift — will do once you confirm where you want the shelf toggle affordances and default heights.

## Loss binding choice — clearer distinction
Here are the two options and their tradeoffs:

### A) **Probe/metric nodes** (recommended)
- **What it is**: Add explicit nodes in the graph whose outputs are used as loss targets.
- **Pros**:
  - Visual, inspectable, and consistent with the graph‑first UI.
  - Loss targets can be reused for logging, analysis, and validation.
  - Easier to debug because the target is a real node output.
- **Cons**:
  - Requires adding a new component type (probe/metric) and wiring it into the graph.
  - Slightly more steps to configure (create probe then bind loss).

### B) **State/port selectors ("where" style)**
- **What it is**: Loss terms refer directly to a state path or port selector string.
- **Pros**:
  - Faster to specify in text form; no extra nodes.
  - Closer to current code patterns if we keep where‑functions.
- **Cons**:
  - Less discoverable in the UI; harder to validate visually.
  - Tends to be brittle if state paths change.

**Recommendation:** A (probe nodes). It scales better for UI + analysis workflows.

## Port type system — clearer distinction

### Simple type system (lightweight)
- **Types**: a few coarse labels (e.g., `float`, `vector`, `state`, `key`, `control`).
- **Pros**: fast to implement, good early guardrails.
- **Cons**: low specificity; won’t catch shape/size mismatches.

### Rich type system (dtype + shape/rank)
- **Types**: dtype plus optional rank/shape constraints (e.g., `float[2]`, `state[6]`).
- **Pros**: prevents subtle wiring errors; better for compile‑time validation.
- **Cons**: more upfront schema work per component and more UI complexity.

**What constrains us:**
- We’ll eventually need shape‑level validation for training/simulation correctness.
- A lightweight system is fine to start, but we should design the schema so it can expand to shape constraints later.

If you want, I can implement a minimal schema now that supports both and only uses the richer info when present.
