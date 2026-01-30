# Web UI Issues & Improvements (Batch 2)

---

## Subgraph Visualization (Descent Mechanism)

**Current behavior**: No subgraph support. All components are flat/leaf nodes on a single canvas.

**Desired behavior (primary)**: Descent/matryoshka navigation for composite nodes:
- Double-click a composite node → opens new canvas layer showing its internals
- Breadcrumb navigation at top: `Model > Network > Encoder`
- Clean separation: each level is its own canvas
- Visual differentiation for composite vs leaf nodes (double-border, folder icon, or mini-preview of internal structure)

**Future enhancement (not v1)**: Inline expansion mode as an alternative view:
- Option to expand a node in-place on the same canvas
- See multiple levels simultaneously
- More complex to implement; design descent mechanism so inline expansion can be added later without major refactoring

**Implementation notes**:
- Keep canvas component reusable so descent just instantiates another canvas
- Store navigation stack (which nodes are "open") in UI state
- Ensure edge routing and layout at each level are independent

---

## SimpleStagedNetwork → Graph-Based Internal Structure

**Current behavior**: `SimpleStagedNetwork` is a monolithic Python class with hardcoded layer structure. Not editable as a subgraph in the UI.

**Desired behavior**: Refactor to use internal `Graph` structure:
- Encoder, hidden layers, readout as separate `Component`s wired together
- Becomes a true composite node, editable via descent mechanism
- Aligns with the "everything is a graph" paradigm

**Future consideration — specialized network layer editing**:
- Networks often have many layers; representing each as a full-sized box may be unwieldy
- Consider a compact "layer row" representation for network internals (smaller visual footprint than standard nodes)
- This is a UI convenience, not a structural change—the underlying graph is still Components + Wires
- Low priority; only implement if it proves necessary after basic descent editing works

**Related**: Potential future use of Penzai for network internals (separate discussion, not blocking this work).

---
