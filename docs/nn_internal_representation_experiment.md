# Experiment: Selector-Friendly NN Tree + Treescope Preview

## Goal

Pick one neural network class and make its internal structure explicitly inspectable and editable as a canonical tree. Then evaluate whether selectors/Treescope make development more elegant.

## Candidate Targets

- `SimpleStagedNetwork` (if still present)
- Any Equinox-based NN module with multiple sublayers

## Steps

1. **Add `nn_tree()`**
   - Return a canonical NN tree (simple `SequentialBlock` or `ParallelBlock` nodes).
   - Add `name`, `role`, `tags` metadata for each internal sublayer.

2. **Add minimal render support**
   - If Treescope is present, register a small renderer for the NN block types.
   - Ensure the tree renders compactly with tag/role text.

3. **Test selectors on the tree**
   - Use `pz.select` to find blocks by type or tag.
   - Perform a no-op or trivial edit (e.g., replace an activation or scale a weight).

4. **Record a short example**
   - Include 1-2 code snippets showing:
     - Select by type or tag
     - Apply an edit
     - Visualize with Treescope

## Success Criteria

- You can locate a sublayer by type or tag without manual traversal.
- A small surgical edit is possible without custom traversal code.
- The model tree display is interpretable and stable across runs.

## Potential Failure Modes

- The canonical tree drifts from actual execution.
- Selectors add marginal value over `eqx.tree_at` in real workflows.
- Treescope display is too noisy without custom renderers.

## Decision Outcomes

- **If successful**: Expand the canonical NN tree approach across more modules.
- **If marginal**: Keep Treescope as a visualization tool only, skip selectors.
- **If noisy**: Invest in better metadata/tags or drop the canonical tree idea.
