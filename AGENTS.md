# Feedbax

## Studio projects

Projects saved by Studio before the rlrmp3 conversion are experiments. When a stored format changes, do not preserve, migrate or convert them, and write no conversion scripts for them. Tests and walkthroughs seed their own fixtures.

## Documents

Every file Feedbax writes states each fact once, in the form its author gave it. Any change to how a document is written or read keeps these properties, and the document harness checks them for every registered kind:

- A rule (a range, a family, a product, a draw) is stored as written and expanded only in Plan and compile.
- A value equal to its default is not written.
- A based document stores only the members that differ from its base.
- An expression is stored as text.

Compactness does not move work into a rollout, a training step or a keystroke. A new document kind registers a fixture with the harness.
