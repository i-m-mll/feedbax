# Stored forms of Feedbax documents

This document specifies how Feedbax stores the files of a project: the one writer that produces every file, keyed member collections, expressions as text, sequence rules, member families and validation-trial rules. It is the precise form of the design in [issue:5c8f146] and is delivered by [issue:fc1b84c]. It replaces the stored-tree decision of [issue:342d3d3] and amends one sentence of [issue:b584e17] (section 7.3).

Nothing here is implemented. Sizes and timings quoted from the plan are measurements of the present forms; every statement about the new forms is a specification.

## 1. Three forms

| Form | What it is | Who reads it |
|---|---|---|
| **Stored** | The file. What the author stated, each fact once: rules unexpanded, expressions as text, members keyed, defaults absent. | Authors, reviewers, `diff`, differences, step paths, row patches, Studio's editing operations. |
| **Lowered** | The typed documents the compiler receives today: member lists with ids and expression trees, with every family expanded and every reference resolved to an id. | Request identity, compile, executors, records. |
| **Editor** | Expression text spelled with labels instead of keys, shown and typed in Studio. | Studio users. |

`lower(schema, stored payload, families) -> lowered document | Problems` relates the first two. It is pure, takes no scope from other documents (references are resolved from their spelling alone, section 3.3; whether a referent exists is a compile Problem, as today), and its cost is linear in its output. `families` is `all` or `first`: with `first`, each family is lowered for its first combination only.

**Every author-written path addresses the stored form.** Composition, differences, step deltas, sweep row patches and Evaluate's `variation` edits act on stored payloads in Plan, before `lower`. A request carries lowered documents and no path into a document.

**Where `lower` runs.**

| Caller | `families` |
|---|---|
| Plan, per resolved row: `_encode_row` (Train, Synthesize), `encode_document_input` (document inputs), `evaluation_slots` (the Task an Evaluate varies), all in `feedbax/planner/project_plan.py` and `feedbax/contracts/model_input.py` | `all` |
| `feedbax check` | `all` |
| The Set reader (`read_set`), which expands entry families before the sweep's Set axes are built | `all` |
| Studio views and the validation of a committed edit (`feedbax/web/services/trial_authoring.py`) | `first` |

**The performance boundary.** A family is expanded only by `lower(..., all)`. A sequence rule is never expanded into nodes (section 4). Nothing is expanded in a rollout step, a training step, a view or an edit: a view and an edit cost what the author wrote, and Plan costs what the rules generate. `lower` and `parse` are memoized in bounded least-recently-used caches keyed on the stored bytes as the writer gives them (so member order is part of the key); lane P3 sets their sizes. A document whose lowered form would exceed the member budget (`docs/design/authoring_budgets.md`; lane P4 sets it) is a Problem on the rule that exceeds it.

**Identity.** Request identity is taken over the lowered form. Two spellings of one expression have one identity. A family and the members it generates, written out by hand with the substituted expressions, have one identity, except that an Objective family's terms carry `family` (5.4). A sequence rule and its enumeration are different trees and different requests. Which labels enter identity is the ruling of [issue:ffd1429] ("a name never enters scientific identity"); this specification adds no rule to it.

## 2. The canonical writer

One function writes every file of a project: documents, steps, variants, the project declaration and `presentation.json`. For every file it wrote, `write(read(file)) == file`.

### 2.1 Numbers

A stored number is its value. `1` and `1.0` are the same number, in a JSON value and in expression text.

- A number is written in the form of RFC 8785 (the ECMAScript number-to-string form that Feedbax's identity encoding already uses): the shortest decimal that reads back to the same double, a whole value as an integer (`1`, `0.1`, `1e-7`, `1e+21`). `-0` is written `0`. An integer beyond ±(2^53 − 1) is refused.
- A typed field does not change this. A field typed real that holds 1 is written `1`, and the editor prints `1`. A literal's base type is taken from its site ([issue:342d3d3]), so nothing reads the difference between `1` and `1.0`. A literal typed `1.0` is printed `1`.
- Equality of stored values compares numbers by value. A boolean is not a number.
- **The writer never changes a number.** `0.00030000000000000003` and `0.15000000596046448` are different values from `0.0003` and `0.15`, and rounding them would change a document's meaning.
- Such numbers are prevented where they arise. A stored number is the decimal its author typed, read as a double. Feedbax writes into a document no number it computed from other stored values (a schedule's `init_value` that is a tenth of its `peak_value` is a default or a rule, never a stored value) and no number read back from an array or a sample (the goals in `reach-check` are a rule's output, section 6).
- `feedbax check` advises on a stored number that equals a float32 value widened to a double and has a shorter float32 decimal, and on one with 16 or more significant digits. It is advice (lane P6); nothing rewrites the number.

### 2.2 Defaults

- The writer validates the payload with its typed stored model and writes no field whose value equals the field's default, compared as stored values (2.1). A required field is always written.
- A default is a constant of the schema. No default depends on another field. A generated manifest of every stored model's defaults is checked in tests, as `request_identity_contract.json` is for identity, so a default cannot change without a schema version.
- A discriminator (`kind`, `op`) and a schema id have no default.
- **Refinement models are exempt.** In a model whose absent field means "inherit from the parent" (the sweep's `Sweep`, `NamedAxis`, `AxisValue`, `Row` and `SweepException`, which `inherit_sweep` reads through `model_fields_set`), a field the author set is written even when it equals the default, because writing `"label": null` clears the parent's label.
- A region with no typed model (`JsonValue`: a leaf component's `fields`, a Train step's `settings`, a Figure payload) is written as authored. Typing those regions is lane P5.
- A based document's differences are written as authored. The writer does not recompute them against the base: a difference that restates the base's present value is the author's statement, and it survives a later change of the base. `diff` runs on payloads in canonical form, so a content value equal to its default is absent and appears as a remove (3.4).

### 2.3 Text

- UTF-8, no byte-order mark, `\n` line ends, one final newline. Strings are escaped as RFC 8785 escapes them: `"`, `\` and U+0000 to U+001F only, with the short forms where JSON has them.
- A typed object's keys are written in the model's field order; a map's keys in member order.
- A typed expression field is written in its canonical form (4.1, 4.5).

### 2.4 Layout

Width is counted in code points. The limit is 100 columns.

1. The top-level object of a file is written one key per line.
2. Any other object or array is written on one line, as `{"a": 1, "b": [1, 2]}` with `, ` and `: ` and no padding inside the brackets, when the line it is on, counting its indent, its key and its trailing comma, ends at or before column 100.
3. Otherwise an array whose members are all scalars is filled: the opening bracket ends its line, as many members as fit follow on each line at two more spaces of indent, and the closing bracket has its own line.
4. Otherwise the opening bracket ends its line, each member has its own lines at two more spaces of indent, laid out by rules 2 to 4, and the closing bracket has its own line.
5. A string is never broken.

A 6 by 6 matrix is eight lines. A term with a long expression has that expression on one line longer than 100 columns.

### 2.5 Envelope

- A document's `schema` is its versioned schema id. An id that has no version today takes `.v2` (`feedbax.spec.task_document.v2`); `feedbax.set_document.v1` becomes `feedbax.set_document.v2`. A payload carries no second schema identifier.
- A kind's version changes once, in the merge where its stored form first changes. A file of another version is refused with a message naming both. Projects are not migrated (Feedbax `AGENTS.md`).
- A schema id with no registered stored model is refused.

### 2.6 Phasing

| Merge | Lane | What activates |
|---|---|---|
| 1 | P1 ([issue:55a43a4]) | On the existing schemas and schema ids: the one writer; numbers (2.1); default omission with the refinement exemption and the defaults manifest (2.2); text (2.3) except canonical expression text; layout (2.4); the harness with properties 1 and 2. A project written before merge 1 still reads. |
| 2 | P2 ([issue:177c355]) | Versioned schema ids (2.5); keyed collections; the three differences, including remove of an absent key (3.4). |
| 2 | P3 ([issue:523a56a]) | Expression fields as text, canonical expression text, Model address text (section 4). |
| 2 | P4 ([issue:c9a11f3]) | Sequence rules, families, validation-trial members (sections 5 to 7). |

Nothing in merge 1 changes what a file means or which files read. In merge 1 an expression field is written as the tree it is today, laid out by 2.4, and a remove of an absent key is the error it is today.

## 3. Keyed member collections

### 3.1 Rule

A collection whose members have identity is stored as a JSON object from member key to member, in member order. A collection whose members have no identity is a JSON array and is one value. No stored collection is an array of objects that carry `id`.

- The member's key is its object key. The member stores no `id`.
- The member's presentation name is the field `label`, for every kind; the fields called `name` today are renamed. Lane P2 performs the rename.
- A member key matches `^[A-Za-z_][A-Za-z0-9_-]*$`. It contains no `.`, so a dotted step path names a member unambiguously, and it does not start with a digit, so no JavaScript engine reorders it. Model element ids keep `ModelElementId` with the same two restrictions.
- A member key that an expression can name is not a reserved word (4.3).
- Object order is member order, and it is significant: the lowered list has the same order, and order enters identity where it does today.

### 3.2 Collections

| Document | Keyed collection | Label |
|---|---|---|
| Task | `variables`, `epochs`, `signals` | required |
| Task | `repeats`, each signal's `cells`, `validation_trials` (section 7) | none |
| Task | `links`: Model address text → variable key | none |
| Objective | `terms` | optional |
| Model | `ports.inputs`, `ports.outputs`, `ports.physical`, `components` | required |
| Model | `wires`, `connections`; each wire's `attachments`; each port's attachments in `attachments.sources` and `attachments.sinks` | none, except a probe's |
| Set | `entries` | as [issue:ffd1429] rules |
| Set, Step | an entry's, sweep value's or row's `task_variables` (required) and `task_links` (none) | |

- Keys of a Task's variables and signals are disjoint, since an expression names both by key.
- A Task signal feeds the Model input whose key equals the signal's key; exceptions are stored on the step, as today.
- A Task or Objective payload keeps its `id`. It is not the document's key: it is the root of the address every draw key is derived from, and a based document inherits it so that it shares its base's random streams.
- `links` and `task_links` keys contain `.`, so a link is addressed by JSON pointer in a document's differences, and no dotted step path passes into `links`. A Set entry changes links through `task_links`.
- Arrays that stay arrays, each one value: `Window.spans`, vectors and matrices, `Step.deltas`, `Step.requires`. Figure layers and panels, Report blocks and the remaining kinds are decided by the audit in lane P5 with the rule of 3.1.

**Arrays as values.** An array is a document value when an author stated it and it has at most the authoring budget of scalars (lane P5 sets it; the `LinearStateSpace` matrices of `delayed-point-mass` are within it). An array that a computation produced, such as a synthesized gain schedule, is never a document value: the document holds the rule or step that produces it, or a content reference to it.

### 3.3 Addressing

A JSON pointer in a document's differences and a dotted path in a step delta, row patch or `variation` edit name a member by its key at each level: `/variables/goal/value` and `trial_documents.task.variables.goal.value`. The by-id resolution of list members in `feedbax/contracts/matrix_core.py` has no remaining case and is deleted. A path through a real array is refused, as today.

A dotted path may continue into an expression field. The remaining segments are a node path into the field's parsed tree (`args.<n>`, `kwargs.<name>`), optionally ending in `value`, which names a literal's number and keeps its unit. The patch is applied by parsing the field, replacing the node and printing. `…variables.target.value.args.1.value` therefore still sweeps one number inside an expression, and `…epochs.prep.end.length.value` sweeps the number of `"10 st"`. A generated or listed axis writes numbers; the unit is the field's.

Setting a key that the collection lacks appends the member. Removing a key removes the member.

### 3.4 Differences

`Differences` maps a pointer to one of three differences:

| Difference | Stored | Meaning |
|---|---|---|
| Set | `{"value": V}` | The value at the pointer is `V`. A new key is appended to its collection. |
| Remove | `{"op": "remove"}` | The key at the pointer is absent. It is not an error if the key is already absent and its parent exists: an absent key reads as its default. A missing parent is `DifferenceNotApplied`. Both appliers (`apply_differences` and the sweep patch path) follow this. |
| Order | `{"order": [key, …]}` | The keyed collection at the pointer has its members in this order. |

- Differences apply in stored order, and order differences after all others, outermost first.
- An order difference lists keys. Listed keys the collection has come first, in the listed order; a listed key it lacks is ignored; members not listed follow in their existing relative order. A member the base gains later is therefore kept, and one the base loses is not an error.
- No set or remove difference lies under another. Any difference may lie under a pointer that has an order difference.
- An order difference is legal only at a keyed collection of the typed stored model. Key order elsewhere is not significant: a typed object is written in field order, and an untyped region is compared without order. A Set entry's differences hold no order difference, since a sweep patch has no form for one.
- `diff(base, content)` emits member-local set and remove differences in the order of the content's members, then, for each keyed collection whose key order in the content differs from the order those differences produce, one order difference listing every key of the content's collection. `apply(base, diff(base, content)) == content`, order included.
- Changing one member of a collection of N yields differences whose size does not depend on N. A based document that changes one member keeps receiving the base's later changes to every other member, which removes the hazard recorded in 7e2c1b0.

Precedent: Kubernetes strategic merge patch merges a list by a merge key and carries order in a separate `$setElementOrder` directive ([design proposal](https://github.com/kubernetes/design-proposals-archive/blob/main/cli/preserve-order-in-strategic-merge-patch.md)); RFC 7396, which replaces arrays whole, is the rule this section retires.

## 4. Expressions as text

### 4.1 The field

An expression field holds one of:

- a JSON number or boolean: the literal with no unit;
- a JSON string: expression text in the grammar of 4.2.

The canonical form of a unitless numeric or boolean literal is the JSON scalar; every other expression is text. `"weight": 2`, `"value": "choice(0.1, 0.2)"`, `"length": "10 st"`.

Every field that holds an expression, a time, or a number with a text alternative is an expression field; its site says what it accepts. They are: `Variable.value` and `count`; `Length.length`; `Until.condition` and `limit`; `Repeat.count`; `Cell.value`; `Signal.absent`; `ObjectiveTerm.measure` and `weight`; every Window offset; `TrialSetField.condition`; `ConditionalProduct.conditions`; validation-trial values (section 7); `ModelDocument.step_seconds`; `CompositeComponent.substeps`; `DelayAttachment.delay`; `DelayInitialValue.value`; `StepReference.checkpoint`; and a numeric leaf field (which leaf fields are numeric is the component type's declaration, as today). A time is literal text (`"30 st"`, `"50 ms"`); the object `{"value": 30, "unit": "st"}` is not a stored form. The lowered models are unchanged.

`UnresolvedValue` is not a stored form and not a node.

- Text that does not parse is stored as typed and is a Problem on its field. It is editor text, so a later rename can leave it stale; it is already a Problem.
- Text that parses is stored in canonical form with every name the site knows spelled by key. A name the site does not know is kept as typed and is a Problem on that name. It is read as a key from then on: it resolves when a member with that key exists, and the editor shows it marked missing until then.

The tree stays the in-memory form for checking, menus and compiling.

### 4.2 Grammar

The grammar is the field grammar of `feedbax/contracts/expression_text.py` with these changes.

1. **Single-quoted names.** A name that is not a `NAME` token, or is a keyword, is written `'…'` with `\'` and `\\` escapes. The printer writes this form; the reader also accepts the double-quoted form. A quoted name in stored text needs no JSON escape unless it holds `"` or `\`.
2. **Sequence rules** (section 5): `first..last`, and `linspace` and `geomspace` in sequence positions. The `UNIT` token's lookahead admits `..` after a unit, so `10 st..30 st` lexes.
3. **Weighted choice:** `choice(a, b, c, weights=[3, 1, 1])`. `weights` is legal only on `choice`; it is a vector of non-negative unitless literals with a positive sum, one per member, counting each sequence rule's members. This gives `DrawValue.weights` a text form and closes [issue:2a7ae3e].
4. **Nodes.** A stored tree's nodes are a literal, a name, a call, a draw and a sequence rule. The field-root forms `a, b, c`, `linspace(…)` and `sample(…)` remain what they are today: an editor gesture that writes a sweep axis on the step. A document never stores one.
5. `lag(x, 0)` is `x`.

### 4.3 References

A name in stored text is spelled from keys only. A rename of a label rewrites no stored text.

| Referent | Stored spelling |
|---|---|
| Task variable, Task signal, a Set entry's variable inside its entry | its key |
| Reserved names | `time`, `t`, `T`, `u`, `i` |
| A family index (section 6) | its name |
| A Model port, a field of a port's value, a component's state | the Model address text of 4.4 |
| An epoch's end, in Objective scope | `epoch.<epoch key>.condition`, `epoch.<epoch key>.limit` |

A spelling that is a `NAME` token is written bare; any other is single-quoted. The reserved words are the reserved names, the keywords `true false and or not`, the head `epoch`, and the head words of 4.4 (`graph inputs outputs boundary in out physical component wire connection`). No member key that an expression can name is a reserved word, so every stored spelling has one referent and `lower` maps it to today's reference (a member id, an `epoch:<id>:…` id, or `ModelAddress.to_text()`) without a scope. How a variable of a Set placed in a step's trials is spelled from outside its entry is decided in lane P3, once it has found where such references are stored today.

**The editor form** prints the same tree with labels and parses against the site's labels, as today. A label is written bare when it is a `NAME` token and quoted otherwise, so a label may hold spaces. A label that expressions can name is unique in its scope and is not a reserved word; the Service refuses a rename that breaks either ([issue:ffd1429]).

### 4.4 Model address text

`ModelAddress.label()` is not injective (a boundary output `x` and the port `x` of a component named `outputs` both print `outputs.x`) and has no inverse. A Model address in stored text, and wherever a stored document holds a Model address as a string (`links` keys, `Signal.intervention`), uses this form, which is injective and has an inverse:

```
address   := (ID "/")* element
element   := "graph" | "inputs" | "outputs"
           | "inputs." ID | "outputs." ID | "boundary." ID                boundary port of the scope
           | "in." ID "." ID | "out." ID "." ID | "physical." ID "." ID   component, then port
           | "component." ID | "wire." ID | "connection." ID
           | port "#" ID | wire "#" ID                                     attachment
           | element ("." SEG)+                                            field; the parent is not "inputs" or "outputs" alone
ID        := ModelElementId
SEG       := [A-Za-z_][A-Za-z0-9_-]* | index | JSON string
```

- Each `ID "/"` prefix enters one composite, outermost first. The text is tokenized before it is split, so a `/` or `.` inside a JSON-string segment is part of the segment.
- The number of segments an element takes is fixed by its head word, so what follows it is a field path.
- An `index` is a decimal integer with no leading zero. A field segment that is a string of another shape, or a string of digits, is a JSON string in RFC 8785 form, so an index and a digit string differ. Text that is not in this canonical form is refused, as `ModelAddress.from_text` refuses non-canonical JSON today.
- `outputs.effector.pos` is the field `pos` of boundary output `effector`; `out.plant.force` is output port `force` of component `plant`; `component.mech.state.pos` is a component's state.
- An address is a `NAME` token when it has no composite, no attachment and no JSON-string or index segment, and each id and segment is identifier-shaped with no hyphen. `out.controller-input.x` is written `'out.controller-input.x'`.

Sweep targets and `presentation.json` hold Model addresses as canonical-JSON strings today. Lane P5 moves them to this form; a sweep target keeps its braces around the address (`documents.m.{component.plant.fields.mass}`), which is what tells an address from a dotted patch path.

### 4.5 Canonical text

Stored text that parses is the canonical print of its tree. `print(parse(s)) == s` for every canonical `s`, and `parse(print(tree)) == tree` for every tree in normal form.

- Binary operators are printed with one space on each side; `..` with none; parentheses only where precedence needs them; arguments separated by `, `; keyword arguments after positional ones, sorted by name.
- A number prints as 2.1 gives it. A unit follows its number after one space, except `°`, which follows directly.
- Tree normal form: a number that is whole is an integer; a unit is a unit of the table or absent, never `""`; `neg` and `pos` applied to a number literal are that signed literal, with its unit.
- Stored text is at most 8,192 characters, the limit `FieldTextEdit` has today.

## 5. Sequence rules

A sequence rule denotes a finite, ordered sequence of literals.

| Rule | Members | Bounds |
|---|---|---|
| `first..last` | the integers from `first` to `last`, step 1 | both included; `first <= last`; integers, with the same unit or none |
| `linspace(first, last, count)` | `count` values, evenly spaced | both included |
| `geomspace(first, last, count)` | `count` values, evenly spaced in the logarithm | both included; both positive |

- Every bound is included. This matches every range Feedbax already has: the sweep `RangeAxis`, a run rule's `name=first..last`, and the integer `uniform` on an epoch length. `range` is not a function of the language: `range(6)` is an unknown-function Problem whose reason names `0..5`.
- `first..last` is a production of a sequence position only, with each side a sum: `0..n - 1` is `0..(n - 1)`. It is not an operator, so `a == 0..5`, `(0..5)` and `0..1..2` do not parse.
- `first` and `last` of `linspace` and `geomspace` carry the same unit or none, which every member carries. `count` is a positive integer. With `count` 1 the sequence is `first`.
- Members of `linspace` and `geomspace` are computed as the sweep's real range computes them (`_range_values`), in normal form, so a whole member is an integer. A member in `st` that is not whole is the Problem it is today.
- The arguments are structural (6.3): integer constant expressions for `..` and `count`, literals for the bounds of `linspace` and `geomspace`.

**Sequence positions.** A sequence rule is legal in exactly these places:

1. An argument of `choice`. Its members are members of the choice, in place: `choice(linspace(0°, 315°, 8))` has eight members, and `choice(0..2, 10)` has four.
2. An element of a vector constructor: `[linspace(0, 1, 5)]` is a vector of five.
3. The item of an index: `x[0..1]` is the vector of `x[0]` and `x[1]`. This is the vector slice of [issue:493a1f5], built with lane P4.
4. An index rule of a family (6.1).

Anywhere else a sequence rule is a `structure` Problem. A rule is not broadcast through a function: `polar(0.15 m, choice(linspace(0°, 315°, 8)))` is legal, and `choice(polar(0.15 m, linspace(0°, 315°, 8)))` is not.

**The node is kept.** `SequenceValue(rule, first, last, count)` is a node of the stored tree and of the lowered tree. In positions 1 to 3 it evaluates to one run-level array of its members, and `choice`, the vector constructor and `index` take that array as a run of members. The checker counts a rule's members statically. A 56-point `choice` written as five rules is five nodes to check and compile, not 169. The field-root `SweepRange` becomes this node.

Precedent: Hydra stores `range` and `choice` in configuration and expands at launch ([override grammar](https://hydra.cc/docs/advanced/override_grammar/extended/)); its `range` is half-open, which this specification does not follow, for the consistency reason above.

## 6. Member families

### 6.1 Stored form

A member of an Objective's `terms`, a Set's `entries` or a Task's `validation_trials` may carry `over`, which makes it a family: one stored member that stands for one member per combination of its indices. No other collection takes `over`.

```json
"movement-state": {
  "label": "Movement state",
  "over": {"k": "0..5"},
  "measure": "1000000 * sqnorm(lag(outputs.effector.pos, k + 1) - outputs.aim / 1 m) + 100000 * sqnorm(lag(outputs.effector.vel, k + 1))",
  "weight": "((t + (11 - k) * 1 st) / 60 st) ** 6 / 6",
  "reduction": "sum"
}
```

- `over` is an ordered object from index name to index rule. An index name is a `NAME` with no `.`, not a reserved word, and not a key the member itself defines. It shadows a same-named key of the surrounding scope.
- An index rule is one of:
  - text holding one sequence rule: `"0..5"`, `"linspace(0°, 315°, 8)"`;
  - a named-value object from value key to value: `{"x": 0, "y": 1}`. A value is an expression field holding a constant (a number, a boolean, or text such as `"90°"` or `"[1, 0]"`), or `{"value": constant, "label": text}`. A value key matches `^[A-Za-z_][A-Za-z0-9_]*$`.
- An index rule names no index.
- `combination` says how the indices combine: `"cross"` (the default, not written) or `"zip"`, with the meaning and the order the sweep contract gives them (`feedbax/contracts/project_sweep.py`): in a cross the first index varies slowest, and a zip of unequal lengths is a Problem. The same function that expands a sweep's combination produces a family's combinations. Groups and manual combinations are not available to a family.

### 6.2 Where an index is in scope

An index is a name in every expression field of the family member, at any depth, and a placeholder `{name}` in its `label`. In an expression it is a run-level constant of its values' type.

In a Set entry family the indices are in scope in the entry's `task_variables` and its `label`. They are not in scope in `differences`: a difference's value is JSON written into a Model that is not known until a step binds the Set, so text in it cannot be told from a string. A Model value that varies over the family is a Task variable of the entry, read by a Link or by an operand (`{"kind": "task", "variable": …}`). A family therefore covers entries that differ in values; entries that differ in which element of the Model they change are separate members.

### 6.3 Expansion

One function, `expand_family`, maps a parsed family member to its generated members. Plan, `feedbax check`, the Set reader, the quadratic extraction of [issue:94021b6] and Studio's views (through `lower(..., first)`) all call it. For each combination, in order:

1. **Substitute.** In each expression of the member, each index name node is replaced by its value's node. Nothing else in the tree changes, so a node of a generated member has the path it has in the stored member, and a Problem on it carries the stored field, that path and the combination.
2. **Key.** The generated member's key is the family key followed by `-` and each index's key part, in `over` order:

   | Index rule | Key part |
   |---|---|
   | `first..last` | the integer, a negative one written `m` then its magnitude (`m3`) |
   | named values | the value key |
   | `linspace`, `geomspace` | the zero-based position |

   A generated member's coordinate is its value or its name wherever that is a key, so extending `0..5` to `0..7` keeps the first six keys. A `linspace` member is a function of its position and the rule, so position is its coordinate.
3. **Label.** Each `{name}` in the label is replaced by the named value's label if it has one, else its value key, else the canonical print of the value. `{{` and `}}` are braces. A placeholder that names no index is a Problem. A label of a member that is not a family is literal text.

Generated keys are unique among the collection's members and generated members. Generated labels of a Set family are distinct. A collision is a Problem on the family.

**Structural arguments.** An argument that must be static is an integer constant expression: integer literals and family indices combined with `+`, `-`, `*`, `%` and sign. It is evaluated when the tree is checked and is not rewritten in the tree; `%` by zero is a Problem. The structural arguments are: the step count of `lag`; `n` of `diff` and `ddt`; the item of `index` into a list variable; `Variable.count` and `Repeat.count`; the bounds of `first..last` and the `count` of `linspace` and `geomspace`; a validation trial's `epoch_steps`. `lag(x, k + 1)` with `k` an index is legal, in a family and (with literals only) outside one. `lag(x, n)` with `n` a Task variable, a draw or a model output is the `structure` Problem it is today, with the reason "Lag steps must be a non-negative whole number, or an expression of family indices that gives one": the lag's state has a static shape, so its count cannot be a value known only in a trial. `lag(x, k / 2)` is refused, since `/` is not an integer operation.

No other subtree is evaluated at expansion. `((60 - k) / 60) ** 6` stays a run-level subtree, which the compiler evaluates once per run.

Precedent: Modelica evaluates structural ("evaluable") parameters at translation and keeps the source hierarchical ([specification](https://specification.modelica.org/master/arrays.html)). Terraform's `for_each` identifies instances by key and its documentation calls `count`'s positional addresses fragile ([for_each](https://developer.hashicorp.com/terraform/language/meta-arguments/for_each)), which is the reason for key parts by value or name. No library was found that supplies index families for a document model of this kind; the mechanism is Feedbax's sweep contract applied to members.

### 6.4 Objective families

An Objective family is one term: its value is the sum of its generated members' values, and it is reported, plotted and named as one term under the family's key and label. Its label takes no placeholders. The lowered Objective holds the generated terms, each with `family` set to the family's key; a per-term report sums the terms that share a `family`. `family` is part of identity, since it changes what is recorded.

### 6.5 References to generated members

- A step's `set_fields` weights name entries by key. A family's key there gives its weight to each generated entry; a generated key gives one entry its own.
- A selection of validation trials by id may name a family's key, which selects every trial it generates.
- Row keys and results name generated keys.

### 6.6 Families, based documents and Studio

A difference or a path addresses the stored member, so a family is addressed as one member by its key: `/terms/movement-state/weight`, `/terms/movement-state/over/k`. A based document changes a family's rule, expressions or window with a difference whose size does not depend on the family's count.

- A generated member cannot be addressed. A pointer to a generated key (`/terms/movement-state-3/weight`) names no stored member; a set difference there creates a member whose key collides with the family's, which is the collision Problem of 6.3.
- One generated member is made to differ by an expression (`where(k == 3, …, …)`) or by narrowing the rule and adding a member beside it. There is no exception list.
- A based document may add `over` to a member of its base, or remove it; the member's expressions must then check without the indices.
- Studio shows a family as one member: its fields, addressed at the stored member, with a site that includes the indices; its index rules as fields; a Problem with the combination it arose in. The Set view lists a family as one entry with its count, and a weight is edited on the family. A generated member is not edited on its own.

## 7. Validation trials

`validation_trials` is a keyed collection of trial members. An empty collection is the default and means the Task's trials are sampled.

```json
"validation_trials": {
  "reach": {
    "over": {"direction": "0..3", "repeat": "0..63"},
    "variables": {"hold_trial": false, "catch_trial": false, "goal": "polar(0.15 m, direction * 90°)"},
    "epoch_steps": {"prep": "10 + repeat % 21"}
  }
}
```

- A trial member has `variables` (variable key → value) and `epoch_steps` (epoch key → steps), both optional, and may be a family.
- A value in `variables` is an expression field checked against the variable's type. It may name the member's indices and nothing else, and holds no draw. It replaces the variable's own expression for the trial, as a row value does today.
- A value in `epoch_steps` is a structural argument (6.3) that gives a non-negative integer.
- A member with no `over` is one trial. This is the listed row, for a cohort whose rows are the information.
- A repeat is an index. A variable that a member does not give is drawn under the trial's random key, so the members of `repeat` differ in their draws. No variable exists only to tell rows apart.
- All members give the same variable keys and epoch keys, as rows must today.

### 7.1 Trial ids and keys

A trial's id is its member key, which for a family member is the generated key (`reach-2-17`). It replaces the id taken from the canonical JSON of a row's variable values, which cannot tell apart two rows that differ only in `epoch_steps`.

- A trial's random key is derived from the seed and the trial's id, for Train validation as it is for Evaluate today. A trial member's key is therefore scientific identity: changing it changes the draws of its trials.
- Where the trials are crossed with the conditions of a Set (`lower_trial_sets`), a crossed trial's id is `<trial id>--<axis_values_id of its Set coordinates>`, with Set coordinates by entry key.
- Sample ids stay the draw index.

### 7.2 Lowered form and compiler changes

- `ValidationTrial` gains `id`. `variables` holds constant expression trees, which `compile_task` evaluates against each variable's type as it evaluates row values today; `epoch_steps` holds integers.
- The empty stored collection lowers to today's default: one row with no values and no `id`. `condition_ids` is empty for exactly that row, so every test of `condition_ids` reads as it does today. A stored member with no values is one trial with an id.
- `ObjectiveTerm.family` (6.4).
- `lower_trial_sets` forms crossed ids as 7.1 says and writes crossed values as literal trees. Train validation keys its trials by id (`validation_trial_keys`).
- `choice`, the vector constructor and `index` take sequence nodes (section 5); `lag` takes 0 and an integer constant expression.

Each lowered-form change enters `request_identity_contract.json`, so every request id changes and results stored before the change are not current.

### 7.3 Amendment to b584e17

[issue:b584e17] rules that a condition's id is "its coordinates over those variables, by member id, never its position" and that "No row id is authored by hand". The first part stands and is how a family's generated keys are formed. The second is amended: a listed trial is a member of a keyed collection and has a key like every member, and a family's key is authored once for all of its trials. No id is authored per generated row. The amendment needs the owner's approval, because that ruling is the owner's.

## 8. What `lower` does, in order

1. Validate the stored payload with its typed stored model.
2. Parse every expression field, from the cache keyed on text.
3. Expand families (6.3), all combinations or the first.
4. Map every spelling to its reference (4.3).
5. Check structural arguments (6.3).
6. Emit the lowered document: each keyed collection as a list in object order, each member with its key as `id` and its `label`.

Type checking stays where it is, in the compile of the lowered trees. Because expansion changes no node path and a sequence rule stays one node, a Problem's node path is a path in the stored tree, and the Service maps it to a text span by parsing the stored text, as it does today.

## 9. Examples

`reach-check`, a Task based on `reach-and-hold-task` (4,763 lines today):

```json
{
  "schema": "feedbax.spec.task_document.v2",
  "label": "Reach check",
  "base": {"document": "reach-and-hold-task"},
  "differences": {
    "/validation_trials/reach": {
      "value": {
        "over": {"direction": "0..3", "repeat": "0..63"},
        "variables": {
          "hold_trial": false,
          "catch_trial": false,
          "goal": "polar(0.15 m, direction * 90°)"
        },
        "epoch_steps": {"prep": "10 + repeat % 21"}
      }
    }
  }
}
```

The differences of `delayed-point-mass-conditioned`, a Model that adds one input port and one wire:

```json
"differences": {
  "/ports/inputs/level": {"value": {"label": "level", "type": {"shape": [1]}}},
  "/interior/components/controller-input/fields/n_inputs": {"value": 3},
  "/interior/wires/level": {
    "value": {
      "source": {"owner": "inputs", "port": "level"},
      "target": {"owner": "component", "component": "controller-input", "port": "in_2"}
    }
  }
}
```

A Set entry family of `hold-bank` (four entries today). It generates `initial-position-x-minus`, `initial-position-x-plus`, `initial-position-y-minus` and `initial-position-y-plus`, the keys the entries have today:

```json
"initial-position": {
  "label": "Initial position {axis} {sign}",
  "over": {
    "axis": {"x": "[1, 0, 0, 0, 0, 0]", "y": "[0, 1, 0, 0, 0, 0]"},
    "sign": {"minus": {"value": -1, "label": "−"}, "plus": {"value": 1, "label": "+"}}
  },
  "task_variables": {
    "kick": {"label": "kick", "type": {"base": "real", "shape": [6]}, "value": "0.01 * sign * axis"}
  },
  "differences": {
    "/interior/wires/epsilon/attachments/check": {
      "value": {
        "kind": "intervention",
        "operation": "add",
        "operands": {"value": {"kind": "task", "variable": "kick"}},
        "window": {
          "spans": [
            {
              "kind": "interval",
              "start": {"anchor": {"kind": "trial", "edge": "start"}, "offset": "29 st"},
              "end": {"offset": "1 st"}
            }
          ]
        }
      }
    }
  }
}
```

Line counts after re-authoring remain the plan's estimates until lane P7 measures them.

## 10. Alternatives considered

| Decision | Alternative | Why not |
|---|---|---|
| A number is its value, in RFC 8785 form | Keep integer and real apart (`1` and `1.0`) | Nothing reads the difference: a literal's base type comes from its site, and identity already encodes both as `1`. Keeping it shows a user `1.0` for a typed `1`. |
| The writer never rounds | Round to 15 digits, or to float32 | It would change stored values. Float noise is prevented at its source and reported by `feedbax check`. |
| Identity over the lowered form | Identity over the stored form | A respelling, a relabelling or a move into a family would make a trained run stale. |
| Lowered documents keep lists with ids | The compiler reads keyed stored models | The compiler would expand families itself, and JCS sorts object keys, which loses member order in identity. |
| A sequence rule stays one node | Expand it into literals in `lower` | View and compile cost follow node count; expansion would restore the 169 nodes of the 56-point choice. |
| `first..last`, bounds included | Python `range(start, stop)` | Every existing Feedbax range includes its last value; a half-open form beside them invites off-by-one errors, and the name `range` would promise Python's meaning. |
| Family on the member (`over`) | A `sum` or `for` in the expression language | [issue:342d3d3] keeps the language free of binders; a family also covers Set entries and trials, which are not expressions. |
| Substitution only; integers evaluated where structure needs them | Fold every index-only subtree to a literal | Folding needs a second implementation of each function's semantics, in double precision, beside the compiled one, and changes node paths. |
| Key part by value or name | Key part by position everywhere | A positional key changes when a rule is extended at the front or a value is inserted. |
| No exception list for families | Per-combination overrides, as sweeps have | No case needs it; an expression or a narrower rule states the same fact once. |
| Index not in scope in Set `differences` | Substitute into difference values | The Model a Set entry writes into is unknown until a step binds it, so the type at a pointer is not a function of the Set's file. |
| Trial members keyed | One anonymous rule with ids derived from values | Value-derived ids cannot separate rows that differ only in epoch steps or only in their draws, which is why the `draw` variable existed. |
| Unitless literal as a JSON scalar | Every expression as a string | A sweep axis writes numbers; one canonical form per value is kept by making the scalar the canonical form of exactly those literals. |
| Every time as literal text | `TimeValue` objects | An object beside text gives a time two stored spellings in fields that take both. |
| A node path may follow an expression field in a step path | A sweep writes whole fields only | Sweeping one number inside an expression exists today and Objectives have no variables to hoist a number into. |
| Differences written as authored | Recompute differences on write | A difference equal to the base's present value would vanish and then follow the base's next change. |
| JSON pointers in differences, dotted paths in steps | One path syntax | Both name members by key, which is the property that matters; changing step paths is outside changes 1 to 6. |

## 11. Tests

The document harness of [issue:5c8f146] checks these properties over every registered kind. None compares authored strings.

1. Deleting any key from a written file changes the value read back or fails validation.
2. `write(read(file))` returns the same bytes.
3. For each rule form, the stored size at count 1,000 equals the size at count 10 up to the digits of the count, and the rule read back is the rule written.
4. Changing one member of a based document with N members yields differences whose size does not depend on N, and `apply(base, diff(base, content)) == content` over generated adds, removes and reorders together.
5. `parse(print(tree)) == tree` over generated trees in normal form that include draws, weights, keyword arguments, every unit, sequence nodes and quoted names.
6. `ModelAddress` text round-trips over generated addresses, including ids named `inputs` and `outputs`, digit-string field segments, hyphenated ids and composites.
7. A family lowers to the same lowered document as its members written by hand with the generated keys and the substituted expressions (for an Objective, with `family` set aside).
8. A sweep refinement that sets a field to its default reads back as set.

## 12. Not verified

- That a Set entry's Task variable can drive an intervention operand and a Link for a listed (not sampled) Set axis. Section 6.2 depends on it; lane P4 verifies it first.
- That Evaluate's `variation` edits can all be applied in Plan, so that no executor applies a path to a document (section 1). Lane P2 verifies it; if an executor must, by-id resolution stays for the lowered form only.
- That `polar(0.15 m, direction * 90°)` and run-level weights evaluated at compile equal the numbers the present documents store, to rlrmp3's test tolerance. A value may be written as literal numbers where they do not.
- That pydantic's default exclusion matches 2.2 for every stored model. Property 1 decides.
- Where trial records and request identity hold labels today, which [issue:ffd1429] must settle before a label rename leaves a current record with an old label (`feedbax/persistence/evaluation_trial_records.py`).
