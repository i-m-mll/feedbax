"""The one experiment-envelope compiler.

Feedbax owns the dialect (:mod:`feedbax.contracts.experiment_envelope_dialect`)
and this is the compiler for it. There is one compiler, one compiler contract,
and five layers, because there is one dialect. A project contributes a data
declaration saying where its files live and nothing else; it supplies no
callable, so there is no seam through which a project could change what a
compiled document means.

## What one compile does

1. Read the authored bytes under the layer's budget and parse them into the
   closed dialect model. Noncanonical bytes, unknown fields, and more than one
   authored layer are refused here, before anything is resolved.
2. Resolve the one parent — another envelope of the same layer named by alias,
   or a frozen document named by repo-relative path — and pin its content and
   the lineage behind it.
3. Verify the envelope's assertions against that lineage, refusing an assertion
   that guards a path this envelope itself changes.
4. Lower the authored layer into ordered
   :class:`~feedbax.contracts.run_matrix.MatrixCompositionDelta` layers and
   apply them to the parent document.
5. Validate the result against the layer's real Feedbax output model.
6. Resolve the authored references into the compile lock's closed typed union.
7. Emit the compiled document and its lock.

## Where the refusals live

Step 4 is where the interesting failures happen, and they are not new code: they
are Feedbax's existing patch semantics. ``add`` refuses a path that already
exists, ``replace`` and ``remove`` refuse one that does not, and a layer that
overwrites a path an ancestor layer wrote must acknowledge it explicitly. An
envelope that states a change the base already made, or removes something the
base does not have, fails on those rules alone — which is why no derivation
check, no recomputation, and no project validator is needed to catch it.

Compilation is a pure function of tracked content. It allocates nothing, writes
nothing outside an explicitly requested output directory, and touches no
network. Everything it emits is therefore a compile *plan*: it may quote an
authenticated reference a previous run produced, and it may never author one.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_sha256,
    emit_text,
)
from feedbax.contracts.authoring_budget import AuthoringBudgets
from feedbax.contracts.experiment_compile_lock import (
    AnalysisInputBinding,
    AuthenticatedReceiptReference,
    CheckpointInitializationBinding,
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    EvaluationSubjectBinding,
    FigureRuntimeInputBinding,
    NotApplicableReference,
    PlannedProductReference,
    ContentPinReference,
    ReceiptLocatorReference,
    ReportParentBinding,
    build_compile_lock,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.experiment_envelope_dialect import (
    ANALYSIS_BUNDLE_OUTPUT,
    ANALYSIS_RUN_OUTPUT,
    EVALUATION_OUTPUT,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
    EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
    EXPERIMENT_ENVELOPE_SUFFIX,
    FIGURE_COMPOSITION_OUTPUT,
    FIGURE_OUTPUT,
    REPORT_OUTPUT,
    TRAINING_OUTPUT,
    AnalysisLayerAuthoring,
    EvaluationLayerAuthoring,
    ExperimentEnvelope,
    ExperimentEnvelopeLayer,
    FigureLayerAuthoring,
    LayerOutputContract,
    NotApplicableAuthoring,
    ReceiptReference,
    ReportLayerAuthoring,
    TrainingLayerAuthoring,
    output_contract_of_document,
    parse_experiment_envelope,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project_experiment import ProjectExperimentDeclaration
from feedbax.contracts.run_matrix import (
    MatrixCompositionDelta,
    apply_composition_deltas,
    apply_override_patches,
)
from feedbax.envelope.authoring import enforce_assertion_budget, read_authored_document
from feedbax.envelope.resolution import (
    Lineage,
    PinnedDocument,
    build_lineage,
    load_pinned,
)

_DELTA_ONLY_HOME = "an envelope carries only what changes; delete the line"


def _reject(
    category: ExperimentEnvelopeRejectionCategory,
    field: str,
    message: str,
    *,
    correct_home: str | None = None,
) -> NoReturn:
    raise ExperimentEnvelopeRejection(
        category, message, field=field, correct_home=correct_home
    )


@dataclass(frozen=True)
class EnvelopeLayout:
    """Where a project files its authored envelopes and its compiled outputs.

    This is repository layout, not dialect: the engine needs it to turn an alias
    into a path and to refuse a base that names compiled output, and needs
    nothing else from it.
    """

    envelope_directory: str
    output_directory: str
    envelope_suffix: str = EXPERIMENT_ENVELOPE_SUFFIX

    def __post_init__(self) -> None:
        for name in ("envelope_directory", "output_directory", "envelope_suffix"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"EnvelopeLayout {name} must be nonempty")

    @classmethod
    def of(cls, declaration: ProjectExperimentDeclaration) -> "EnvelopeLayout":
        """Return the layout one project's data declaration states."""
        return cls(
            envelope_directory=declaration.envelope_directory,
            output_directory=declaration.output_directory,
        )

    def alias_ref(self, alias: str) -> str:
        """Return the repo-relative envelope path one alias names."""
        return str(PurePosixPath(self.envelope_directory) / f"{alias}{self.envelope_suffix}")


@dataclass(frozen=True)
class ResolvedParent:
    """The one experiment parent, resolved and content-pinned."""

    kind: str
    ref: str
    pinned: PinnedDocument
    lineage: Lineage
    layer: ExperimentEnvelopeLayer
    contract: LayerOutputContract

    def base_block(self) -> dict[str, Any]:
        """Return the block a compiled document records its parent in."""
        block: dict[str, Any] = {
            "kind": "authored_intent",
            "ref": self.ref,
            "content_hash": self.pinned.content_hash,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }
        symbolic = self.pinned.document.get("name")
        if isinstance(symbolic, str):
            block["symbolic_name"] = symbolic
        return block

    def lock_record(self) -> dict[str, str]:
        """Return the parent's compile-lock pin record."""
        return {
            "kind": self.kind,
            "ref": self.ref,
            "content_hash": self.pinned.content_hash,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }


@dataclass(frozen=True)
class LayerCompileContext:
    """Everything the engine resolved before one layer lowers.

    Attributes:
        envelope: The parsed, validated authored envelope.
        envelope_ref: Its repo-relative path.
        layer: The one layer this envelope authors.
        parent: The resolved, content-pinned parent.
        repo_root: Root every reference resolves against.
        layout: Where envelopes and outputs live.
        declaration: The project's data declaration.
        compile_upstream: Compile another envelope by alias, cycle-checked. This
            is how a layer resolves a cross-layer reference into the two facts
            that exist before anything runs.
    """

    envelope: ExperimentEnvelope
    envelope_ref: str
    layer: ExperimentEnvelopeLayer
    parent: ResolvedParent
    repo_root: Path
    layout: EnvelopeLayout
    declaration: ProjectExperimentDeclaration
    compile_upstream: Callable[[str, str], "EnvelopeCompileOutcome"]

    @property
    def lineage(self) -> Lineage:
        """Return the parent's content-pinned lineage."""
        return self.parent.lineage


@dataclass(frozen=True)
class LoweredLayer:
    """What one layer lowered to, before the deltas are applied.

    Attributes:
        contract: The output contract the compiled document must satisfy.
        deltas: Ordered composition layers to apply to the parent document.
        references: Typed compile-lock references this layer resolved.
        identity_contributions: Compile-time facts beyond the document that make
            two otherwise-identical plans different executions.
    """

    contract: LayerOutputContract
    deltas: Sequence[MatrixCompositionDelta]
    references: Sequence[Any] = ()
    identity_contributions: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @property
    def authored_paths(self) -> dict[str, str]:
        """Return every path this layer decides, mapped to the field deciding it."""
        return {
            patch.path: f"envelope.{delta.layer_id}"
            for delta in self.deltas
            for patch in delta.patches
        }


@dataclass(frozen=True)
class EnvelopeCompileOutcome:
    """The two outputs of one compile: a document and its compile lock."""

    name: str
    family: str
    layer: ExperimentEnvelopeLayer
    document: Any
    compile_lock: dict[str, Any]


def scalar_equal(left: Any, right: Any) -> bool:
    """Compare two authored scalars without conflating booleans with integers."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def reject_echo(field: str, value: Any, owner_ref: str) -> None:
    """Refuse an authored leaf that merely restates what it inherits."""
    _reject(
        ExperimentEnvelopeRejectionCategory.ECHOED_INHERITED_VALUE,
        field,
        f"{value!r} is already the inherited value, owned by {owner_ref}",
        correct_home=_DELTA_ONLY_HOME,
    )


def check_echo(lineage: Lineage, path: str, value: Any, *, field: str) -> None:
    """Refuse ``value`` at ``field`` when the lineage already states it at ``path``."""
    found = lineage.lookup(path)
    if found is not None and scalar_equal(found.value, value):
        reject_echo(field, value, found.owner_ref)


def verify_assertions(
    assertions: Sequence[Any],
    lineage: Lineage,
    overridden: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Check inherited preconditions, refusing one that guards an authored path.

    An assertion guards a value the envelope inherits *unchanged*. Guarding a
    path the envelope itself decides is a tautology that would silently start
    passing for the wrong reason, so it is refused rather than checked.
    """
    records: list[dict[str, Any]] = []
    for index, assertion in enumerate(assertions):
        owner_field = overridden.get(assertion.path) or next(
            (
                authored
                for path, authored in overridden.items()
                if assertion.path.startswith(f"{path}.")
            ),
            None,
        )
        if owner_field is not None:
            _reject(
                ExperimentEnvelopeRejectionCategory.ILLEGAL_ASSERTION_PATH,
                f"envelope.assert[{index}].path",
                f"{assertion.path!r} is changed by this envelope at {owner_field}; an "
                "assertion may only guard a value the envelope inherits unchanged",
                correct_home=f"the value this envelope authors is already stated at "
                f"{owner_field}; delete the assertion",
            )
        found = lineage.lookup(assertion.path)
        if found is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.ILLEGAL_ASSERTION_PATH,
                f"envelope.assert[{index}].path",
                f"{assertion.path!r} is not inherited from the resolved parent, so there is "
                "no precondition to check",
                correct_home="an assertion guards an inherited value; a fact the base does "
                "not state belongs in the base, not in an assertion",
            )
        if not scalar_equal(found.value, assertion.equals):
            _reject(
                ExperimentEnvelopeRejectionCategory.ASSERTION_FAILED,
                f"envelope.assert[{index}]",
                f"{assertion.path!r} expected {assertion.equals!r} but the resolved parent "
                f"has {found.value!r}, owned by {found.owner_ref}",
                correct_home=f"{found.owner_ref} owns this value; change the assertion to "
                "match it, or change the base if the base is what is wrong",
            )
        records.append(
            {
                "path": assertion.path,
                "expected": assertion.equals,
                "actual": found.value,
                "owner_ref": found.owner_ref,
            }
        )
    return records


# -- authored references -> typed lock references -------------------------


def _reference_for(
    context: LayerCompileContext,
    authored: Any,
    *,
    role_path: str,
    field: str,
    consumer_of: Callable[[str, str], Any],
) -> Any:
    """Lower one authored reference onto the compile lock's closed union.

    ``consumer_of`` receives the resolved upstream ``(kind, id)`` so a layer can
    state its own consumer binding without this function knowing what a subject,
    an alias-role, an input authority, or a parent means.
    """
    if isinstance(authored, NotApplicableAuthoring):
        return NotApplicableReference(
            role_path=role_path, basis="authored", reason=authored.reason
        )
    if isinstance(authored, ReceiptReference):
        consumer = consumer_of(authored.manifest_kind, authored.manifest_id)
        if authored.is_authenticated:
            return AuthenticatedReceiptReference(
                manifest_kind=authored.manifest_kind,
                manifest_id=authored.manifest_id,
                manifest_sha256=str(authored.manifest_sha256),
                size_bytes=int(authored.size_bytes or 0),
                role_path=role_path,
                consumer=consumer,
                execution_uri=authored.execution_uri,
            )
        return ReceiptLocatorReference(
            manifest_kind=authored.manifest_kind,
            manifest_id=authored.manifest_id,
            role_path=role_path,
            consumer=consumer,
        )
    upstream = context.compile_upstream(authored.alias, field)
    lock = upstream.compile_lock
    contract = output_contract_of_document(upstream.document)
    if contract is None:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_UPSTREAM_REFERENCE,
            field,
            f"{authored.alias!r} compiles to a document of no known Feedbax output family",
        )
    return PlannedProductReference(
        envelope_ref=str(lock["envelope"]["ref"]),
        envelope_hash=str(lock["envelope"]["envelope_hash"]),
        product_name=upstream.name,
        product_schema_id=contract.schema_id,
        product_schema_version=contract.schema_version,
        compiled_content_hash=str(lock["compiled_document"]["content_hash"]),
        role_path=role_path,
        consumer=consumer_of(contract.schema_id, upstream.name),
    )


# -- generic patch helpers -------------------------------------------------


def _params_patches(params: Mapping[str, Any], prefix: str) -> list[OverridePatch]:
    """Return one ``add`` patch per authored parameter.

    ``add`` is deliberate: a parameter the parent already states is not a new
    parameter, and restating it is the echo an envelope must not carry. Changing
    an inherited parameter is what the layer's native ``delta`` is for, where the
    change is visible as a ``replace``.
    """
    return [
        OverridePatch(path=f"{prefix}.{key}", op="add", value=params[key])
        for key in sorted(params)
    ]


def _tag_patches(
    inherited: Sequence[Any], tags: Any, *, field: str
) -> list[OverridePatch]:
    """Return ordered patches realizing a tags delta over the inherited list.

    Additions are emitted first, appending past the end, and removals second
    against the list as it then stands. Doing it the other way round would have a
    removal free an index that a later addition immediately reclaims, and one
    composition layer may not write the same path twice.
    """
    if tags is None:
        return []
    current = list(inherited)
    patches: list[OverridePatch] = []
    for tag in tags.add:
        if tag in current:
            reject_echo(f"{field}.add", tag, "the resolved parent")
        patches.append(OverridePatch(path=f"tags.{len(current)}", op="add", value=tag))
        current.append(tag)
    for tag in tags.remove:
        if tag not in current:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{field}.remove",
                f"{tag!r} is not an inherited tag, so there is nothing to remove",
                correct_home="a tags delta removes a tag the base states",
            )
        index = current.index(tag)
        patches.append(OverridePatch(path=f"tags.{index}", op="remove"))
        current.pop(index)
    return patches


def _resolve_matrix_base_payload(
    base: Any, repo_root: Path, *, field: str
) -> tuple[dict[str, Any], ContentPinReference | None]:
    """Return the payload a matrix's ``base`` block names, and the bytes it read.

    This is the document a row's patches actually apply to, which is why the row
    delta's add/replace/remove legality can be decided at compile time at all. A
    payload that lives in its own file is bytes this compile *read* and is
    reported as a content pin: it is an input, and nothing runs because of it.
    """
    if not isinstance(base, Mapping):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            "the resolved parent states no base payload to patch",
        )
    if base.get("kind") == "inline":
        inline = base.get("inline")
        if not isinstance(inline, Mapping):
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                field,
                "an inline base states an object payload",
            )
        payload: Any = deepcopy(dict(inline))
        pin: ContentPinReference | None = None
    else:
        ref = base.get("ref")
        if not isinstance(ref, str):
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"a {base.get('kind')!r} base names the document it pins",
            )
        pinned = load_pinned(repo_root, ref)
        if pinned is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{ref!r} is not a readable repo-relative JSON document",
            )
        declared = base.get("content_hash") or base.get("resolved_root_hash")
        if isinstance(declared, str) and declared != pinned.content_hash:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{ref!r} has content hash {pinned.content_hash} but the base pins {declared}",
                correct_home="repin the base against the bytes actually tracked",
            )
        payload = deepcopy(dict(pinned.document))
        pin = ContentPinReference(ref=ref, content_hash=pinned.content_hash)
    payload_path = base.get("payload_path")
    if isinstance(payload_path, str) and payload_path:
        for part in payload_path.split("."):
            if not isinstance(payload, Mapping) or part not in payload:
                _reject(
                    ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                    field,
                    f"the base payload path {payload_path!r} is not present",
                )
            payload = payload[part]
        payload = deepcopy(dict(payload))
    return payload, pin


def _prove_patches_apply(
    payload: Mapping[str, Any], patches: Sequence[OverridePatch], *, field: str
) -> None:
    """Prove one row's patches are legal against the payload they will patch.

    Nothing is kept: the point is the refusal. ``add`` on an existing path,
    ``replace`` or ``remove`` on an absent one, and any malformed path all raise
    here, at compile time, naming the authored field.
    """
    try:
        apply_override_patches(dict(deepcopy(payload)), list(patches))
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"the authored delta does not apply to the base payload: {exc}",
            correct_home="a delta states what changes relative to the base it inherits; "
            "add a path that is absent, replace or remove one that is present",
        )


# -- the five layers -------------------------------------------------------


def _lower_training(context: LayerCompileContext) -> LoweredLayer:
    """Lower authored rows, tags, and checkpoint sources over a run matrix."""
    authored = context.envelope.content
    assert isinstance(authored, TrainingLayerAuthoring)
    parent = dict(context.parent.pinned.document)
    rows = list(parent.get("rows") or [])
    by_id = {str(row.get("row_id")): row for row in rows if isinstance(row, Mapping)}
    base_payload, base_pin = _resolve_matrix_base_payload(
        parent.get("base"), context.repo_root, field=f"{context.parent.ref}#base"
    )

    references: list[Any] = [] if base_pin is None else [base_pin]
    patches: list[OverridePatch] = []
    if context.envelope.name != parent.get("name"):
        patches.append(
            OverridePatch(path="name", op="replace", value=context.envelope.name)
        )
    else:
        check_echo(context.lineage, "name", context.envelope.name, field="envelope.name")

    for index, row in enumerate(authored.rows):
        field = f"training.rows[{index}]"
        source = by_id.get(row.from_)
        if source is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY,
                f"{field}.from",
                f"{row.from_!r} names no row in {context.parent.ref}; "
                f"rows: {sorted(by_id)}",
            )
        if row.id in by_id:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{field}.id",
                f"{row.id!r} is already a row in {context.parent.ref}",
                correct_home="a new row states a new id; changing an inherited row is a "
                "delta on that row, not a second row with its name",
            )
        new_row = deepcopy(dict(source))
        new_row["row_id"] = row.id
        if row.seed is not None:
            if scalar_equal(source.get("seed"), row.seed):
                reject_echo(f"{field}.seed", row.seed, context.parent.ref)
            new_row["seed"] = row.seed
        if row.replaces is not None:
            metadata = dict(new_row.get("metadata") or {})
            metadata["replaces"] = row.replaces.model_dump(mode="json", exclude_none=True)
            new_row["metadata"] = metadata
        inherited_overrides = [
            OverridePatch.model_validate(item) for item in (source.get("overrides") or [])
        ]
        if row.delta is not None:
            _prove_patches_apply(
                apply_override_patches(dict(base_payload), list(inherited_overrides)),
                row.delta.patches,
                field=f"{field}.delta",
            )
            new_row["overrides"] = [
                patch.model_dump(mode="json", exclude_none=True)
                for patch in (*inherited_overrides, *row.delta.patches)
            ]
        patches.append(
            OverridePatch(path=f"rows.{len(rows)}", op="add", value=new_row)
        )
        rows.append(new_row)
        by_id[row.id] = new_row

    patches.extend(
        _tag_patches(parent.get("tags") or [], authored.tags, field="training.tags")
    )

    for index, item in enumerate(authored.checkpoint_initialization):
        if item.row not in by_id:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY,
                f"training.checkpoint_initialization[{index}].row",
                f"{item.row!r} is neither an inherited nor an authored row",
            )
        references.append(
            _reference_for(
                context,
                item.source,
                role_path=f"rows.{item.row}.checkpoint_initialization",
                field=f"training.checkpoint_initialization[{index}].source",
                consumer_of=lambda _kind, _id, item=item: CheckpointInitializationBinding(
                    mode=item.mode, row_id=item.row
                ),
            )
        )
    return LoweredLayer(
        contract=TRAINING_OUTPUT,
        deltas=_one_delta(context, patches),
        references=references,
    )


def _lower_evaluation(context: LayerCompileContext) -> LoweredLayer:
    """Lower a typed subject reference and recipe parameters over an eval matrix."""
    authored = context.envelope.content
    assert isinstance(authored, EvaluationLayerAuthoring)
    patches: list[OverridePatch] = []
    if authored.recipe is not None:
        check_echo(
            context.lineage,
            "base.evaluation_type",
            authored.recipe,
            field="evaluation.recipe",
        )
        patches.append(
            OverridePatch(path="base.evaluation_type", op="replace", value=authored.recipe)
        )
    patches.extend(
        _params_patches(authored.params, "base.params")
    )
    reference = _reference_for(
        context,
        authored.subject,
        role_path=f"subjects.{authored.subject_id}",
        field="evaluation.subject",
        consumer_of=lambda _kind, _id: EvaluationSubjectBinding(
            subject_id=authored.subject_id
        ),
    )
    return LoweredLayer(
        contract=EVALUATION_OUTPUT,
        deltas=_one_delta(context, patches, authored.delta),
        references=[reference],
    )


def _lower_analysis(context: LayerCompileContext) -> LoweredLayer:
    """Lower a run-or-bundle delta, typed subjects, and parameters."""
    authored = context.envelope.content
    assert isinstance(authored, AnalysisLayerAuthoring)
    contract = ANALYSIS_RUN_OUTPUT if authored.target == "run" else ANALYSIS_BUNDLE_OUTPUT
    if context.parent.contract is not contract:
        _reject(
            ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE,
            "analysis.target",
            f"an analysis {authored.target!r} envelope resolves {context.parent.ref}, which "
            f"is a {context.parent.contract.family!r} document",
        )
    patches: list[OverridePatch] = []
    if authored.recipe is not None:
        check_echo(
            context.lineage, "analysis_type", authored.recipe, field="analysis.recipe"
        )
        patches.append(
            OverridePatch(path="analysis_type", op="replace", value=authored.recipe)
        )
    prefix = "params" if authored.target == "run" else "params_base"
    patches.extend(_params_patches(authored.params, prefix))
    references = [
        _reference_for(
            context,
            subject.ref,
            role_path=f"inputs.{subject.alias}",
            field=f"analysis.subjects[{index}].ref",
            consumer_of=lambda _kind, _id, subject=subject: AnalysisInputBinding(
                alias=subject.alias, role=subject.role
            ),
        )
        for index, subject in enumerate(authored.subjects)
    ]
    return LoweredLayer(
        contract=contract,
        deltas=_one_delta(context, patches, authored.delta),
        references=references,
    )


def _lower_figure(context: LayerCompileContext) -> LoweredLayer:
    """Lower figure input-role bindings and a native figure composition delta."""
    authored = context.envelope.content
    assert isinstance(authored, FigureLayerAuthoring)
    contract = (
        FIGURE_COMPOSITION_OUTPUT
        if context.parent.contract is FIGURE_COMPOSITION_OUTPUT
        else FIGURE_OUTPUT
    )
    delta = None if authored.delta is None else authored.delta.matrix_delta()
    references = [
        _reference_for(
            context,
            item.ref,
            role_path=f"inputs.{item.input_role}",
            field=f"figure.inputs[{index}].ref",
            consumer_of=lambda _kind, _id, item=item: FigureRuntimeInputBinding(
                input_role=item.input_role
            ),
        )
        for index, item in enumerate(authored.inputs)
    ]
    return LoweredLayer(
        contract=contract,
        deltas=_one_delta(context, [], delta),
        references=references,
    )


def _lower_report(context: LayerCompileContext) -> LoweredLayer:
    """Lower ordered report role bindings, including authored not-applicability."""
    authored = context.envelope.content
    assert isinstance(authored, ReportLayerAuthoring)
    references = [
        _reference_for(
            context,
            binding.ref,
            role_path=binding.role_path,
            field=f"report.bindings[{index}].ref",
            consumer_of=lambda kind, identifier: ReportParentBinding(
                parent_kind=kind, parent_id=identifier
            ),
        )
        for index, binding in enumerate(authored.bindings)
    ]
    return LoweredLayer(
        contract=REPORT_OUTPUT,
        deltas=_one_delta(context, [], authored.delta),
        references=references,
    )


def _one_delta(
    context: LayerCompileContext,
    generated: Sequence[OverridePatch],
    authored: MatrixCompositionDelta | None = None,
) -> list[MatrixCompositionDelta]:
    """Return the ordered composition layers one envelope contributes.

    The generated layer comes first and the authored native delta second, so an
    authored patch that overwrites something the envelope's own structured fields
    decided has to acknowledge it exactly as it would an ancestor's.
    """
    deltas: list[MatrixCompositionDelta] = []
    if generated:
        deltas.append(
            MatrixCompositionDelta(
                layer_id=f"{context.envelope.name}.{context.layer.value}",
                patches=list(generated),
            )
        )
    if authored is not None:
        deltas.append(authored)
    return deltas


_LAYER_LOWERERS: Mapping[ExperimentEnvelopeLayer, Callable[[LayerCompileContext], LoweredLayer]] = {
    ExperimentEnvelopeLayer.TRAINING: _lower_training,
    ExperimentEnvelopeLayer.EVALUATION: _lower_evaluation,
    ExperimentEnvelopeLayer.ANALYSIS: _lower_analysis,
    ExperimentEnvelopeLayer.FIGURE: _lower_figure,
    ExperimentEnvelopeLayer.REPORT: _lower_report,
}


def authored_layer_of(document: Mapping[str, Any]) -> str | None:
    """Return the single layer key an authored document states, before parsing.

    The budget for a layer has to be chosen before the document is parsed, and
    this is the cheapest honest way to choose it: a document naming zero or more
    than one layer key gets the widest caps and is refused by the closed model a
    moment later.
    """
    if not isinstance(document, Mapping):
        return None
    present = [layer.value for layer in ExperimentEnvelopeLayer if layer.value in document]
    return present[0] if len(present) == 1 else None


class EnvelopeKernel:
    """The one compiler, bound to one project's data declaration."""

    def __init__(
        self,
        *,
        declaration: ProjectExperimentDeclaration,
        budgets: AuthoringBudgets,
        implementation: CompilerImplementation,
    ) -> None:
        self.declaration = declaration
        self.layout = EnvelopeLayout.of(declaration)
        self.budgets = budgets
        self.implementation = implementation
        self.contract = CompilerContract(
            contract_id=EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
            contract_version=EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION,
        )

    # -- envelope reading ------------------------------------------------

    def read_envelope(self, raw: bytes, *, envelope_ref: str) -> ExperimentEnvelope:
        """Read one authored envelope under its layer's budget."""
        document = read_authored_document(
            raw, self.budgets, field=envelope_ref, layer_of=authored_layer_of
        )
        return parse_experiment_envelope(document, field=envelope_ref)

    # -- parent resolution -----------------------------------------------

    def refuse_compiled_output_base(self, base: str, field: str) -> None:
        """Refuse a base that names this engine's own compiled output.

        A base is authored intent: a frozen document nobody compiles, or another
        envelope named by alias. A compiled document is the engine's own product;
        inheriting from one would couple authored intent to output bytes and
        invert the compile order. The output directory is also a runtime choice,
        so a document pinned there is pinned to a path that can move.

        The path is normalized first, so ``out/x.json``, ``./out/x.json``, and
        ``specs/../out/x.json`` are one rule rather than three holes.
        """
        normalized = posixpath.normpath(base)
        if PurePosixPath(normalized).parts[:1] != (self.layout.output_directory,):
            return
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            field,
            f"{base!r} is under the {self.layout.output_directory!r} directory, which holds "
            "compiled output rather than authored intent; a compiled document is this "
            "engine's own product and is not an experiment parent",
            correct_home="a base is either a frozen document nobody compiles, named by its "
            f"repo-relative path outside {self.layout.output_directory!r}, or another "
            f"envelope named by its alias in {self.layout.envelope_directory!r}",
        )

    def resolve_parent(
        self,
        repo_root: Path,
        base: str,
        stack: tuple[str, ...],
        *,
        expected_layer: ExperimentEnvelopeLayer,
        field: str = "envelope.base",
    ) -> ResolvedParent:
        """Resolve the single parent: an envelope alias or a frozen document.

        The parent must belong to ``expected_layer``. The layer is read from the
        *resolved document's own declared schema id*, never from the base string,
        so an alias chain cannot smuggle a cross-layer parent past the check.
        """
        self.refuse_compiled_output_base(base, field)
        if base.endswith(".json"):
            parent = self._resolve_frozen_parent(repo_root, base, field)
        else:
            parent = self._resolve_alias_parent(repo_root, base, stack, field)
        if parent.layer is not expected_layer:
            _reject(
                ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE,
                field,
                f"a {expected_layer.value!r} envelope resolves {base!r} to {parent.ref}, "
                f"which is a {parent.layer.value!r} document; an envelope inherits from its "
                "own layer only",
                correct_home=f"the {parent.layer.value} document is the base of a "
                f"{parent.layer.value} envelope; a {expected_layer.value} envelope reaches "
                "it by name, as a cross-layer reference, not as a parent",
            )
        return parent

    def _resolve_frozen_parent(self, repo_root: Path, base: str, field: str) -> ResolvedParent:
        pinned = load_pinned(repo_root, base, skips=("rows",))
        if pinned is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{base!r} is not a readable repo-relative JSON document",
            )
        contract = output_contract_of_document(pinned.document)
        if contract is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{base!r} declares schema_id {pinned.document.get('schema_id')!r}, which is "
                "no Feedbax layer output family, so it is not an experiment parent",
                correct_home="a base is a document of the same layer as the envelope that "
                "names it; an unrelated document is read through its own layer's envelope",
            )
        return ResolvedParent(
            "frozen_document",
            base,
            pinned,
            build_lineage(repo_root, pinned),
            contract.layer,
            contract,
        )

    def _resolve_alias_parent(
        self, repo_root: Path, base: str, stack: tuple[str, ...], field: str
    ) -> ResolvedParent:
        alias_ref = self.layout.alias_ref(base)
        if not (repo_root / alias_ref).is_file():
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{base!r} resolves to neither the envelope {alias_ref!r} nor a "
                "repo-relative path to an existing document",
            )
        if base in stack:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"envelope alias cycle: {' -> '.join((*stack, base))}",
            )
        outcome = self.compile_envelope_file(
            repo_root / alias_ref, repo_root=repo_root, _stack=(*stack, base)
        )
        pinned = PinnedDocument.of(alias_ref, outcome.document, skips=("rows",))
        contract = output_contract_of_document(outcome.document)
        assert contract is not None
        return ResolvedParent(
            "envelope_alias",
            alias_ref,
            pinned,
            build_lineage(repo_root, pinned),
            contract.layer,
            contract,
        )

    # -- compilation ------------------------------------------------------

    def compile_envelope(
        self,
        raw: bytes,
        *,
        repo_root: Path,
        envelope_ref: str,
        _stack: tuple[str, ...] = (),
    ) -> EnvelopeCompileOutcome:
        """Compile one authored envelope into a document and its compile lock."""
        envelope = self.read_envelope(raw, envelope_ref=envelope_ref)
        layer = envelope.layer
        budget = self.budgets.for_layer(layer.value)
        enforce_assertion_budget(
            len(envelope.assert_), budget, field=f"{envelope_ref}#assert"
        )
        if envelope.base is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
                f"{envelope_ref}#base",
                "an envelope states the document it changes; there is no rootless "
                "envelope, because a delta with nothing to apply to is a whole document",
                correct_home="state a frozen document by repo-relative path, or another "
                f"envelope in {self.layout.envelope_directory!r} by alias",
            )
        parent = self.resolve_parent(repo_root, envelope.base, _stack, expected_layer=layer)

        def compile_upstream(alias: str, field: str) -> EnvelopeCompileOutcome:
            upstream_ref = self.layout.alias_ref(alias)
            if not (repo_root / upstream_ref).is_file():
                _reject(
                    ExperimentEnvelopeRejectionCategory.UNRESOLVED_UPSTREAM_REFERENCE,
                    field,
                    f"{alias!r} names no envelope at {upstream_ref!r}",
                    correct_home="a cross-layer reference names the alias of the envelope "
                    "that produces it",
                )
            if alias in _stack:
                _reject(
                    ExperimentEnvelopeRejectionCategory.UNRESOLVED_UPSTREAM_REFERENCE,
                    field,
                    f"envelope reference cycle: {' -> '.join((*_stack, alias))}",
                )
            return self.compile_envelope_file(
                repo_root / upstream_ref, repo_root=repo_root, _stack=(*_stack, alias)
            )

        context = LayerCompileContext(
            envelope=envelope,
            envelope_ref=envelope_ref,
            layer=layer,
            parent=parent,
            repo_root=repo_root,
            layout=self.layout,
            declaration=self.declaration,
            compile_upstream=compile_upstream,
        )
        lowered = _LAYER_LOWERERS[layer](context)
        assertion_records = verify_assertions(
            envelope.assert_, context.lineage, lowered.authored_paths
        )
        document = self._compose(context, lowered)

        lock = build_compile_lock(
            CompileLockInputs(
                envelope_ref=envelope_ref,
                envelope_document=envelope.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
                envelope_schema=envelope.schema_,
                name=envelope.name,
                family=lowered.contract.family,
                compiled_document=document,
                contract=self.contract,
                implementation=self.implementation,
                base=parent.lock_record(),
                lineage_pins=parent.lineage.pins(),
                resolved_deltas={
                    delta.layer_id: delta.model_dump(mode="json", exclude_none=True)
                    for delta in lowered.deltas
                },
                references=lowered.references,
                assertions=assertion_records,
                identity_contributions=lowered.identity_contributions,
                issue=envelope.issue,
            )
        )
        return EnvelopeCompileOutcome(
            name=envelope.name,
            family=lowered.contract.family,
            layer=layer,
            document=document,
            compile_lock=lock,
        )

    def _compose(self, context: LayerCompileContext, lowered: LoweredLayer) -> dict[str, Any]:
        """Apply the lowered layers to the parent and prove the result is valid."""
        try:
            document, _attribution, _written = apply_composition_deltas(
                deepcopy(dict(context.parent.pinned.document)), list(lowered.deltas)
            )
        except ValueError as exc:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                context.envelope_ref,
                f"the envelope's deltas do not apply to {context.parent.ref}: {exc}",
                correct_home="a delta states what changes relative to the base it inherits, "
                "and acknowledges every ancestor-written path it overwrites",
            )
        try:
            lowered.contract.model().model_validate(document)
        except Exception as exc:  # noqa: BLE001 - any model failure is one rejection
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                context.envelope_ref,
                f"the compiled document is not a valid {lowered.contract.schema_id}: {exc}",
                correct_home="the compiled document must be a member of the Feedbax family "
                "its layer produces; the envelope's delta is what makes it one",
            )
        return document

    def compile_envelope_file(
        self,
        path: Path,
        *,
        repo_root: Path,
        _stack: tuple[str, ...] = (),
    ) -> EnvelopeCompileOutcome:
        """Compile the envelope at ``path``, which must lie inside ``repo_root``."""
        resolved = path.resolve()
        try:
            envelope_ref = str(resolved.relative_to(repo_root.resolve()))
        except ValueError as exc:
            raise ExperimentEnvelopeRejection(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                "an envelope must be a repo-relative document inside the repository",
                field=str(path),
            ) from exc
        return self.compile_envelope(
            resolved.read_bytes(),
            repo_root=repo_root,
            envelope_ref=envelope_ref,
            _stack=_stack,
        )

    # -- output ------------------------------------------------------------

    def output_paths(self, outcome: EnvelopeCompileOutcome, out_dir: Path) -> dict[str, Path]:
        """Return where one compile's two outputs belong, without writing them."""
        return {
            "compile_lock": out_dir / f"{outcome.name}.compile-lock.json",
            "document": out_dir / f"{outcome.name}.{outcome.family}.json",
        }

    def write_outputs(self, outcome: EnvelopeCompileOutcome, out_dir: Path) -> dict[str, Path]:
        """Write both outputs deterministically; re-running rewrites identical bytes."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = self.output_paths(outcome, out_dir)
        paths["compile_lock"].write_text(emit_text(outcome.compile_lock), encoding="utf-8")
        paths["document"].write_text(emit_text(outcome.document), encoding="utf-8")
        return paths

    def envelopes(self, repo_root: Path) -> list[Path]:
        """Return every authored envelope in the project's envelope directory."""
        directory = repo_root / self.layout.envelope_directory
        if not directory.is_dir():
            return []
        return sorted(directory.glob(f"*{self.layout.envelope_suffix}"))


def check_no_co_created_protected_document(
    changed_paths: Sequence[str],
    envelope_ref: str,
    protected_suffixes: Sequence[str],
) -> None:
    """Refuse an authoring change that also creates a protected document.

    A protected document is one whose creation is a stop-and-ask event getting
    concentrated review. Letting an ordinary authoring change introduce one is
    how content launders itself into the inherited material nobody re-reads.
    Which suffixes are protected is the caller's policy; that the rule exists is
    the engine's.
    """
    if not protected_suffixes:
        return
    for path in changed_paths:
        if path == envelope_ref:
            continue
        if path.endswith(tuple(protected_suffixes)):
            _reject(
                ExperimentEnvelopeRejectionCategory.CO_CREATED_PROTECTED_DOCUMENT,
                path,
                "a protected document may not be created in the same change as an "
                "authored envelope; land it separately so it gets its own review",
            )


def compiled_document_pin(document: Any) -> dict[str, str]:
    """Return the content pin a cross-layer reference records for a document."""
    return {
        "compiled_document_hash": canonical_sha256(document),
        "pin_algorithm": CANONICAL_PIN_ALGORITHM,
    }


__all__ = [
    "EnvelopeCompileOutcome",
    "EnvelopeKernel",
    "EnvelopeLayout",
    "LayerCompileContext",
    "LoweredLayer",
    "ResolvedParent",
    "authored_layer_of",
    "check_echo",
    "check_no_co_created_protected_document",
    "compiled_document_pin",
    "reject_echo",
    "scalar_equal",
    "verify_assertions",
]
