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
   the lineage behind it. The lineage follows the parent's ``sources`` block as
   well as its base chain, because a document a parent draws values from is a
   document this compile read.
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

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import UnionType
from typing import Annotated, Any, NoReturn, Union, get_args, get_origin

from pydantic import BaseModel

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
    RowProvenanceReference,
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
    REPORT_PARAMS_MODELS,
    TRAINING_OUTPUT,
    AnalysisLayerAuthoring,
    EvaluationLayerAuthoring,
    ExperimentEnvelope,
    ExperimentEnvelopeLayer,
    FigureLayerAuthoring,
    FigureLayerMode,
    LayerOutputContract,
    NotApplicableAuthoring,
    ReceiptReference,
    ReportLayerAuthoring,
    TrainingLayerAuthoring,
    TrainingRowsMode,
    output_contract_of_document,
    parse_experiment_envelope,
)
from feedbax.contracts.figure_roles import (
    FigureRoleBindingContract,
    FigureRowExpansionRequest,
    expand_figure_rows_structure,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.row_index import (
    ROW_INDEX_SCHEMA_ID,
    AuthenticatedRowIndex,
    RowSelectionError,
    RowSelectionErrorCode,
    expand_row_selector,
)
from feedbax.contracts.project_experiment import (
    ProjectExperimentDeclaration,
    path_is_within,
)
from feedbax.contracts.run_matrix import (
    MatrixCompositionDelta,
    apply_composition_deltas,
    apply_override_patches,
)
from feedbax.envelope.authoring import (
    enforce_assertion_budget,
    enforce_row_budget,
    read_authored_document,
)
from feedbax.envelope.resolution import (
    Lineage,
    PinnedDocument,
    build_lineage,
    load_pinned,
)

_DELTA_ONLY_HOME = "an envelope carries only what changes; delete the line"

#: Row-selection failures, mapped onto the authoring rejection vocabulary. The
#: selector machinery has its own stable codes because it is used outside
#: authoring too; an envelope's author needs the answer in one vocabulary.
_ROW_SELECTION_REJECTIONS: Mapping[
    RowSelectionErrorCode, ExperimentEnvelopeRejectionCategory
] = {
    RowSelectionErrorCode.EMPTY_SELECTION: (
        ExperimentEnvelopeRejectionCategory.EMPTY_SELECTION
    ),
    RowSelectionErrorCode.UNRESOLVED_ROW_KEY: (
        ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY
    ),
    RowSelectionErrorCode.DUPLICATE_ROW_ID: (
        ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY
    ),
    RowSelectionErrorCode.AMBIGUOUS_ROW_BINDING: (
        ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    ),
    RowSelectionErrorCode.INDEX_MISMATCH: (
        ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    ),
}


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


#: The block a Feedbax matrix document names the further documents it reads in.
#: Each entry is a :class:`~feedbax.contracts.extraction.SourceBinding`: an alias,
#: a kind, and a repo-relative ``uri``. A source is a document the compile *read*,
#: so its bytes belong in the content-pinned lineage exactly as a chained base's
#: do — naming one without pinning it is a lineage entry the lock quietly loses.
DOCUMENT_SOURCES_KEY = "sources"


def source_refs_of(repo_root: Path, ref: str) -> Callable[[Mapping[str, Any]], list[str]]:
    """Return how one parent document names the sources its lineage must pin.

    Every entry must resolve. A source that cannot be read is exactly the silent
    gap this pass exists to close, so it is a rejection rather than a skipped
    link. The single exception is a source the binding itself declares
    ``optional``, which is the *document* stating that its absence is intended
    rather than the walk inferring it.

    Args:
        repo_root: Root every source uri is resolved against.
        ref: The parent document, named for the rejections raised against it.

    Returns:
        The callable :func:`~feedbax.envelope.resolution.build_lineage` follows.
    """

    def source_refs(document: Mapping[str, Any]) -> list[str]:
        sources = document.get(DOCUMENT_SOURCES_KEY)
        if sources is None:
            return []
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{ref}#{DOCUMENT_SOURCES_KEY}",
                f"{DOCUMENT_SOURCES_KEY!r} is a list of source bindings",
            )
        refs: list[str] = []
        for index, source in enumerate(sources):
            field = f"{ref}#{DOCUMENT_SOURCES_KEY}[{index}].uri"
            if not isinstance(source, Mapping) or not isinstance(source.get("uri"), str):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    field,
                    "a source binding names the repo-relative document it reads",
                )
            uri = str(source["uri"])
            if load_pinned(repo_root, uri) is not None:
                refs.append(uri)
            elif source.get("optional") is not True:
                _reject(
                    ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                    field,
                    f"{uri!r} is not a readable repo-relative JSON document, so the bytes "
                    "this compile reads through it cannot be pinned",
                    correct_home="a source names a tracked JSON document; a source that may "
                    "legitimately be absent declares itself optional and states the payload "
                    "to use when it is",
                )
        return refs

    return source_refs


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

    A layer either *patches* its parent, stating ordered ``deltas``, or
    *constructs* its output from the pinned parent, stating ``document``. The two
    are exclusive: a constructed document is already the whole answer, and a
    delta applied afterwards would be a second, invisible authority over it.

    Attributes:
        contract: The output contract the compiled document must satisfy.
        deltas: Ordered composition layers to apply to the parent document.
        document: The compiled document, when this layer constructs rather than
            patches. ``None`` means the deltas decide.
        references: Typed compile-lock references this layer resolved.
        row_provenance: One typed record per compiled row this layer derived from
            a row of the resolved parent.
        identity_contributions: Compile-time facts beyond the document that make
            two otherwise-identical plans different executions.
    """

    contract: LayerOutputContract
    deltas: Sequence[MatrixCompositionDelta]
    document: Mapping[str, Any] | None = None
    references: Sequence[Any] = ()
    row_provenance: Sequence[Any] = ()
    identity_contributions: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.document is not None and self.deltas:
            raise ValueError(
                "a lowered layer either patches its parent or constructs its document; "
                "stating both would give the compiled document two authorities"
            )

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
) -> tuple[list[OverridePatch], list[str]]:
    """Return ordered patches realizing a tags delta, and the paths they rewrite.

    Additions are emitted first, appending past the end, and removals second
    against the list as it then stands. Doing it the other way round would have a
    removal free an index that a later addition immediately reclaims.

    Removing more than one tag necessarily writes the same index twice: the list
    closes up behind the first removal, so the second tag arrives at a position
    an earlier patch already decided. Composition refuses an unacknowledged
    overwrite, and a training envelope has no layer-level delta to acknowledge
    through, so the *generated* layer states the acknowledgement itself. That is
    normative derivation, not an authoring loophole: the engine is acknowledging
    a path the engine's own derivation just wrote, computed from the tags delta
    and the inherited list and from nothing else. Add-only authoring writes each
    index once and acknowledges nothing, exactly as before.

    Returns:
        The ordered patches, and the sorted paths the generated layer rewrites.
    """
    if tags is None:
        return [], []
    current = list(inherited)
    patches: list[OverridePatch] = []
    written: set[str] = set()
    rewritten: set[str] = set()

    def emit(patch: OverridePatch) -> None:
        if patch.path in written:
            rewritten.add(patch.path)
        written.add(patch.path)
        patches.append(patch)

    for tag in tags.add:
        if tag in current:
            reject_echo(f"{field}.add", tag, "the resolved parent")
        emit(OverridePatch(path=f"tags.{len(current)}", op="add", value=tag))
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
        emit(OverridePatch(path=f"tags.{index}", op="remove"))
        current.pop(index)
    return patches, sorted(rewritten)


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


#: Top-level matrix fields a derived training document never inherits.
#:
#: ``issue`` names the work the *parent* was authored for; carrying it forward
#: would state, in the compiled artifact, that this matrix belongs to a ticket it
#: has nothing to do with. The envelope's own issue is recorded in the compile
#: lock, which is where provenance lives. ``metadata`` is opaque: Feedbax cannot
#: tell an inherited launch set or orchestration root from a still-true fact, so
#: it inherits none of it rather than propagating something obsolete.
TRAINING_UNINHERITED_TOP_LEVEL_FIELDS: tuple[str, ...] = ("issue", "metadata")


def _lower_training(context: LayerCompileContext) -> LoweredLayer:
    """Lower authored rows, tags, and checkpoint sources over a run matrix."""
    authored = context.envelope.content
    assert isinstance(authored, TrainingLayerAuthoring)
    parent = dict(context.parent.pinned.document)
    inherited_rows = list(parent.get("rows") or [])
    by_id = {
        str(row.get("row_id")): row for row in inherited_rows if isinstance(row, Mapping)
    }
    # The row keys the *parent document* declares, fixed before any authored row
    # joins `by_id`. A row derived from one of these is derived from the pinned
    # parent, and says so in the lock; a row derived from a row this same envelope
    # authors was resolved inside this compile, where the parent pin names
    # nothing, and its ancestry is the authored chain the lock already records
    # through that earlier row.
    inherited_row_keys = frozenset(by_id)
    base_payload, base_pin = _resolve_matrix_base_payload(
        parent.get("base"), context.repo_root, field=f"{context.parent.ref}#base"
    )
    appending = authored.rows_mode is TrainingRowsMode.APPEND

    references: list[Any] = [] if base_pin is None else [base_pin]
    patches: list[OverridePatch] = []
    if context.envelope.name != parent.get("name"):
        patches.append(
            OverridePatch(path="name", op="replace", value=context.envelope.name)
        )
    else:
        check_echo(context.lineage, "name", context.envelope.name, field="envelope.name")

    for name in TRAINING_UNINHERITED_TOP_LEVEL_FIELDS:
        if parent.get(name):
            patches.append(OverridePatch(path=name, op="remove"))

    rows = list(inherited_rows) if appending else []
    authored_rows: list[dict[str, Any]] = []
    row_provenance: list[Any] = []
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
        new_row["label"] = row.effective_label
        # A changed row inherits none of its source's opaque metadata: the source
        # states facts about the experiment it was, and this row is a different
        # one. Authored replacement provenance is the one thing recorded here,
        # exactly as the envelope states it.
        new_row.pop("metadata", None)
        if row.seed is not None:
            if scalar_equal(source.get("seed"), row.seed):
                reject_echo(f"{field}.seed", row.seed, context.parent.ref)
            new_row["seed"] = row.seed
        if row.replaces is not None:
            new_row["metadata"] = {
                "replaces": row.replaces.model_dump(mode="json", exclude_none=True)
            }
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
        if appending:
            patches.append(OverridePatch(path=f"rows.{len(rows)}", op="add", value=new_row))
            rows.append(new_row)
        if row.from_ in inherited_row_keys:
            row_provenance.append(
                RowProvenanceReference(
                    row_id=row.id,
                    source_row_key=row.from_,
                    source_ref=context.parent.ref,
                    source_content_hash=context.parent.pinned.content_hash,
                )
            )
        authored_rows.append(new_row)
        by_id[row.id] = new_row

    if not appending:
        rows = authored_rows
        patches.append(OverridePatch(path="rows", op="replace", value=authored_rows))

    tag_patches, tag_acknowledgements = _tag_patches(
        parent.get("tags") or [], authored.tags, field="training.tags"
    )
    patches.extend(tag_patches)

    runnable = {str(row.get("row_id")) for row in rows}
    for index, item in enumerate(authored.checkpoint_initialization):
        if item.row not in runnable:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY,
                f"training.checkpoint_initialization[{index}].row",
                f"{item.row!r} is not a row this matrix runs; "
                f"rows: {sorted(runnable)}",
                correct_home="checkpoint initialization applies to a row the compiled "
                "matrix declares; under rows_mode 'authored_only' only the authored rows "
                "survive",
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
        deltas=_one_delta(context, patches, acknowledges=tag_acknowledgements),
        references=references,
        row_provenance=row_provenance,
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


#: The versioned structural rule under which a ``per_row`` figure input role
#: carries no single runtime locator. The role is not unbound: row expansion
#: fills it once per expanded row from the row index's own custody, and the
#: per-row profile and its closed artifact contract are recorded in the lock's
#: ``figure_row_expansion`` identity contribution. What is *not* applicable is
#: the single-locator reference slot, and stating that is different from leaving
#: the role silent.
PER_ROW_INPUT_RULE_ID = "feedbax.experiment_envelope.per_row_figure_input.v1"
PER_ROW_INPUT_REASON = (
    "row expansion fills this role once per expanded row from the row index's custody, "
    "so no single locator addresses it; the per-row profile and its artifact contract "
    "are recorded in the figure_row_expansion identity contribution"
)


def _lower_figure(context: LayerCompileContext) -> LoweredLayer:
    """Lower one figure operation, dispatched on the mode the envelope states."""
    authored = context.envelope.content
    assert isinstance(authored, FigureLayerAuthoring)
    if context.parent.contract is not FIGURE_OUTPUT:
        _reject(
            ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE,
            "figure.mode",
            f"a {authored.mode.value!r} figure envelope resolves {context.parent.ref}, which "
            f"is a {context.parent.contract.family!r} document; both figure modes take a "
            f"{FIGURE_OUTPUT.schema_id!r} parent",
            correct_home="the single-row scientific statement is the parent of both a "
            "row_expansion and a composition figure; a composition document is this "
            "engine's own output shape, not an experiment parent",
        )
    references: list[Any] = []
    for index, item in enumerate(authored.inputs):
        role_path = f"inputs.{item.input_role}"
        if item.is_per_row:
            # The dialect refuses a ref on a per-row role, so this rule is the
            # only way a per-row role reaches the lock: the single-locator slot
            # is stated not-applicable, never filled with one row's locator.
            assert item.ref is None  # guaranteed by FigureInputAuthoring
            references.append(
                NotApplicableReference(
                    role_path=role_path,
                    basis="compiler_rule",
                    reason=PER_ROW_INPUT_REASON,
                    rule_id=PER_ROW_INPUT_RULE_ID,
                )
            )
            continue
        references.append(
            _reference_for(
                context,
                item.ref,
                role_path=role_path,
                field=f"figure.inputs[{index}].ref",
                consumer_of=lambda _kind, _id, item=item: FigureRuntimeInputBinding(
                    input_role=item.input_role
                ),
            )
        )
    if authored.mode is FigureLayerMode.ROW_EXPANSION:
        return _lower_figure_row_expansion(context, authored, references)
    return _lower_figure_composition(context, authored, references)


def _lower_figure_row_expansion(
    context: LayerCompileContext,
    authored: FigureLayerAuthoring,
    references: list[Any],
) -> LoweredLayer:
    """Expand the parent figure over the row set the envelope's selector names.

    The expansion is Feedbax's existing derivation, wired to the dialect rather
    than reimplemented: the selector is expanded once against the pinned row
    index, and the structural half of the expansion — which names no produced
    data — becomes the compiled document. Custody stays where it belongs: the
    per-row and shared profiles are recorded in the lock, and the runtime inputs
    are bound from it at fulfillment.
    """
    assert authored.rows is not None  # guaranteed by the dialect model
    index, index_pin = _resolve_row_index(
        str(authored.rows.index), context.repo_root, field="figure.rows.index"
    )
    try:
        resolved_rows = expand_row_selector(authored.rows, index)
    except RowSelectionError as exc:
        _reject(
            _ROW_SELECTION_REJECTIONS.get(
                exc.code, ExperimentEnvelopeRejectionCategory.INVALID_VALUE
            ),
            "figure.rows",
            f"the authored row selector does not resolve against {authored.rows.index!r}: "
            f"{exc}",
            correct_home="a row selector names rows the index declares; a slice the index "
            "does not carry belongs in the index",
        )
    request = _figure_row_expansion_request(context, authored)
    try:
        document = expand_figure_rows_structure(
            context.parent.pinned.document, request, resolved_rows
        )
    except (ValueError, KeyError, TypeError) as exc:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "figure.mode",
            f"{context.parent.ref} cannot be expanded per row: {exc}",
            correct_home="the base figure is the single-row scientific statement the "
            "expansion repeats; express it through panels and trace families",
        )
    return LoweredLayer(
        contract=FIGURE_OUTPUT,
        deltas=(),
        document=document,
        references=[*references, index_pin],
        identity_contributions={
            "figure_row_expansion": request.model_dump(mode="json", exclude_none=True),
            "resolved_row_set": resolved_rows.model_dump(mode="json", exclude_none=True),
        },
    )


def _figure_row_expansion_request(
    context: LayerCompileContext, authored: FigureLayerAuthoring
) -> FigureRowExpansionRequest:
    """Build the closed expansion request the envelope's figure layer states."""
    try:
        return FigureRowExpansionRequest(
            figure_name=context.envelope.name,
            rows=authored.rows,  # type: ignore[arg-type]
            inputs={item.input_role: item.role_reference() for item in authored.inputs},
            role_contracts=[
                FigureRoleBindingContract.model_validate(
                    item.contract.binding_contract(item.input_role)
                )
                for item in authored.inputs
                if item.contract is not None
            ],
            assembler_title=authored.assembler_title,
        )
    except (ValueError, TypeError) as exc:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "figure.inputs",
            f"the authored figure inputs are not one expansion request: {exc}",
            correct_home="every input role states its per-row or shared profile and the "
            "closed artifact contract that profile must satisfy",
        )


def _resolve_row_index(
    ref: str, repo_root: Path, *, field: str
) -> tuple[AuthenticatedRowIndex, ContentPinReference]:
    """Load and pin the authenticated row index one selector names."""
    pinned = load_pinned(repo_root, ref)
    if pinned is None:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            field,
            f"{ref!r} is not a readable repo-relative JSON document",
        )
    try:
        index = AuthenticatedRowIndex.model_validate(dict(pinned.document))
    except (ValueError, TypeError) as exc:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            field,
            f"{ref!r} is not a valid {ROW_INDEX_SCHEMA_ID} document: {exc}",
            correct_home="a row selector is expanded against an authenticated row index; "
            "a row slice is expressed as one, not as a list inside the envelope",
        )
    return index, ContentPinReference(ref=ref, content_hash=pinned.content_hash)


def _lower_figure_composition(
    context: LayerCompileContext,
    authored: FigureLayerAuthoring,
    references: list[Any],
) -> LoweredLayer:
    """Compose the parent figure into a content-pinned composition document."""
    assert authored.delta is not None  # guaranteed by the dialect model
    if context.parent.kind != "frozen_document":
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            "envelope.base",
            f"a composition figure content-pins its parent by repo-relative path, and "
            f"{context.parent.ref!r} is an {context.parent.kind!r} whose document this "
            "engine produces rather than tracks",
            correct_home="a composition figure names a tracked frozen figure document; "
            "chaining envelopes is what an alias base does, and it composes by delta "
            "rather than by pin",
        )
    document = {
        "schema_id": FIGURE_COMPOSITION_OUTPUT.schema_id,
        "schema_version": FIGURE_COMPOSITION_OUTPUT.schema_version,
        "parent": {
            "ref": context.parent.ref,
            "sha256": context.parent.pinned.content_hash,
        },
        "deltas": [authored.delta.model_dump(mode="json", exclude_none=True)],
    }
    return LoweredLayer(
        contract=FIGURE_COMPOSITION_OUTPUT,
        deltas=(),
        document=document,
        references=references,
    )


#: The report types whose ``params`` the report layer contract itself knows well
#: enough to reconcile binding state inside. A type absent from this table
#: carries params whose owner is the recipe that registered it, and nothing here
#: touches them; the table is the same Feedbax-owned identity
#: :data:`~feedbax.contracts.experiment_envelope_dialect.REPORT_PARAMS_MODELS`
#: validates against, because knowing a field's meaning and validating it are the
#: same knowledge.
REPORT_BINDING_STATE_REPORT_TYPES: frozenset[str] = frozenset(REPORT_PARAMS_MODELS)

#: The fields an ordered-figure report node uses to *describe* the state of the
#: role it stands for. The set is closed and enumerated rather than swept for:
#: a heuristic over field names would reconcile authored science the moment a
#: project chose a similar-looking word, and these four are Feedbax's own
#: (:mod:`feedbax.analysis.reports`).
REPORT_APPLICABILITY_FIELD = "applicability"
REPORT_FIGURE_DIGEST_FIELD = "figure_spec_sha256"
REPORT_INPUT_ROLE_FIELD = "input_role"
REPORT_NOT_APPLICABLE_REASON_FIELD = "not_applicable_reason"
REPORT_BINDING_STATE_FIELDS: tuple[str, ...] = (
    REPORT_APPLICABILITY_FIELD,
    REPORT_FIGURE_DIGEST_FIELD,
    REPORT_INPUT_ROLE_FIELD,
    REPORT_NOT_APPLICABLE_REASON_FIELD,
)

#: The applicability value a role bound to authored not-applicability carries.
REPORT_NOT_APPLICABLE_VALUE = "not_applicable"

#: Where a report document carries the content its ``report_type`` names. The
#: params model is the authority for role paths inside this block and for
#: nothing outside it, which is why the walk below starts here rather than at the
#: report document's own root.
REPORT_PARAMS_FIELD = "params"

#: The array a not-applicable ordered-figure section must not carry, because a
#: node the envelope declares not applicable claims no content.
REPORT_SECTION_FIGURES_FIELD = "figures"

#: For each params node the engine reconciles a not-applicable binding at, the
#: descriptors that node must not carry once it is not applicable. The table is
#: keyed by the params model class the role path resolves to — the node's *type*,
#: not the fields the inherited bytes happen to carry — because which descriptors
#: a not-applicable node must shed is a fact about the node type, and a base that
#: states none of them is exactly the case that most needs reconciling. Keys are
#: ``(module, attribute)`` pairs for the same reason the dialect's output table
#: is: naming a model must not import the analysis stack.
#:
#: A params node absent from this table is one the ordered-figure contract gives
#: no applicability to (a scalar table, a projection): the binding stands in the
#: lock and the document says nothing, as before.
REPORT_NOT_APPLICABLE_REMOVALS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("feedbax.analysis.reports", "OrderedFigureReportSection"): (
        REPORT_SECTION_FIGURES_FIELD,
    ),
    ("feedbax.analysis.reports", "OrderedFigureReportFigure"): (
        REPORT_FIGURE_DIGEST_FIELD,
        REPORT_INPUT_ROLE_FIELD,
    ),
}


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
        deltas=_one_delta(
            context, _binding_state_patches(context, authored, references), authored.delta
        ),
        references=references,
    )


def _resolve_role_node(document: Mapping[str, Any], role_path: str) -> Any:
    """Return what the inherited document carries at one dotted role path.

    ``None`` means the document carries nothing there, which is the ordinary case
    and not an error: a report base commonly declares an empty slot list and the
    binding is the only statement about the role.
    """
    node: Any = document
    for part in role_path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return None
            node = node[part]
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            if not part.isdigit() or int(part) >= len(node):
                return None
            node = node[int(part)]
        else:
            return None
    return node


def _annotation_candidates(annotation: Any) -> tuple[Any, ...]:
    """Return the non-``None`` types one annotation may be, unions flattened."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _annotation_candidates(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        return tuple(
            candidate
            for argument in get_args(annotation)
            for candidate in _annotation_candidates(argument)
            if candidate is not type(None)
        )
    return (annotation,)


def _descend_annotation(annotation: Any, part: str) -> Any:
    """Return the annotation one role-path segment names, or ``None`` if unknown.

    A model field is named; a list element is indexed; a mapping value is keyed.
    A union descends into each member and answers only when the members agree,
    because a path whose meaning depends on which member the bytes happen to be
    is not a path this compiler knows the type of.
    """
    resolved: list[Any] = []
    for candidate in _annotation_candidates(annotation):
        origin = get_origin(candidate)
        arguments = get_args(candidate)
        if origin is None:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                declared = candidate.model_fields.get(part)
                found = None if declared is None else declared.annotation
            else:
                found = None
        elif isinstance(origin, type) and issubclass(origin, Mapping):
            found = arguments[1] if len(arguments) == 2 else None
        elif isinstance(origin, type) and issubclass(origin, Sequence) and arguments:
            found = arguments[0] if part.isdigit() else None
        else:
            found = None
        if found is not None and found not in resolved:
            resolved.append(found)
    return resolved[0] if len(resolved) == 1 else None


def _role_node_model(params_model: Any, role_path: str, *, field: str) -> Any:
    """Return the params-model type one role path resolves to.

    ``None`` means the params model is not the authority for this path: either
    the report type is one whose params Feedbax does not own, or the path
    addresses the report document outside its ``params`` block, where the report
    spec itself is the authority and states no binding-state descriptors.

    A path *inside* an owned params block that the model cannot resolve is
    refused rather than skipped: silently ignoring it would leave a binding whose
    role names nothing, which is the one outcome the reconciliation exists to
    prevent.
    """
    parts = role_path.split(".")
    if params_model is None or parts[0] != REPORT_PARAMS_FIELD:
        return None
    node_model: Any = params_model
    for part in parts[1:]:
        node_model = _descend_annotation(node_model, part)
        if node_model is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{field}.role_path",
                f"{role_path!r} names nothing in {params_model.__name__}, which is the "
                "closed content model this report type declares; the compiler cannot say "
                "what kind of role this binding decides the state of",
                correct_home="a report role path addresses a node of the content model its "
                "report_type names; the base states that structure and the binding names a "
                "place in it",
            )
    return node_model


def _not_applicable_removals(node_model: Any) -> tuple[str, ...] | None:
    """Return what a not-applicable node of this type sheds, or ``None``."""
    if not (isinstance(node_model, type) and issubclass(node_model, BaseModel)):
        return None
    return REPORT_NOT_APPLICABLE_REMOVALS.get((node_model.__module__, node_model.__name__))


def _binding_state_patches(
    context: LayerCompileContext,
    authored: ReportLayerAuthoring,
    references: Sequence[Any],
) -> list[OverridePatch]:
    """Return the derived patches that make the document say what the lock says.

    A report binding decides the *state* of one role. When the inherited document
    also describes that state — it says the role is included, or names the figure
    spec digest that fills it — the two can disagree, and a compiled document that
    claims a role is included while its lock records not-applicability is
    misinformation of exactly the species the parent's ``issue`` was.

    These patches are engine derivations, not authored content: every value comes
    from the binding the envelope already stated and from the reference it lowered
    to, so there is nothing here for an author to spell a second way. The
    reconciliation runs only inside a report type Feedbax owns the params of, and
    only over the closed set of fields that *describe* binding state; authored
    science at the same node is untouched.

    What a role's state *is* comes from the node's type in the params model, not
    from which descriptors the inherited bytes carry. A section the envelope
    declares not applicable is the case that makes the difference: its base node
    commonly states none of the descriptor fields and only an inherited
    ``figures`` array, and reading state off the bytes present would leave that
    array standing under a lock that says the section has no content.
    """
    document = context.parent.pinned.document
    if str(document.get("report_type")) not in REPORT_BINDING_STATE_REPORT_TYPES:
        return []
    params_model = REPORT_OUTPUT.params_model(document)
    patches: list[OverridePatch] = []
    for index, (binding, reference) in enumerate(zip(authored.bindings, references)):
        field = f"report.bindings[{index}]"
        node_model = _role_node_model(params_model, binding.role_path, field=field)
        node = _resolve_role_node(document, binding.role_path)
        if not isinstance(node, Mapping):
            continue
        if isinstance(reference, NotApplicableReference):
            removals = _not_applicable_removals(node_model)
            if removals is None:
                continue
            patches.extend(
                _not_applicable_state_patches(binding.role_path, node, reference, removals)
            )
        else:
            patches.extend(
                _bound_state_patches(binding.role_path, node, reference, field=field)
            )
    return patches


def _state_patch(role_path: str, node: Mapping[str, Any], name: str, value: Any) -> OverridePatch:
    """Return the patch that makes one descriptor field state ``value``."""
    return OverridePatch(
        path=f"{role_path}.{name}", op="replace" if name in node else "add", value=value
    )


def _not_applicable_state_patches(
    role_path: str,
    node: Mapping[str, Any],
    reference: NotApplicableReference,
    removals: Sequence[str],
) -> list[OverridePatch]:
    """Reconcile one role the envelope declares not applicable.

    The document is made to state the bound state rather than to forget it:
    dropping the ``applicability`` descriptor would leave the node claiming the
    contract's default, which is inclusion, so the contradiction would survive
    the removal. The reason is the one the envelope authored, which is also the
    one the lock's :class:`NotApplicableReference` records — one authored string,
    two places that quote it.

    ``removals`` is what this node *type* must not carry once it is not
    applicable: for a figure entry, the descriptors naming an artifact it does not
    have; for a section, the ``figures`` array, because a not-applicable section
    claims no content. A field the base does not carry yields no patch, so
    recompiling an already-reconciled base derives nothing.
    """
    patches: list[OverridePatch] = []
    for name, value in (
        (REPORT_APPLICABILITY_FIELD, REPORT_NOT_APPLICABLE_VALUE),
        (REPORT_NOT_APPLICABLE_REASON_FIELD, reference.reason),
    ):
        if name in node and scalar_equal(node[name], value):
            continue
        patches.append(_state_patch(role_path, node, name, value))
    for name in removals:
        if name in node:
            patches.append(OverridePatch(path=f"{role_path}.{name}", op="remove"))
    return patches


def _bound_state_patches(
    role_path: str, node: Mapping[str, Any], reference: Any, *, field: str
) -> list[OverridePatch]:
    """Reconcile one role the envelope binds to a product.

    The inherited ``figure_spec_sha256`` authenticates whatever figure the base
    was authored against, and this envelope has bound the role to a different one.
    When the binding is a planned product that compiles to a figure spec, the
    digest is replaced by that document's content hash: both are the canonical
    hash of the same figure spec bytes, computed in the one hash domain
    (:data:`~feedbax.contracts.authored_canonical.CANONICAL_PIN_ALGORITHM`), so
    this quotes the compile's own product rather than authoring a post-run fact.
    Every other binding is refused: a receipt names a manifest, not a figure spec,
    and clearing the digest alone would leave an included figure the report
    contract cannot validate, so there is no honest reconciliation to derive.
    """
    if node.get(REPORT_APPLICABILITY_FIELD) == REPORT_NOT_APPLICABLE_VALUE:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            f"{field}.role_path",
            f"{role_path!r} is inherited as {REPORT_NOT_APPLICABLE_VALUE!r} while this "
            "envelope binds it to a product; the compiler cannot derive the input role and "
            "figure digest an included role states",
            correct_home="bind a role the base leaves included, or state this binding as "
            "not_applicable and leave the base's statement standing",
        )
    if REPORT_FIGURE_DIGEST_FIELD not in node:
        return []
    if (
        isinstance(reference, PlannedProductReference)
        and reference.product_schema_id == FIGURE_OUTPUT.schema_id
    ):
        if scalar_equal(node[REPORT_FIGURE_DIGEST_FIELD], reference.compiled_content_hash):
            return []
        return [
            OverridePatch(
                path=f"{role_path}.{REPORT_FIGURE_DIGEST_FIELD}",
                op="replace",
                value=reference.compiled_content_hash,
            )
        ]
    _reject(
        ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
        f"{field}.ref",
        f"{role_path!r} inherits a {REPORT_FIGURE_DIGEST_FIELD} authenticating a different "
        "figure spec, and this binding names no compiled figure whose digest could replace "
        "it; a compile may quote a digest something else produced and may never author one",
        correct_home=f"bind the role to the envelope that compiles the figure, or remove the "
        f"inherited {REPORT_FIGURE_DIGEST_FIELD} from the base this envelope changes",
    )


def _one_delta(
    context: LayerCompileContext,
    generated: Sequence[OverridePatch],
    authored: MatrixCompositionDelta | None = None,
    *,
    acknowledges: Sequence[str] = (),
) -> list[MatrixCompositionDelta]:
    """Return the ordered composition layers one envelope contributes.

    The generated layer comes first and the authored native delta second, so an
    authored patch that overwrites something the envelope's own structured fields
    decided has to acknowledge it exactly as it would an ancestor's.

    ``acknowledges`` names the paths the *generated* layer knowingly rewrites
    within itself, which only a derivation that realizes one authored statement
    as several ordered patches produces. It is never widened by anything
    authored: the lowerer computes it beside the patches it computed.
    """
    deltas: list[MatrixCompositionDelta] = []
    if generated:
        deltas.append(
            MatrixCompositionDelta(
                layer_id=f"{context.envelope.name}.{context.layer.value}",
                patches=list(generated),
                acknowledges_ancestor_paths=list(acknowledges),
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
        if not path_is_within(base, self.layout.output_directory):
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
            build_lineage(repo_root, pinned, source_refs=source_refs_of(repo_root, base)),
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
            build_lineage(
                repo_root, pinned, source_refs=source_refs_of(repo_root, alias_ref)
            ),
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
        if envelope.training is not None:
            enforce_row_budget(
                len(envelope.training.rows),
                budget,
                field=f"{envelope_ref}#training.rows",
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
                row_provenance=lowered.row_provenance,
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
        """Apply or take the lowered document and prove the result is valid."""
        if lowered.document is not None:
            document = deepcopy(dict(lowered.document))
        else:
            try:
                document, _attribution, _written = apply_composition_deltas(
                    deepcopy(dict(context.parent.pinned.document)), list(lowered.deltas)
                )
            except ValueError as exc:
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    context.envelope_ref,
                    f"the envelope's deltas do not apply to {context.parent.ref}: {exc}",
                    correct_home="a delta states what changes relative to the base it "
                    "inherits, and acknowledges every ancestor-written path it overwrites",
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
        self._validate_declared_params(context, lowered.contract, document)
        return document

    def _validate_declared_params(
        self,
        context: LayerCompileContext,
        contract: LayerOutputContract,
        document: Mapping[str, Any],
    ) -> None:
        """Validate an inner ``params`` block against the model its type names.

        A top-level document that delegates its authored content to ``params``
        would otherwise be validated only as far as ``dict[str, Any]``, and the
        family's real authored contract would never be checked at compile time at
        all. The content type is read from the document's own discriminator.
        """
        model = contract.params_model(document)
        if model is None:
            return
        try:
            model.model_validate(document.get("params"))
        except Exception as exc:  # noqa: BLE001 - any model failure is one rejection
            discriminator = str(contract.params_discriminator)
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                f"{context.envelope_ref}#params",
                f"the compiled document's params are not valid for "
                f"{discriminator}={document.get(discriminator)!r}: {exc}",
                correct_home="the params block carries this content type's authored "
                "contract; the base states it and the envelope's delta changes it",
            )

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
    "DOCUMENT_SOURCES_KEY",
    "PER_ROW_INPUT_REASON",
    "PER_ROW_INPUT_RULE_ID",
    "REPORT_BINDING_STATE_FIELDS",
    "REPORT_BINDING_STATE_REPORT_TYPES",
    "REPORT_NOT_APPLICABLE_REMOVALS",
    "TRAINING_UNINHERITED_TOP_LEVEL_FIELDS",
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
    "source_refs_of",
    "verify_assertions",
]
