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

import json
import posixpath
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import UnionType
from typing import Annotated, Any, NoReturn, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_sha256,
    emit_text,
)
from feedbax.contracts.applicability_rules import (
    PER_ROW_FIGURE_INPUT_RULE,
    certify_not_applicable,
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
    EXPERIMENT_ENVELOPE_SUFFIX,
    FIGURE_COMPOSITION_OUTPUT,
    FIGURE_OUTPUT,
    REPORT_OUTPUT,
    REPORT_PARAMS_MODELS,
    TRAINING_OUTPUT_V6,
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
    CompositionTrainingRootAuthoring,
    ROOT_TRAINING_AUTHORITY_SCHEMA_VERSION,
    RootTrainingAuthority,
    TrainingRunRootAuthoring,
    TrainingLayerAuthoring,
    TrainingRowsMode,
    compiler_contract_version_for_schema,
    output_contract_of_document,
    parse_experiment_envelope,
)
from feedbax.contracts.figure_roles import (
    FigureRoleBindingContract,
    FigureRowCustodyLocator,
    FigureRowExpansionRequest,
    _expand_figure_rows_structure,
    per_row_binding_keys,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import load_content_pinned_json_document
from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    CompositionNode,
    ResolvedOutputParent,
    authored_envelope_hash,
)
from feedbax.contracts.row_index import (
    ROW_INDEX_SCHEMA_ID,
    AuthenticatedRowIndex,
    ResolvedRowSet,
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
from feedbax.contracts.training import TrainingRunSpec
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
_ROW_SELECTION_REJECTIONS: Mapping[RowSelectionErrorCode, ExperimentEnvelopeRejectionCategory] = {
    RowSelectionErrorCode.EMPTY_SELECTION: (ExperimentEnvelopeRejectionCategory.EMPTY_SELECTION),
    RowSelectionErrorCode.UNRESOLVED_ROW_KEY: (
        ExperimentEnvelopeRejectionCategory.UNRESOLVED_ROW_KEY
    ),
    RowSelectionErrorCode.DUPLICATE_ROW_ID: (ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY),
    RowSelectionErrorCode.AMBIGUOUS_ROW_BINDING: (
        ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    ),
    RowSelectionErrorCode.INDEX_MISMATCH: (ExperimentEnvelopeRejectionCategory.INVALID_VALUE),
}


def _reject(
    category: ExperimentEnvelopeRejectionCategory,
    field: str,
    message: str,
    *,
    correct_home: str | None = None,
) -> NoReturn:
    raise ExperimentEnvelopeRejection(category, message, field=field, correct_home=correct_home)


#: What an alias is, said once, for every refusal that has to explain itself.
_ALIAS_CORRECT_HOME = (
    "an alias names another envelope by its path stem under the project's envelope "
    "directory — 'leaf' or 'study/leaf' — in canonical relative POSIX form"
)


class UncontainedEnvelopeAliasError(ExperimentEnvelopeRejection):
    """An alias does not name a path the envelope directory contains.

    An alias is a *path stem*, so it is exactly the place a directory traversal
    hides in. The rule is proved on the joined path rather than on the authored
    string, because the authored string is what a traversal is written to look
    innocent as.

    Nothing is normalized on the author's behalf. An alias that would have to be
    rewritten to become legal does not unambiguously name one envelope, and this
    engine refuses ambiguity rather than picking a reading of it.

    Attributes:
        alias: The authored alias, exactly as written.
        envelope_directory: The directory the alias had to stay inside.
        reason: Which part of the containment grammar it broke.
        resolved: The joined repo-relative path, when the alias got far enough
            to have one.
    """

    def __init__(
        self,
        alias: str,
        *,
        envelope_directory: str,
        reason: str,
        field: str,
        resolved: str | None = None,
    ) -> None:
        joined = "" if resolved is None else f", which joins to {resolved!r}"
        super().__init__(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            f"{alias!r} is not an envelope alias contained by {envelope_directory!r}: "
            f"{reason}{joined}",
            field=field,
            correct_home=_ALIAS_CORRECT_HOME,
        )
        self.alias = alias
        self.envelope_directory = envelope_directory
        self.reason = reason
        self.resolved = resolved


class DuplicateOutputAddressError(ExperimentEnvelopeRejection):
    """Two authored envelopes in one project compile to the same output address.

    An envelope's authored ``name`` — not its path — addresses both of its
    compiled outputs. Two envelopes stating one name therefore write over each
    other, and the loser is whichever compiled first: a silent loss of a
    compiled document that no per-envelope check can see, because neither
    envelope is wrong on its own. Directory namespacing makes basename
    collisions natural and so makes this collision reachable, which is why the
    corpus is checked as a corpus, before anything is written.

    Attributes:
        name: The authored name both envelopes state.
        envelope_refs: Every colliding envelope, repo-relative, in path order.
        output_path: The output address they share.
    """

    def __init__(
        self,
        name: str,
        envelope_refs: Iterable[str],
        output_path: str,
        *,
        field: str | None = None,
    ) -> None:
        refs = tuple(envelope_refs)
        super().__init__(
            ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
            f"{len(refs)} envelopes state the name {name!r}, so they compile to the one "
            f"output address {output_path!r}: {', '.join(refs)}",
            field=field if field is not None else f"{refs[0]}#name",
            correct_home="an envelope's name is the identity of its compiled outputs and is "
            "unique across the project; a subdirectory namespaces the alias that reaches an "
            "envelope, not the address its outputs are written to",
        )
        self.name = name
        self.envelope_refs = refs
        self.output_path = output_path


def _authored_output_name(path: Path) -> str | None:
    """Return the output-addressing name one authored envelope states, if any.

    This is deliberately a shallow read rather than a compile. An envelope whose
    bytes are unreadable, whose JSON is malformed, or which states no name is
    refused by its own compile with its own diagnostic naming it; claiming that
    failure here would report one broken envelope as a corpus-wide collision.
    """
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(document, Mapping):
        return None
    name = document.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return None


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

    def alias_ref(self, alias: str, *, field: str = "envelope.alias") -> str:
        """Return the repo-relative envelope path one alias names.

        The alias is a path stem under :attr:`envelope_directory`, so it is held
        to the grammar of a contained relative POSIX path before it is joined,
        and the joined path is then proved to lie inside that directory. Both
        halves are kept: the grammar names what the author must fix, and the
        containment proof is what actually holds, so a later loosening of the
        grammar cannot quietly reopen the escape.

        Args:
            alias: The authored alias, exactly as written.
            field: The authored position named in a refusal.

        Returns:
            The repo-relative path of the envelope the alias names.

        Raises:
            UncontainedEnvelopeAliasError: The alias is absolute, carries ``..``,
                a backslash, an empty or ``.`` segment, or any other noncanonical
                form, or joins to a path outside the envelope directory.
        """
        self._refuse_uncontained_alias(alias, field=field)
        ref = str(PurePosixPath(self.envelope_directory) / f"{alias}{self.envelope_suffix}")
        if posixpath.normpath(ref) != ref or not path_is_within(ref, self.envelope_directory):
            raise UncontainedEnvelopeAliasError(
                alias,
                envelope_directory=self.envelope_directory,
                reason="the path it names is not inside the envelope directory",
                field=field,
                resolved=ref,
            )
        return ref

    def _refuse_uncontained_alias(self, alias: str, *, field: str) -> None:
        """Refuse every alias form that is not a contained relative POSIX stem."""

        def refuse(reason: str) -> NoReturn:
            raise UncontainedEnvelopeAliasError(
                alias,
                envelope_directory=self.envelope_directory,
                reason=reason,
                field=field,
            )

        if not isinstance(alias, str) or not alias:
            refuse("an alias is a nonempty string")
        if alias != alias.strip():
            refuse("it carries leading or trailing whitespace")
        if "\\" in alias:
            refuse("it carries a backslash, which is not a POSIX path separator")
        if alias.startswith("/") or PurePosixPath(alias).is_absolute():
            refuse("it is an absolute path rather than a stem under the envelope directory")
        segments = alias.split("/")
        if any(segment == "" for segment in segments):
            refuse("it carries an empty path segment")
        if any(segment in (".", "..") for segment in segments):
            refuse("it carries a '.' or '..' segment, which walks out of the envelope directory")
        if posixpath.normpath(alias) != alias:
            refuse("it is not in canonical form")


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
    parent: ResolvedParent | None
    repo_root: Path
    layout: EnvelopeLayout
    declaration: ProjectExperimentDeclaration
    compile_upstream: Callable[[str, str], "EnvelopeCompileOutcome"]

    @property
    def lineage(self) -> Lineage:
        """Return the parent's content-pinned lineage."""
        return Lineage(()) if self.parent is None else self.parent.lineage


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
    lock_deltas: Sequence[MatrixCompositionDelta] = ()

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
    """The two outputs of one compile: a document and its compile lock.

    ``envelope_schema`` is the dialect version the *authored document* declared,
    not the version this build calls current. A supported older version compiles
    as itself, so reporting the current constant would name a grammar the
    compiled envelope was never held to.
    """

    name: str
    family: str
    layer: ExperimentEnvelopeLayer
    document: Any
    compile_lock: dict[str, Any]
    envelope_schema: str


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
        return NotApplicableReference(role_path=role_path, basis="authored", reason=authored.reason)
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
        OverridePatch(path=f"{prefix}.{key}", op="add", value=params[key]) for key in sorted(params)
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


def _root_json_ref(repo_root: Path, ref: str, *, field: str) -> Path:
    """Resolve one canonical repository-relative JSON reference or refuse it."""
    authored = PurePosixPath(ref)
    if (
        not ref
        or authored.is_absolute()
        or authored.as_posix() != ref
        or any(part in ("", ".", "..") for part in authored.parts)
        or authored.suffix != ".json"
    ):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"{ref!r} is not a canonical repository-relative JSON reference",
        )
    root = repo_root.resolve()
    resolved = (root / Path(*authored.parts)).resolve()
    if not resolved.is_relative_to(root):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"{ref!r} escapes the repository root",
        )
    return resolved


def _load_root_document(repo_root: Path, ref: str, *, field: str) -> PinnedDocument:
    """Load one required root document after proving containment and JSON shape."""
    path = _root_json_ref(repo_root, ref, field=field)
    if not path.is_file():
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            field,
            f"{ref!r} does not name an existing repository JSON document",
        )
    pinned = load_pinned(repo_root, ref)
    if pinned is None:
        _reject(
            ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
            field,
            f"{ref!r} is not a readable JSON object",
        )
    return pinned


def _composition_root_pins(
    repo_root: Path, parent: AuthoredIntentParent, *, field: str
) -> list[ContentPinReference]:
    """Verify every authored composition parent pin without resolving output parents."""
    pins: list[ContentPinReference] = []
    seen: set[str] = set()
    current = parent
    while True:
        if current.ref in seen:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"authored composition reference cycle at {current.ref!r}",
            )
        seen.add(current.ref)
        pinned = _load_root_document(repo_root, current.ref, field=field)
        try:
            node = CompositionNode.model_validate(pinned.document)
        except ValidationError as exc:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                field,
                f"{current.ref!r} is not a feedbax.spec.training_run_composition.v1: {exc}",
            )
        actual = authored_envelope_hash(node)
        if actual != current.content_hash:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{current.ref!r} authored composition hash mismatch: "
                f"declared={current.content_hash!r}, computed={actual!r}",
            )
        pins.append(ContentPinReference(ref=current.ref, content_hash=pinned.content_hash))
        if not isinstance(node.parent, AuthoredIntentParent):
            return pins
        current = node.parent


def _root_source_pins(
    context: LayerCompileContext, sources: Sequence[Any]
) -> list[ContentPinReference]:
    """Pin every present source a root matrix declares, after containment checks."""
    document = {
        "sources": [source.model_dump(mode="json", exclude_none=True) for source in sources]
    }
    for index, source in enumerate(sources):
        _root_json_ref(
            context.repo_root,
            source.uri,
            field=f"training.root.sources[{index}].uri",
        )
    refs = source_refs_of(context.repo_root, context.envelope_ref)(document)
    return [
        ContentPinReference(
            ref=ref,
            content_hash=_load_root_document(
                context.repo_root, ref, field="training.root.sources"
            ).content_hash,
        )
        for ref in refs
    ]


def _load_root_training_authority(
    context: LayerCompileContext, root: Any
) -> tuple[RootTrainingAuthority | None, ContentPinReference | None]:
    """Verify one whole authority document, then validate its selected object."""
    authored = root.authority
    if authored is None:
        return None, None
    _root_json_ref(context.repo_root, authored.ref, field="training.root.authority.ref")
    try:
        _document, selected = load_content_pinned_json_document(
            authored,
            repo_root=context.repo_root,
        )
    except ValueError as exc:
        message = str(exc)
        if "payload_path" in message:
            category = (
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE
                if "must select a JSON object" in message
                else ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
            )
            field = "training.root.authority.payload_path"
        else:
            category = (
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE
                if "must contain a JSON object" in message
                else ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE
            )
            field = "training.root.authority"
        _reject(category, field, message)
    try:
        authority = RootTrainingAuthority.model_validate(selected)
    except ValidationError as exc:
        errors = exc.errors()
        kinds = {entry["type"] for entry in errors}
        if "extra_forbidden" in kinds:
            category = ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD
        elif kinds == {"missing"} and all(len(entry["loc"]) == 1 for entry in errors):
            category = ExperimentEnvelopeRejectionCategory.MISSING_FIELD
        else:
            category = ExperimentEnvelopeRejectionCategory.INVALID_VALUE
        _reject(
            category,
            "training.root.authority",
            f"selected payload is not a {ROOT_TRAINING_AUTHORITY_SCHEMA_VERSION}: {exc}",
        )
    return authority, ContentPinReference(ref=authored.ref, content_hash=authored.sha256)


def _combine_root_training_lists(
    authority: RootTrainingAuthority | None,
    root: Any,
) -> tuple[list[Any], list[Any]]:
    """Prepend authority entries and reject duplicates at their authored origin."""
    authority_sources = [] if authority is None else authority.sources
    authority_derivations = [] if authority is None else authority.derivations
    sources = [*authority_sources, *root.sources]
    derivations = [*authority_derivations, *root.derivations]

    seen_aliases: set[str] = set()
    for origin, entries in (
        ("training.root.authority.sources", authority_sources),
        ("training.root.sources", root.sources),
    ):
        for index, source in enumerate(entries):
            if source.alias in seen_aliases:
                _reject(
                    ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
                    f"{origin}[{index}].alias",
                    f"source alias {source.alias!r} is authored more than once",
                )
            seen_aliases.add(source.alias)

    seen_outputs: set[str] = set()
    for origin, entries in (
        ("training.root.authority.derivations", authority_derivations),
        ("training.root.derivations", root.derivations),
    ):
        for index, derivation in enumerate(entries):
            if derivation.output_path in seen_outputs:
                _reject(
                    ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
                    f"{origin}[{index}].output_path",
                    "derivation output_path "
                    f"{derivation.output_path!r} is authored more than once",
                )
            seen_outputs.add(derivation.output_path)
    return sources, derivations


def _root_payload_path(document: Mapping[str, Any], path: str | None, *, field: str) -> Any:
    """Resolve an optional explicit dotted payload path without fallback."""
    payload: Any = document
    if path is None:
        return payload
    if not path.strip() or any(not part for part in path.split(".")):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"payload_path {path!r} is not a dotted path",
        )
    for part in path.split("."):
        if not part or not isinstance(payload, Mapping) or part not in payload:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"payload_path {path!r} is not present in the pinned training run",
            )
        payload = payload[part]
    if not isinstance(payload, Mapping):
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            field,
            f"payload_path {path!r} does not resolve to a JSON object",
        )
    return payload


def _lower_root_training(
    context: LayerCompileContext, authored: TrainingLayerAuthoring
) -> LoweredLayer:
    """Construct a matrix-v6 root from one of the two closed root kinds."""
    root = authored.root
    assert root is not None
    references: list[Any] = []
    authority, authority_pin = _load_root_training_authority(context, root)
    sources, derivations = _combine_root_training_lists(authority, root)
    if isinstance(root, CompositionTrainingRootAuthoring):
        parent = root.parent
        if isinstance(parent, AuthoredIntentParent):
            _root_json_ref(context.repo_root, parent.ref, field="training.root.parent.ref")
            composition_pins = _composition_root_pins(
                context.repo_root, parent, field="training.root.parent"
            )
            references.extend(composition_pins)
            base: dict[str, Any] = {
                "kind": "authored_intent",
                "ref": parent.ref,
                "content_hash": composition_pins[0].content_hash,
                "pin_algorithm": CANONICAL_PIN_ALGORITHM,
            }
            if parent.symbolic_name is not None:
                base["symbolic_name"] = parent.symbolic_name
        else:
            assert isinstance(parent, ResolvedOutputParent)
            if (parent.row_id is None) != (parent.checkpoint_transaction_id is None):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    "training.root.parent",
                    "a resolved-output parent states row_id and checkpoint_transaction_id together",
                )
            base = {
                "kind": "resolved_output",
                "ref": parent.ref,
                "resolved_root_hash": parent.resolved_root_hash,
            }
            if parent.row_id is not None:
                base["row_id"] = parent.row_id
                base["checkpoint_transaction_id"] = parent.checkpoint_transaction_id
            if parent.symbolic_name is not None:
                base["symbolic_name"] = parent.symbolic_name
        matrix_deltas = list(root.deltas)
    else:
        assert isinstance(root, TrainingRunRootAuthoring)
        pinned = _load_root_document(context.repo_root, root.ref, field="training.root.ref")
        if pinned.content_hash != root.content_hash:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                "training.root.content_hash",
                f"{root.ref!r} content hash mismatch: declared={root.content_hash!r}, "
                f"computed={pinned.content_hash!r}",
            )
        try:
            TrainingRunSpec.model_validate(pinned.document)
        except ValidationError as exc:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                "training.root.ref",
                f"{root.ref!r} is not a feedbax.spec.training_run.v4: {exc}",
            )
        _root_payload_path(pinned.document, root.payload_path, field="training.root.payload_path")
        references.append(ContentPinReference(ref=root.ref, content_hash=pinned.content_hash))
        base = {
            "kind": "authored_intent",
            "ref": root.ref,
            "content_hash": root.content_hash,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }
        if root.payload_path is not None:
            base["payload_path"] = root.payload_path
        if root.symbolic_name is not None:
            base["symbolic_name"] = root.symbolic_name
        matrix_deltas = []

    if authority_pin is not None:
        references.append(authority_pin)
    references.extend(_root_source_pins(context, sources))
    rows: list[dict[str, Any]] = []
    for row in root.rows:
        compiled_row: dict[str, Any] = {
            "row_id": row.id,
            "label": row.effective_label,
            "overrides": [
                patch.model_dump(mode="json", exclude_none=True)
                for patch in (() if row.delta is None else row.delta.patches)
            ],
            "metadata": {},
        }
        if row.seed is not None:
            compiled_row["seed"] = row.seed
        rows.append(compiled_row)
    for index, item in enumerate(root.checkpoint_initialization):
        references.append(
            _reference_for(
                context,
                item.source,
                role_path=f"rows.{item.row}.checkpoint_initialization",
                field=f"training.root.checkpoint_initialization[{index}].source",
                consumer_of=lambda _kind, _id, item=item: CheckpointInitializationBinding(
                    mode=item.mode, row_id=item.row
                ),
            )
        )
    dependencies = [
        dependency.model_dump(mode="json", exclude_none=True)
        for dependency in root.execution_dependencies
    ]
    if isinstance(root, CompositionTrainingRootAuthoring) and root.selected_checkpoint is not None:
        parent = root.parent
        assert isinstance(parent, ResolvedOutputParent)
        assert parent.row_id is not None
        assert parent.checkpoint_transaction_id is not None
        dependencies.append(
            {
                "kind": "fork_from_selected_checkpoint",
                "source_authority": {
                    "kind": "resolved_output_root",
                    "source_run_id": root.selected_checkpoint.source_run_id,
                    "resolved_root_hash": parent.resolved_root_hash,
                },
                "source_row_id": parent.row_id,
                "checkpoint_transaction_id": parent.checkpoint_transaction_id,
                "checkpoint_root_hash": root.selected_checkpoint.checkpoint_root_hash,
                "source_barrier": root.selected_checkpoint.source_barrier,
                "slot_transforms": [
                    transform.model_dump(mode="json", exclude_none=True)
                    for transform in root.selected_checkpoint.slot_transforms
                ],
            }
        )
    document: dict[str, Any] = {
        "schema_id": TRAINING_OUTPUT_V6.schema_id,
        "schema_version": TRAINING_OUTPUT_V6.schema_version,
        "name": context.envelope.name,
        "base": base,
        "deltas": [delta.model_dump(mode="json", exclude_none=True) for delta in matrix_deltas],
        "execution_dependencies": dependencies,
        "sources": [source.model_dump(mode="json", exclude_none=True) for source in sources],
        "derivations": [
            derivation.model_dump(mode="json", exclude_none=True) for derivation in derivations
        ],
        "rows": rows,
        "axes": [],
        "combination": {
            "mode": "cross",
            "groups": [],
            "manual_coordinates": [],
            "metadata": {},
        },
        "tags": list(root.tags),
        "metadata": {},
    }
    if context.envelope.issue is not None:
        document["issue"] = context.envelope.issue
    if root.fork is not None:
        document["fork"] = root.fork.model_dump(mode="json", exclude_none=True)
    lock_deltas = [*matrix_deltas]
    lock_deltas.extend(row.delta for row in root.rows if row.delta is not None)
    return LoweredLayer(
        contract=TRAINING_OUTPUT_V6,
        deltas=(),
        document=document,
        references=references,
        identity_contributions={"training_root": root.model_dump(mode="json", exclude_none=True)},
        lock_deltas=lock_deltas,
    )


def _lower_training(context: LayerCompileContext) -> LoweredLayer:
    """Lower authored rows, tags, and checkpoint sources over a run matrix."""
    authored = context.envelope.content
    assert isinstance(authored, TrainingLayerAuthoring)
    if authored.root is not None:
        return _lower_root_training(context, authored)
    assert context.parent is not None
    parent = dict(context.parent.pinned.document)
    inherited_rows = list(parent.get("rows") or [])
    by_id = {str(row.get("row_id")): row for row in inherited_rows if isinstance(row, Mapping)}
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
        patches.append(OverridePatch(path="name", op="replace", value=context.envelope.name))
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
                f"{row.from_!r} names no row in {context.parent.ref}; rows: {sorted(by_id)}",
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
                f"{item.row!r} is not a row this matrix runs; rows: {sorted(runnable)}",
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
        contract=context.parent.contract,
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
    patches.extend(_params_patches(authored.params, "base.params"))
    references = [
        _reference_for(
            context,
            authored.subject,
            role_path=f"subjects.{authored.subject_id}",
            field="evaluation.subject",
            consumer_of=lambda _kind, _id: EvaluationSubjectBinding(subject_id=authored.subject_id),
        )
    ]
    # A further staged prerequisite is bound exactly as the subject is: by
    # binding name, into the same named-parent slot every materialized row
    # inherits. It is a second authenticated reference in the lock, which is the
    # only thing that can authenticate one — the compiled document states the
    # same names as a plan, and lowering refuses any it does not find here.
    for name, ref in (authored.prerequisites or {}).items():
        references.append(
            _reference_for(
                context,
                ref,
                role_path=f"subjects.{name}",
                field=f"evaluation.prerequisites.{name}",
                consumer_of=lambda _kind, _id, name=name: (
                    EvaluationSubjectBinding(subject_id=name)
                ),
            )
        )
    return LoweredLayer(
        contract=EVALUATION_OUTPUT,
        deltas=_one_delta(context, patches, authored.delta),
        references=references,
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
        check_echo(context.lineage, "analysis_type", authored.recipe, field="analysis.recipe")
        patches.append(OverridePatch(path="analysis_type", op="replace", value=authored.recipe))
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
    # A bundle's root set is the one thing about a bundle a predicate cannot
    # pin: the predicate re-selects whatever the manifest repository holds at
    # execution time. Authoring the roots puts each one in the lock as an
    # authenticated reference, and the adapter then binds by exact
    # manifest-identity set instead of by ambient selection.
    for index, root in enumerate(authored.roots or ()):
        references.append(
            _reference_for(
                context,
                root.ref,
                role_path=f"roots.{root.alias}",
                field=f"analysis.roots[{index}].ref",
                consumer_of=lambda _kind, _id, root=root: AnalysisInputBinding(
                    alias=root.alias, role=root.alias
                ),
            )
        )
    return LoweredLayer(
        contract=contract,
        deltas=_one_delta(context, patches, authored.delta),
        references=references,
    )


#: The versioned structural rule under which a ``per_row`` figure input role
#: carries no single runtime locator, and the reason it states. Both come from
#: the closed rule table in
#: :mod:`feedbax.contracts.applicability_rules`, which owns every structural
#: applicability rule this build can certify a decision under; restating them
#: here would let the compile's reason drift away from the rule it quotes.
PER_ROW_INPUT_RULE_ID = PER_ROW_FIGURE_INPUT_RULE.rule_id
PER_ROW_INPUT_REASON = PER_ROW_FIGURE_INPUT_RULE.reason


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
            references.append(certify_not_applicable(role_path, PER_ROW_FIGURE_INPUT_RULE))
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
            f"the authored row selector does not resolve against {authored.rows.index!r}: {exc}",
            correct_home="a row selector names rows the index declares; a slice the index "
            "does not carry belongs in the index",
        )
    request = _figure_row_expansion_request(context, authored)
    try:
        document = _expand_figure_rows_structure(
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
    contributions: dict[str, Any] = {
        "figure_row_expansion": request.model_dump(mode="json", exclude_none=True),
        "resolved_row_set": resolved_rows.model_dump(mode="json", exclude_none=True),
    }
    locator = _figure_row_custody_locator(authored, request, resolved_rows)
    if locator is not None:
        contributions["row_custody"] = locator.model_dump(mode="json", exclude_none=True)
    return LoweredLayer(
        contract=FIGURE_OUTPUT,
        deltas=(),
        document=document,
        references=[*references, index_pin],
        identity_contributions=contributions,
    )


def _figure_row_custody_locator(
    authored: FigureLayerAuthoring,
    request: FigureRowExpansionRequest,
    resolved_rows: ResolvedRowSet,
) -> FigureRowCustodyLocator | None:
    """Record where this figure's per-row custody bindings are to be found.

    The locator is a compile-time fact: it says which document fulfillment must
    read to fill the per-row roles, and which row index that document has to
    belong to. It pins the *index* digest, which the compile did read, and never
    the custody document's own bytes, which are post-run.

    There is nothing to record when the figure fills no role per row, and nothing
    to record when the envelope declares no custody: a compile states what the
    envelope said, so an undeclared locator is an absent contribution rather than
    one this function invents. A row-expansion envelope authored before the
    declaration existed therefore compiles to exactly the bytes it always did,
    and the refusal for its unfillable per-row roles lands at fulfillment, which
    is where a custody document is actually needed.
    """
    keys = per_row_binding_keys(request)
    if not keys or authored.row_custody is None:
        return None
    try:
        return FigureRowCustodyLocator(
            index_id=resolved_rows.index_id,
            index_sha256=resolved_rows.index_sha256,
            ref=authored.row_custody,
            binding_keys=list(keys),
        )
    except (ValueError, TypeError) as exc:
        _reject(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "figure.row_custody",
            f"the authored row custody declaration is not one locator: {exc}",
            correct_home="a row custody declaration is the repo-relative path of the "
            "custody bindings document the expanded rows are produced into",
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
    ("feedbax.analysis.reports", "OrderedFigureReportSection"): (REPORT_SECTION_FIGURES_FIELD,),
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
    derived, owners = _binding_state_patches(context, authored, references)
    _reject_authored_binding_overlap(authored.delta, owners, field="report.delta")
    return LoweredLayer(
        contract=REPORT_OUTPUT,
        deltas=_one_delta(context, derived, authored.delta),
        references=references,
    )


def _paths_overlap(left: str, right: str) -> bool:
    """Whether two dotted paths address the same node or one inside the other."""
    return left == right or left.startswith(f"{right}.") or right.startswith(f"{left}.")


def _reject_authored_binding_overlap(
    authored: MatrixCompositionDelta | None,
    owners: Mapping[str, str],
    *,
    field: str,
) -> None:
    """Refuse an authored patch that lands on state a binding decided.

    Binding-state reconciliation derives what the document says about a role
    from the binding the envelope already stated. An authored patch over that
    same node is a second authority for one fact, and the delta is applied after
    the derivation, so the authored value would simply win: the compiled document
    would state one thing and the compile lock another. There is no
    acknowledgement for this, because acknowledging it would only record that the
    author knew the two disagreed.

    ``acknowledges_ancestor_paths`` is untouched by this rule. It records paths an
    *ancestor* layer wrote that this envelope knowingly rewrites, which is a
    different relationship from two layers of the same compile claiming one fact.
    """
    if authored is None or not owners:
        return
    for index, patch in enumerate(authored.patches):
        for derived_path, role_path in owners.items():
            if _paths_overlap(patch.path, derived_path):
                _reject(
                    ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                    f"{field}.patches[{index}].path",
                    f"{patch.path!r} lands on {derived_path!r}, which this compile derived "
                    f"from the binding on {role_path!r}; the derivation and the patch are "
                    "two authorities for one fact, and the patch is applied last",
                    correct_home="a binding decides its role's state; state the role "
                    "differently in the binding, or patch a path no binding decides",
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
) -> tuple[list[OverridePatch], dict[str, str]]:
    """Return the derived patches, and which binding owns each path they write.

    The ownership map is what makes an authored delta over derived state a
    refusal rather than a silent overwrite: every derived path is attributable to
    the one binding whose state it states.

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
        return [], {}
    params_model = REPORT_OUTPUT.params_model(document)
    patches: list[OverridePatch] = []
    owners: dict[str, str] = {}
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
            derived = _not_applicable_state_patches(binding.role_path, node, reference, removals)
        else:
            derived = _bound_state_patches(binding.role_path, node, reference, field=field)
        patches.extend(derived)
        owners.update({patch.path: binding.role_path for patch in derived})
    return patches, owners


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
        alias_ref = self.layout.alias_ref(base, field=field)
        # The joined path is checked too, not only the authored string: a project
        # whose envelope directory sits under its output directory would
        # otherwise reach compiled output through an alias that looks contained.
        self.refuse_compiled_output_base(alias_ref, field)
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
            build_lineage(repo_root, pinned, source_refs=source_refs_of(repo_root, alias_ref)),
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
        compiler_contract = CompilerContract(
            contract_id=EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID,
            contract_version=compiler_contract_version_for_schema(envelope.schema_),
        )
        layer = envelope.layer
        budget = self.budgets.for_layer(layer.value)
        enforce_assertion_budget(len(envelope.assert_), budget, field=f"{envelope_ref}#assert")
        if envelope.training is not None:
            enforce_row_budget(
                len(
                    envelope.training.root.rows
                    if envelope.training.root is not None
                    else envelope.training.rows
                ),
                budget,
                field=f"{envelope_ref}#training.rows",
            )
        root_training = envelope.training is not None and envelope.training.root is not None
        if envelope.base is None and not root_training:
            _reject(
                ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
                f"{envelope_ref}#base",
                "an envelope states the document it changes; there is no rootless "
                "envelope, because a delta with nothing to apply to is a whole document",
                correct_home="state a frozen document by repo-relative path, or another "
                f"envelope in {self.layout.envelope_directory!r} by alias",
            )
        parent = (
            None
            if envelope.base is None
            else self.resolve_parent(repo_root, envelope.base, _stack, expected_layer=layer)
        )

        def compile_upstream(alias: str, field: str) -> EnvelopeCompileOutcome:
            upstream_ref = self.layout.alias_ref(alias, field=field)
            self.refuse_compiled_output_base(upstream_ref, field)
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
                contract=compiler_contract,
                implementation=self.implementation,
                base=None if parent is None else parent.lock_record(),
                lineage_pins=[] if parent is None else parent.lineage.pins(),
                resolved_deltas={
                    delta.layer_id: delta.model_dump(mode="json", exclude_none=True)
                    for delta in (lowered.lock_deltas or lowered.deltas)
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
            envelope_schema=envelope.schema_,
        )

    def _compose(self, context: LayerCompileContext, lowered: LoweredLayer) -> dict[str, Any]:
        """Apply or take the lowered document and prove the result is valid."""
        if lowered.document is not None:
            document = deepcopy(dict(lowered.document))
        else:
            assert context.parent is not None
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
        self._precheck_root_output(context)
        try:
            lowered.contract.model().model_validate(document)
        except ValidationError as exc:
            _reject(
                ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
                self._output_validation_field(context, exc),
                f"the compiled document is not a valid {lowered.contract.schema_id}: {exc}",
                correct_home="the compiled document must be a member of the Feedbax family "
                "its layer produces; the envelope's delta is what makes it one",
            )
        self._validate_declared_params(context, lowered.contract, document)
        return document

    @staticmethod
    def _precheck_root_output(context: LayerCompileContext) -> None:
        """Refuse root duplicates whose output-model validator has no field location."""
        training = context.envelope.training
        root = None if training is None else training.root
        if root is None:
            return
        seen_sources: set[str] = set()
        for index, source in enumerate(root.sources):
            if source.alias in seen_sources:
                _reject(
                    ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
                    f"training.root.sources[{index}].alias",
                    f"source alias {source.alias!r} is authored more than once",
                )
            seen_sources.add(source.alias)
        seen_outputs: set[str] = set()
        for index, derivation in enumerate(root.derivations):
            if derivation.output_path in seen_outputs:
                _reject(
                    ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY,
                    f"training.root.derivations[{index}].output_path",
                    f"derivation output_path {derivation.output_path!r} is authored more than once",
                )
            seen_outputs.add(derivation.output_path)

    @staticmethod
    def _output_validation_field(context: LayerCompileContext, error: ValidationError) -> str:
        """Map structured output-model locations back onto root authoring fields."""
        training = context.envelope.training
        root = None if training is None else training.root
        if root is None:
            return context.envelope_ref
        errors = error.errors()
        location = tuple(errors[0].get("loc", ())) if errors else ()
        if location and location[0] == "rows":
            index = location[1] if len(location) > 1 else 0
            return f"training.root.rows[{index}].id"
        if location and location[0] == "sources":
            index = location[1] if len(location) > 1 else 0
            return f"training.root.sources[{index}]"
        if location and location[0] == "derivations":
            index = location[1] if len(location) > 1 else 0
            return f"training.root.derivations[{index}]"
        if location and location[0] == "base":
            return (
                "training.root.parent"
                if isinstance(root, CompositionTrainingRootAuthoring)
                else "training.root.ref"
            )
        if not location and root.derivations and not root.sources:
            return "training.root.derivations"
        return "training.root"

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
        except ValidationError as exc:
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

    def compile_lock_path(self, name: str, out_dir: Path) -> Path:
        """Return the compile-lock address one authored name claims.

        This address is derived from the name alone, so it is the address two
        envelopes collide at whatever layers they author.
        """
        return out_dir / f"{name}.compile-lock.json"

    def output_paths(self, outcome: EnvelopeCompileOutcome, out_dir: Path) -> dict[str, Path]:
        """Return where one compile's two outputs belong, without writing them."""
        return {
            "compile_lock": self.compile_lock_path(outcome.name, out_dir),
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
        """Return every authored envelope in the project's envelope directory.

        The walk recurses, so a project may file its envelopes in subdirectories
        and still have the whole corpus enumerated: an alias is a path stem, so a
        subdirectory is a real part of an envelope's address rather than a place
        the engine stops looking. Order is the deterministic repo-relative POSIX
        path order, so two runs on one tree enumerate identically.

        Compiled output stays flat and is not affected: this is the enumeration
        of *authored* documents only.
        """
        directory = repo_root / self.layout.envelope_directory
        if not directory.is_dir():
            return []
        return sorted(
            directory.rglob(f"*{self.layout.envelope_suffix}"),
            key=lambda path: path.as_posix(),
        )

    def output_claims(self, repo_root: Path) -> dict[str, tuple[str, ...]]:
        """Return which envelopes claim each authored output name.

        Every envelope in the corpus is read shallowly for the one field that
        addresses its outputs. An envelope that states no readable name is
        omitted rather than guessed at: its own compile refuses it, and naming
        the same failure twice in two vocabularies helps nobody.
        """
        claims: dict[str, list[str]] = {}
        for path in self.envelopes(repo_root):
            name = _authored_output_name(path)
            if name is None:
                continue
            claims.setdefault(name, []).append(_repo_relative(path, repo_root))
        return {name: tuple(refs) for name, refs in claims.items()}

    def refuse_duplicate_output_addresses(
        self, repo_root: Path, *, out_dir: Path | None = None
    ) -> None:
        """Refuse a corpus in which two envelopes compile to one output address.

        This is a property of the corpus, not of any one envelope, so it is
        checked over the whole envelope directory and before anything is
        written. Compiling first and detecting the collision afterwards would
        already have overwritten one of the two compiled documents.

        Args:
            repo_root: Root the envelope directory resolves against.
            out_dir: Where compiled output is written, so the refusal can name
                the address that collides. Defaults to the layout's output
                directory under ``repo_root``.

        Raises:
            DuplicateOutputAddressError: Two or more envelopes state one name.
        """
        output_root = out_dir if out_dir is not None else repo_root / self.layout.output_directory
        for name, refs in sorted(self.output_claims(repo_root).items()):
            if len(refs) > 1:
                raise DuplicateOutputAddressError(
                    name,
                    refs,
                    _repo_relative(self.compile_lock_path(name, output_root), repo_root),
                )


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return *path* as a repo-relative POSIX string when it is one."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


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
    "DuplicateOutputAddressError",
    "EnvelopeCompileOutcome",
    "EnvelopeKernel",
    "EnvelopeLayout",
    "LayerCompileContext",
    "LoweredLayer",
    "ResolvedParent",
    "UncontainedEnvelopeAliasError",
    "authored_layer_of",
    "check_echo",
    "check_no_co_created_protected_document",
    "compiled_document_pin",
    "reject_echo",
    "scalar_equal",
    "source_refs_of",
    "verify_assertions",
]
