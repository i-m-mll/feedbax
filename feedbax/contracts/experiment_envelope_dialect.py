"""The one authored-experiment dialect Feedbax owns.

``feedbax.experiment_envelope.v1`` is the only envelope dialect there is. A
project does not define an envelope family, a layer, a lowerer, or a rule: it
authors documents in this dialect and Feedbax compiles them into the spec
families it already owns.

## The common fields and the one-layer rule

Every envelope states ``schema``, ``name``, and optionally ``base``, ``issue``,
``reason``, and ``assert``. It then states **exactly one** of ``training``,
``evaluation``, ``analysis``, ``figure``, or ``report``. One envelope, one
artifact: an envelope that authored two layers would compile to two documents
with one lock, one name, and one identity, and nothing downstream could say
which of the two a later reference meant.

The five layers are Feedbax's five existing artifact families, not a taxonomy
invented here. A sixth family gets a new dialect version, not an extension slot.

## Authored references are typed and closed

A layer names something outside itself in exactly three ways: an upstream
envelope by alias, a run receipt by manifest kind and id, or an explicit
statement that a role is not applicable. Those three lower onto the compile
lock's closed reference union without a translation table. An authored receipt
that carries a digest and a size lowers to an authenticated reference; one that
carries neither lowers to a locator, which is the honest record when the
receipt does not exist yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, ValidationError, model_validator

from feedbax.contracts.checkpoint_initialization import CheckpointInitializationMode
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.figure_roles import (
    FigureInputReference,
    PerRowInputReference,
    SharedInputReference,
)
from feedbax.contracts.figures import FigureCompositionDelta
from feedbax.contracts.manifest import StrictModel
from feedbax.contracts.row_index import RowSetSelector
from feedbax.contracts.run_matrix import MatrixCompositionDelta

#: The unversioned identity of the dialect family.
EXPERIMENT_ENVELOPE_FAMILY = "feedbax.experiment_envelope"

#: The one authored schema string an envelope declares.
EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1 = f"{EXPERIMENT_ENVELOPE_FAMILY}.v1"
EXPERIMENT_ENVELOPE_SCHEMA_VERSION = EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1

#: Enumerated, never inferred.
EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1,
)

#: Versions accepted and migrated. Empty at v1: nothing Feedbax-owned precedes it.
EXPERIMENT_ENVELOPE_MIGRATION_TABLE: dict[str, str] = {}

#: The compiler contract is global. There is no per-project contract indirection:
#: one dialect compiled by one compiler means one contract for every project.
EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID = f"{EXPERIMENT_ENVELOPE_FAMILY}.compiler"
EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION = (
    f"{EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID}.v1"
)

#: The suffix an authored envelope file carries.
EXPERIMENT_ENVELOPE_SUFFIX = ".envelope.json"


class ExperimentEnvelopeLayer(StrEnum):
    """The five artifact families an envelope may author, and no others."""

    TRAINING = "training"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    FIGURE = "figure"
    REPORT = "report"


class DialectModel(StrictModel):
    """Closed authored model: unknown keys are rejected, aliases are accepted."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# -- authored assertions ---------------------------------------------------


class EnvelopeAssertion(DialectModel):
    """One inherited precondition an envelope states before its delta applies."""

    path: str
    equals: Any = None

    @model_validator(mode="after")
    def _validate(self) -> "EnvelopeAssertion":
        if not self.path.strip() or any(not part for part in self.path.split(".")):
            raise ValueError(f"assert path is not a dotted path: {self.path!r}")
        if "equals" not in self.model_fields_set:
            raise ValueError("an assertion states the value it expects at 'equals'")
        return self


# -- authored references ---------------------------------------------------


class UpstreamEnvelopeReference(DialectModel):
    """Another envelope in this project, named by its alias."""

    kind: Literal["envelope"] = "envelope"
    alias: str

    @model_validator(mode="after")
    def _validate(self) -> "UpstreamEnvelopeReference":
        if not self.alias.strip():
            raise ValueError("an envelope reference names a nonempty alias")
        return self


class ReceiptReference(DialectModel):
    """A run receipt, named by manifest kind and id.

    The digest and size are stated together or not at all. Stating them makes
    this an authenticated quote of a receipt that exists; omitting them makes it
    a locator for one that does not exist yet. A half-stated profile would be a
    third thing that is neither, so it is refused.
    """

    kind: Literal["receipt"] = "receipt"
    manifest_kind: str
    manifest_id: str
    manifest_sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    execution_uri: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ReceiptReference":
        if not self.manifest_kind.strip() or not self.manifest_id.strip():
            raise ValueError("a receipt reference names a nonempty manifest kind and id")
        authenticated = self.manifest_sha256 is not None
        if authenticated != (self.size_bytes is not None):
            raise ValueError(
                "a receipt reference states both manifest_sha256 and size_bytes or neither"
            )
        if not authenticated and self.execution_uri is not None:
            raise ValueError(
                "a receipt locator has no execution uri; only a produced receipt does"
            )
        return self

    @property
    def is_authenticated(self) -> bool:
        """Return whether this reference quotes a receipt that already exists."""
        return self.manifest_sha256 is not None


class NotApplicableAuthoring(DialectModel):
    """An explicit statement that a role is deliberately unfilled."""

    kind: Literal["not_applicable"] = "not_applicable"
    reason: str

    @model_validator(mode="after")
    def _validate(self) -> "NotApplicableAuthoring":
        if not self.reason.strip():
            raise ValueError("authored not-applicability states why")
        return self


AuthoredReference: TypeAlias = Annotated[
    UpstreamEnvelopeReference | ReceiptReference | NotApplicableAuthoring,
    Field(discriminator="kind"),
]


# -- training --------------------------------------------------------------


class RowReplacement(DialectModel):
    """What a new row replaces, recorded as provenance rather than as a delete."""

    row: str
    seed: int | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RowReplacement":
        if not self.row.strip():
            raise ValueError("a replacement names the row it replaces")
        return self


class TrainingRowAuthoring(DialectModel):
    """One authored training row: inherit a row, name it, and change it.

    ``delta`` is a native :class:`MatrixCompositionDelta` authored directly. There
    is no shorthand layer above it: a shorthand would have to know what the
    patched paths mean, which is the project's science and not Feedbax's.

    ``label`` is the row's display name. It defaults to the row's own ``id`` and
    is never copied from the source row: a changed row that kept its source's
    label would advertise, in the one field a reader sees first, an experiment it
    is no longer running.
    """

    from_: str = Field(alias="from")
    id: str
    label: str | None = None
    seed: int | None = None
    replaces: RowReplacement | None = None
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TrainingRowAuthoring":
        if not self.from_.strip():
            raise ValueError("a training row inherits a nonempty parent row id")
        if not self.id.strip():
            raise ValueError("a training row states its own nonempty id")
        if self.label is not None and not self.label.strip():
            raise ValueError("a training row that states a label states a nonempty one")
        return self

    @property
    def effective_label(self) -> str:
        """Return the label this row carries: the authored one, or its own id."""
        return self.id if self.label is None else self.label


class TagsDelta(DialectModel):
    """Tags added and removed, as a delta rather than a restated whole."""

    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "TagsDelta":
        if not self.add and not self.remove:
            raise ValueError("a tags delta adds or removes something")
        overlap = sorted(set(self.add) & set(self.remove))
        if overlap:
            raise ValueError(f"tags both added and removed: {overlap}")
        for group in (self.add, self.remove):
            if len(set(group)) != len(group):
                raise ValueError("tags delta entries must be unique")
        return self


class CheckpointInitializationAuthoring(DialectModel):
    """One row initialized or continued from a stated source."""

    row: str
    mode: CheckpointInitializationMode
    source: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "CheckpointInitializationAuthoring":
        if not self.row.strip():
            raise ValueError("checkpoint initialization names the row it applies to")
        if isinstance(self.source, NotApplicableAuthoring):
            raise ValueError(
                "checkpoint initialization that is not applicable is simply not authored"
            )
        return self


class TrainingRowsMode(StrEnum):
    """What the compiled row set is, relative to the rows the parent declares.

    * ``authored_only`` — the compiled matrix runs exactly the rows this envelope
      authors. Every inherited row is dropped.
    * ``append`` — the inherited rows keep running and the authored rows are
      added after them.

    The distinction is not presentational. An inherited row that survives is
    runnable work: a compile that silently kept the parent's rows would launch
    training nobody in this envelope asked for. So the mode is stated, never
    defaulted and never inferred — an absent ``rows_mode`` is a missing field,
    not a request for the historical behavior.
    """

    AUTHORED_ONLY = "authored_only"
    APPEND = "append"


class TrainingLayerAuthoring(DialectModel):
    """Rows inherited from the parent matrix, plus tags and checkpoint sources.

    There is deliberately no top-level ``delta`` here. An arbitrary patch over the
    whole matrix would make every structured field on this model ornamental: the
    same change could always be written as a raw path, and the row/tag/checkpoint
    vocabulary would stop being the thing that decides what runs. A change this
    layer cannot express belongs in the base, not in a free-form patch.

    Checkpoint initialization is a contribution in its own right, not a decoration
    on an authored row. A fork that inherits every row and only attaches
    authenticated checkpoint sources to them changes what those runs *are* —
    initialized or continued from stated bytes rather than from scratch — so a
    layer whose sole contribution is that is a whole authored layer.
    """

    rows_mode: TrainingRowsMode
    rows: list[TrainingRowAuthoring] = Field(default_factory=list)
    tags: TagsDelta | None = None
    checkpoint_initialization: list[CheckpointInitializationAuthoring] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _validate(self) -> "TrainingLayerAuthoring":
        if not self.rows and self.tags is None and not self.checkpoint_initialization:
            raise ValueError(
                "a training layer authors rows, tags, checkpoint initialization, or a "
                "combination of them"
            )
        if self.rows_mode is TrainingRowsMode.AUTHORED_ONLY and not self.rows:
            raise ValueError(
                "rows_mode 'authored_only' states that the compiled matrix runs exactly "
                "the authored rows; authoring none would compile a matrix that runs nothing"
            )
        ids = [row.id for row in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("training row ids must be unique within one envelope")
        layer_ids = [row.delta.layer_id for row in self.rows if row.delta is not None]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("training row delta layer ids must be unique")
        initialized = [item.row for item in self.checkpoint_initialization]
        if len(set(initialized)) != len(initialized):
            raise ValueError("a row states at most one checkpoint initialization")
        return self


# -- evaluation ------------------------------------------------------------


class StagedPrerequisiteAuthoring(DialectModel):
    """One further named prerequisite every row of an evaluation inherits.

    An evaluation has one subject. It may still need other already-produced
    artifacts to run — a trial bank a paired controller is replayed against, a
    reference evaluation a contrast is measured from — and those are named staged
    prerequisites, addressed by binding name exactly as the subject is.

    The compiled matrix's ``staged_parents`` block is *a plan*: it cannot
    authenticate a parent, and lowering refuses a staged parent the compile lock
    does not bind. This is the authoring form that puts one in the lock.
    """

    name: str
    ref: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "StagedPrerequisiteAuthoring":
        if not self.name.strip():
            raise ValueError("a staged prerequisite states a nonempty binding name")
        if isinstance(self.ref, NotApplicableAuthoring):
            raise ValueError(
                "a staged prerequisite that is not applicable is simply not authored; a "
                "named prerequisite binding nothing would name an empty staged parent"
            )
        return self


class EvaluationLayerAuthoring(DialectModel):
    """The subject an evaluation evaluates, and the recipe parameters it runs.

    ``staged_prerequisites`` are the further named artifacts every row inherits
    alongside the subject. Without them an evaluation could author exactly one
    reference, which is why a document whose compiled base states a second staged
    parent had no way to authenticate it and refused at lowering.

    Absent and empty are distinct here, as everywhere else in this dialect. An
    absent list is an evaluation with one subject and nothing further; an empty
    one states a prerequisite block with no prerequisite in it, and is refused
    rather than treated as the absent case.
    """

    subject: AuthoredReference
    subject_id: str
    staged_prerequisites: list[StagedPrerequisiteAuthoring] | None = None
    recipe: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "EvaluationLayerAuthoring":
        if not self.subject_id.strip():
            raise ValueError("an evaluation names its subject id")
        if self.staged_prerequisites is None:
            return self
        if not self.staged_prerequisites:
            raise ValueError(
                "an evaluation that states staged prerequisites states at least one; omit "
                "'staged_prerequisites' entirely when the subject is the whole input"
            )
        names = [item.name for item in self.staged_prerequisites]
        if len(set(names)) != len(names):
            raise ValueError("staged prerequisite binding names must be unique")
        if self.subject_id in names:
            raise ValueError(
                f"staged prerequisite {self.subject_id!r} names the evaluation's own "
                "subject; one binding name addresses exactly one authenticated parent"
            )
        return self


# -- analysis --------------------------------------------------------------


class AnalysisSubjectAuthoring(DialectModel):
    """One analysis input, addressed by alias and role."""

    alias: str
    role: str
    ref: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "AnalysisSubjectAuthoring":
        if not self.alias.strip() or not self.role.strip():
            raise ValueError("an analysis subject states a nonempty alias and role")
        return self


class AnalysisBundleRootAuthoring(DialectModel):
    """One member of the exact root set a bundle executes over.

    A bundle addresses its roots by manifest identity rather than by role: its
    stages run over a *set* of receipts, and the alias is the author's name for
    one member of that set, not a slot inside a stage.
    """

    alias: str
    ref: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "AnalysisBundleRootAuthoring":
        if not self.alias.strip():
            raise ValueError("a bundle root states a nonempty alias")
        if isinstance(self.ref, NotApplicableAuthoring):
            raise ValueError(
                "a bundle root that is not applicable is simply not a member of the root "
                "set; an inapplicable root would name a receipt the bundle cannot run over"
            )
        return self


class AnalysisLayerAuthoring(DialectModel):
    """A run or a bundle, its typed subjects or roots, and its parameters.

    ``roots`` is a bundle's exact root set, and it is the only way to lock one:
    the alternative is the bundle's own authored predicate, which selects
    whatever the manifest repository holds when it runs, so a converted bundle
    would silently widen as that repository grows. Declaring roots makes the
    binding an exact manifest-identity set and the predicate a claim that must
    agree with it.

    An absent ``roots`` is the honest record of a bundle that genuinely selects
    ambiently; an empty ``roots`` is refused, because a declared root set with no
    members would describe a bundle that executes over nothing. 
    """

    target: Literal["run", "bundle"] = "run"
    subjects: list[AnalysisSubjectAuthoring] = Field(default_factory=list)
    roots: list[AnalysisBundleRootAuthoring] | None = None
    recipe: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "AnalysisLayerAuthoring":
        aliases = [subject.alias for subject in self.subjects]
        if len(set(aliases)) != len(aliases):
            raise ValueError("analysis subject aliases must be unique")
        if self.target == "bundle" and (self.subjects or self.recipe is not None):
            raise ValueError(
                "an analysis bundle states its templates as a delta; subjects and a "
                "recipe belong to a single analysis run"
            )
        if self.roots is not None:
            if self.target != "bundle":
                raise ValueError(
                    "only an analysis bundle states a root set; a single analysis run "
                    "states its inputs as typed subjects"
                )
            if not self.roots:
                raise ValueError(
                    "an analysis bundle that states a root set states at least one root; "
                    "omit 'roots' entirely to select roots by the bundle's own predicate"
                )
            root_aliases = [root.alias for root in self.roots]
            if len(set(root_aliases)) != len(root_aliases):
                raise ValueError("analysis bundle root aliases must be unique")
        return self


# -- figure ----------------------------------------------------------------


class FigureLayerMode(StrEnum):
    """The one operation a figure envelope performs, stated explicitly.

    * ``row_expansion`` — a :class:`~feedbax.contracts.figures.FigureSpec` parent
      is repeated over a resolved row set and compiles to a ``FigureSpec``. The
      derivation is normative and unauthored: row-index order alone decides
      namespaces, panel placement, titles, legend ownership, and height.
    * ``composition`` — a ``FigureSpec`` parent is content-pinned and compiles to
      a :class:`~feedbax.contracts.figures.FigureCompositionSpec` carrying the
      authored ordered deltas over it.

    The mode is authored because it is a semantic choice, and a semantic choice
    is never read off a filename, a family word, or the shape of the parent.
    """

    ROW_EXPANSION = "row_expansion"
    COMPOSITION = "composition"


class FigureRoleContractAuthoring(DialectModel):
    """The declared closed artifact contract for one row-expanded input role.

    These are the fields of
    :class:`~feedbax.contracts.figure_roles.FigureRoleBindingContract` an author
    decides; ``input_role`` is not among them because the enclosing input already
    states it, and ``manifest_status`` is not among them because only a completed
    manifest is ever bound.
    """

    kind: str = "AnalysisRunManifest"
    authority: Literal["artifact_provider", "analysis_data_product"] = "artifact_provider"
    artifact_role: str
    artifact_provider: str
    media_type: str = "application/json"
    payload_name: str | None = None
    product_role: str | None = None
    product_schema_id: str | None = None
    product_schema_version: str | None = None
    analysis_type: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FigureRoleContractAuthoring":
        for name in ("kind", "artifact_role", "artifact_provider", "media_type"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"a figure role contract states a nonempty {name}")
        return self

    def binding_contract(self, input_role: str) -> dict[str, Any]:
        """Return the ``FigureRoleBindingContract`` payload for *input_role*."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["input_role"] = input_role
        return payload


class FigureInputAuthoring(DialectModel):
    """One figure runtime input, addressed by its input role.

    ``ref`` is the runtime locator the compile lock records; it never enters the
    compiled document, because authenticating an input takes a run. Under
    ``row_expansion`` the input additionally states *how* the role is filled —
    once per expanded row from the row index's custody (``per_row``), or once for
    every row from one named manifest (``shared``) — and the closed artifact
    contract that fill must satisfy.

    A ``per_row`` role is the one case with no single locator to state: the
    expansion fills it once per expanded row from the row index's own custody, so
    any single ``ref`` would name one row's artifact and be false for every other
    row. ``ref`` is therefore *forbidden* exactly there — an authored locator on a
    per-row role is not a harmless extra fact, it is a false one, and the compile
    lock would carry it as though it addressed the role. A ``shared`` role is
    filled once for every row from one named manifest, so it states that manifest,
    and a figure input outside row expansion has nothing filling it at all unless
    it states one: both fail closed on an omitted ``ref``.
    """

    input_role: str
    ref: AuthoredReference | None = None
    binding: Literal["per_row", "shared"] | None = None
    binding_key: str | None = None
    contract: FigureRoleContractAuthoring | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FigureInputAuthoring":
        if not self.input_role.strip():
            raise ValueError("a figure input states a nonempty input role")
        expanded = (self.binding, self.binding_key, self.contract)
        if any(item is None for item in expanded) and any(
            item is not None for item in expanded
        ):
            raise ValueError(
                "a row-expanded figure input states binding, binding_key, and contract "
                "together; a partial profile is neither a per-row nor a shared role"
            )
        if self.binding_key is not None and not self.binding_key.strip():
            raise ValueError("a figure input states a nonempty binding key")
        if self.ref is None and self.binding != "per_row":
            raise ValueError(
                f"figure input {self.input_role!r} states no ref; only a 'per_row' "
                "row-expansion role omits it, because row expansion fills that role once "
                "per expanded row and no single locator addresses it"
            )
        if self.ref is not None and self.binding == "per_row":
            raise ValueError(
                f"figure input {self.input_role!r} is a 'per_row' role and states a ref; row "
                "expansion fills that role once per expanded row from the row index's own "
                "custody, so any single locator names one row's artifact and is false for "
                "every other row. Remove the ref: the per-row binding key and its artifact "
                "contract are the whole binding"
            )
        return self

    @property
    def is_per_row(self) -> bool:
        """Whether row expansion fills this role once per expanded row."""
        return self.binding == "per_row"

    @property
    def is_row_expanded(self) -> bool:
        """Whether this input declares the per-row/shared profile expansion needs."""
        return self.binding is not None

    def role_reference(self) -> FigureInputReference:
        """Return the closed ``per_row``/``shared`` reference this input declares."""
        if self.binding is None or self.binding_key is None:
            raise ValueError(
                f"figure input {self.input_role!r} declares no row-expansion binding"
            )
        if self.binding == "per_row":
            return PerRowInputReference(per_row=self.binding_key)
        return SharedInputReference(shared=self.binding_key)


class FigureLayerAuthoring(DialectModel):
    """One figure operation: its mode, its inputs, and what that mode needs.

    ``row_expansion`` states the row set it repeats the parent over and declares
    every input's per-row/shared profile. It authors no delta at all: the
    expansion is derived, and a patch layered over a derived document would make
    the derivation negotiable.

    When any role is filled per row, it may also state ``row_custody``: the
    repo-relative path of the
    :class:`~feedbax.contracts.row_index.RowIndexCustodyBindings` document the
    rows are produced into. That is a locator, not a production record — the
    document is written after the rows run, and this envelope compiles whether or
    not it exists yet — but it is stated rather than derived from the index id,
    because a naming convention is a rule nothing states.

    It is optional here, and its absence is caught where it actually bites. A
    row-expansion envelope that predates the declaration still compiles, to
    byte-identical output, because a compile that recorded a locator nobody
    authored would be inventing one. What such an envelope cannot do is be
    *fulfilled*: binding a per-row role needs a custody document to read, so
    :mod:`feedbax.analysis.fulfillment_row_custody` refuses the figure by name
    rather than rendering it with the role unfilled. Declaring ``row_custody``
    when no role is filled per row is still refused here, because a shared role
    names its own manifest and the declaration would address nothing.

    ``composition`` states the ordered deltas and nothing about rows.
    """

    mode: FigureLayerMode
    inputs: list[FigureInputAuthoring] = Field(default_factory=list)
    rows: RowSetSelector | None = None
    row_custody: str | None = None
    assembler_title: str | None = None
    delta: FigureCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FigureLayerAuthoring":
        roles = [item.input_role for item in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError("figure input roles must be unique")
        if self.mode is FigureLayerMode.ROW_EXPANSION:
            return self._validate_row_expansion()
        return self._validate_composition()

    def _validate_row_expansion(self) -> "FigureLayerAuthoring":
        if self.delta is not None:
            raise ValueError(
                "a row_expansion figure authors no delta; the expansion is derived from "
                "row-index order alone, and a patch over it would make that derivation "
                "negotiable"
            )
        if self.rows is None:
            raise ValueError("a row_expansion figure states the row set it expands over")
        if not self.rows.index or not self.rows.index.strip():
            raise ValueError(
                "a row_expansion figure's row selector names the row index document it "
                "is expanded against, by repo-relative path"
            )
        if not self.inputs:
            raise ValueError("a row_expansion figure binds at least one input role")
        unprofiled = [item.input_role for item in self.inputs if not item.is_row_expanded]
        if unprofiled:
            raise ValueError(
                f"row_expansion figure inputs {sorted(unprofiled)} state no per-row or "
                "shared binding profile; expansion cannot fill a role it cannot address"
            )
        per_row = sorted(item.input_role for item in self.inputs if item.is_per_row)
        if per_row and self.row_custody is not None and not self.row_custody.strip():
            raise ValueError(
                "a row_expansion figure states a nonempty 'row_custody' path or states none "
                "at all; an empty string names no document"
            )
        if not per_row and self.row_custody is not None:
            raise ValueError(
                "a row_expansion figure states 'row_custody' only when a per-row role is "
                "filled from it; every role here is shared, and a shared role names its own "
                "manifest"
            )
        return self

    def _validate_composition(self) -> "FigureLayerAuthoring":
        if self.delta is None:
            raise ValueError("a composition figure states the ordered deltas it composes")
        if (
            self.rows is not None
            or self.assembler_title is not None
            or self.row_custody is not None
        ):
            raise ValueError(
                "a composition figure states no row set, no row custody, and no assembler "
                "title; those are row_expansion vocabulary"
            )
        profiled = [item.input_role for item in self.inputs if item.is_row_expanded]
        if profiled:
            raise ValueError(
                f"composition figure inputs {sorted(profiled)} state a per-row or shared "
                "binding profile, which only row_expansion resolves"
            )
        return self


# -- report ----------------------------------------------------------------


class ReportBindingAuthoring(DialectModel):
    """One ordered report role and what fills it, including nothing at all."""

    role_path: str
    ref: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "ReportBindingAuthoring":
        if not self.role_path.strip() or any(
            not part for part in self.role_path.split(".")
        ):
            raise ValueError(f"report role path is not a dotted path: {self.role_path!r}")
        return self


class ReportLayerAuthoring(DialectModel):
    """Ordered field, section, and figure bindings for one report."""

    bindings: list[ReportBindingAuthoring] = Field(default_factory=list)
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ReportLayerAuthoring":
        paths = [item.role_path for item in self.bindings]
        if len(set(paths)) != len(paths):
            raise ValueError("report role paths must be bound at most once")
        if not self.bindings and self.delta is None:
            raise ValueError("a report layer binds roles, states a delta, or both")
        return self


# -- the envelope ----------------------------------------------------------


class ExperimentEnvelope(DialectModel):
    """One authored envelope: common fields plus exactly one layer."""

    schema_: Literal["feedbax.experiment_envelope.v1"] = Field(alias="schema")
    name: str
    base: str | None = None
    issue: str | None = None
    reason: str | None = None
    assert_: list[EnvelopeAssertion] = Field(default_factory=list, alias="assert")
    training: TrainingLayerAuthoring | None = None
    evaluation: EvaluationLayerAuthoring | None = None
    analysis: AnalysisLayerAuthoring | None = None
    figure: FigureLayerAuthoring | None = None
    report: ReportLayerAuthoring | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ExperimentEnvelope":
        if not self.name.strip():
            raise ValueError("an envelope states a nonempty name")
        if self.base is not None and not self.base.strip():
            raise ValueError("an envelope that states a base states a nonempty one")
        authored = [
            layer.value
            for layer in ExperimentEnvelopeLayer
            if self.layer_of(layer) is not None
        ]
        if len(authored) != 1:
            raise ValueError(
                f"an envelope authors exactly one layer, found {authored or 'none'}; "
                f"layers={[layer.value for layer in ExperimentEnvelopeLayer]}"
            )
        paths = [assertion.path for assertion in self.assert_]
        if len(set(paths)) != len(paths):
            raise ValueError("an envelope asserts each path at most once")
        return self

    def layer_of(self, layer: ExperimentEnvelopeLayer) -> Any:
        """Return the authored content of *layer*, or ``None``."""
        return getattr(self, ExperimentEnvelopeLayer(layer).value)

    @property
    def layer(self) -> ExperimentEnvelopeLayer:
        """Return the one layer this envelope authors."""
        return next(
            layer for layer in ExperimentEnvelopeLayer if self.layer_of(layer) is not None
        )

    @property
    def content(self) -> Any:
        """Return the one authored layer body."""
        return self.layer_of(self.layer)


# -- the layer output table -----------------------------------------------


@dataclass(frozen=True)
class LayerOutputContract:
    """What one layer compiles into. Data, resolved lazily to avoid import cycles.

    ``model_ref`` is a ``module:attribute`` pair rather than the class itself so
    that importing the dialect does not drag in the analysis and figure stacks.

    A family whose top-level document delegates its authored content to an inner
    ``params`` block states ``params_discriminator`` — the top-level field naming
    which content this is — and ``params_models``, the closed table of the models
    Feedbax validates that content with. A discriminator value absent from the
    table is content Feedbax does not own, and its ``params`` are left to whoever
    does.
    """

    layer: ExperimentEnvelopeLayer
    family: str
    schema_id: str
    schema_version: str
    model_ref: tuple[str, str]
    params_discriminator: str | None = None
    params_models: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def model(self) -> Any:
        """Import and return the Feedbax output model this layer compiles into."""
        return _import_attribute(self.model_ref)

    def params_model(self, document: Mapping[str, Any]) -> Any | None:
        """Return the model *document*'s declared content type validates against.

        ``None`` means this family takes no inner ``params`` block, or the
        document declares a content type Feedbax does not own.
        """
        if self.params_discriminator is None:
            return None
        ref = self.params_models.get(str(document.get(self.params_discriminator)))
        return None if ref is None else _import_attribute(ref)


def _import_attribute(ref: tuple[str, str]) -> Any:
    """Import and return one ``(module, attribute)`` pair."""
    from importlib import import_module

    module, attribute = ref
    return getattr(import_module(module), attribute)


TRAINING_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.TRAINING,
    "training_run_matrix",
    "feedbax.spec.training_run_matrix",
    "feedbax.spec.training_run_matrix.v5",
    ("feedbax.contracts.run_matrix", "TrainingRunMatrixSpec"),
)
EVALUATION_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.EVALUATION,
    "evaluation_run_matrix",
    "feedbax.spec.evaluation_run_matrix",
    "feedbax.spec.evaluation_run_matrix.v3",
    ("feedbax.analysis.evaluation", "EvaluationRunMatrixSpec"),
)
ANALYSIS_RUN_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.ANALYSIS,
    "analysis_run",
    "feedbax.spec.analysis_run",
    "feedbax.spec.analysis_run.v2",
    ("feedbax.contracts.manifest", "AnalysisRunSpec"),
)
ANALYSIS_BUNDLE_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.ANALYSIS,
    "analysis_bundle",
    "feedbax.spec.analysis_bundle",
    "feedbax.spec.analysis_bundle.v6",
    ("feedbax.analysis.bundles", "AnalysisBundleSpec"),
)
FIGURE_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.FIGURE,
    "figure",
    "feedbax.spec.figure",
    "feedbax.spec.figure.v2",
    ("feedbax.contracts.figures", "FigureSpec"),
)
FIGURE_COMPOSITION_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.FIGURE,
    "figure_composition",
    "feedbax.spec.figure_composition",
    "feedbax.spec.figure_composition.v2",
    ("feedbax.contracts.figures", "FigureCompositionSpec"),
)
#: Report ``report_type`` to the closed model validating that report's ``params``.
#: Both sides are Feedbax-owned, so this is Feedbax code; a report type absent
#: from it carries params whose owner is the recipe that registered the type.
REPORT_PARAMS_MODELS: Mapping[str, tuple[str, str]] = {
    "feedbax.ordered_figure_report": (
        "feedbax.analysis.reports",
        "OrderedFigureReportParams",
    ),
}

REPORT_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.REPORT,
    "report",
    "feedbax.spec.report",
    "feedbax.spec.report.v1",
    ("feedbax.contracts.manifest", "ReportSpec"),
    params_discriminator="report_type",
    params_models=REPORT_PARAMS_MODELS,
)

#: Every output a layer may compile into, keyed by the compiled document's own
#: ``schema_id``. This is also how a frozen base document announces its layer:
#: the document says what it is, and the table says which layer owns that.
LAYER_OUTPUT_CONTRACTS: Mapping[str, LayerOutputContract] = {
    contract.schema_id: contract
    for contract in (
        TRAINING_OUTPUT,
        EVALUATION_OUTPUT,
        ANALYSIS_RUN_OUTPUT,
        ANALYSIS_BUNDLE_OUTPUT,
        FIGURE_OUTPUT,
        FIGURE_COMPOSITION_OUTPUT,
        REPORT_OUTPUT,
    )
}

#: Where each layer places authored ``params``, as a dotted prefix into the
#: compiled document. A layer absent from this table takes no ``params``.
LAYER_PARAMS_PREFIX: Mapping[str, str] = {
    "evaluation": "base.params",
    "analysis:run": "params",
    "analysis:bundle": "params_base",
}

#: Where each layer places an authored ``recipe`` id.
LAYER_RECIPE_PATH: Mapping[str, str] = {
    "evaluation": "base.evaluation_type",
    "analysis:run": "analysis_type",
}


def layer_of_document(document: Mapping[str, Any]) -> ExperimentEnvelopeLayer | None:
    """Return the layer a compiled or frozen document belongs to, or ``None``.

    The answer comes from the document's own declared ``schema_id``, never from
    a filename or a path, so a document cannot be filed into a layer it is not a
    member of.
    """
    if not isinstance(document, Mapping):
        return None
    contract = LAYER_OUTPUT_CONTRACTS.get(str(document.get("schema_id")))
    return None if contract is None else contract.layer


def output_contract_of_document(document: Mapping[str, Any]) -> LayerOutputContract | None:
    """Return the output contract a compiled or frozen document conforms to."""
    if not isinstance(document, Mapping):
        return None
    return LAYER_OUTPUT_CONTRACTS.get(str(document.get("schema_id")))


def parse_experiment_envelope(
    document: Mapping[str, Any], *, field: str
) -> ExperimentEnvelope:
    """Parse one authored envelope, failing closed on anything unsupported.

    The declared ``schema`` is checked before the body so an envelope written
    against another version is refused by version rather than by whichever field
    happened to differ first.
    """
    if not isinstance(document, Mapping):
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.INVALID_VALUE,
            "an authored envelope is a JSON object",
            field=field,
        )
    declared = document.get("schema")
    if declared not in EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS:
        raise ExperimentEnvelopeRejection(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"unsupported envelope schema {declared!r}; "
            f"supported={list(EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS)}; "
            f"migration table={EXPERIMENT_ENVELOPE_MIGRATION_TABLE!r}; "
            "migration_intentionally_absent=yes",
            field=f"{field}#schema",
        )
    try:
        return ExperimentEnvelope.model_validate(document)
    except ValidationError as exc:
        raise ExperimentEnvelopeRejection(
            _rejection_category(exc),
            str(exc),
            field=field,
            correct_home="an envelope carries the scientific delta only, in the closed "
            f"{EXPERIMENT_ENVELOPE_SCHEMA_VERSION} dialect",
        ) from exc


def _rejection_category(error: ValidationError) -> ExperimentEnvelopeRejectionCategory:
    """Map one pydantic failure onto the closed authoring rejection vocabulary."""
    kinds = {entry["type"] for entry in error.errors()}
    if "extra_forbidden" in kinds:
        return ExperimentEnvelopeRejectionCategory.UNKNOWN_FIELD
    if "missing" in kinds:
        return ExperimentEnvelopeRejectionCategory.MISSING_FIELD
    return ExperimentEnvelopeRejectionCategory.INVALID_VALUE


__all__ = [
    "ANALYSIS_BUNDLE_OUTPUT",
    "ANALYSIS_RUN_OUTPUT",
    "EVALUATION_OUTPUT",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_ID",
    "EXPERIMENT_ENVELOPE_COMPILER_CONTRACT_VERSION",
    "EXPERIMENT_ENVELOPE_FAMILY",
    "EXPERIMENT_ENVELOPE_MIGRATION_TABLE",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION",
    "EXPERIMENT_ENVELOPE_SCHEMA_VERSION_V1",
    "EXPERIMENT_ENVELOPE_SUFFIX",
    "EXPERIMENT_ENVELOPE_SUPPORTED_SCHEMA_VERSIONS",
    "FIGURE_COMPOSITION_OUTPUT",
    "FIGURE_OUTPUT",
    "LAYER_OUTPUT_CONTRACTS",
    "LAYER_PARAMS_PREFIX",
    "LAYER_RECIPE_PATH",
    "REPORT_OUTPUT",
    "REPORT_PARAMS_MODELS",
    "TRAINING_OUTPUT",
    "AnalysisBundleRootAuthoring",
    "AnalysisLayerAuthoring",
    "AnalysisSubjectAuthoring",
    "AuthoredReference",
    "CheckpointInitializationAuthoring",
    "DialectModel",
    "EnvelopeAssertion",
    "EvaluationLayerAuthoring",
    "ExperimentEnvelope",
    "ExperimentEnvelopeLayer",
    "FigureInputAuthoring",
    "FigureLayerAuthoring",
    "FigureLayerMode",
    "FigureRoleContractAuthoring",
    "LayerOutputContract",
    "NotApplicableAuthoring",
    "ReceiptReference",
    "ReportBindingAuthoring",
    "ReportLayerAuthoring",
    "RowReplacement",
    "StagedPrerequisiteAuthoring",
    "TagsDelta",
    "TrainingLayerAuthoring",
    "TrainingRowAuthoring",
    "TrainingRowsMode",
    "UpstreamEnvelopeReference",
    "layer_of_document",
    "output_contract_of_document",
    "parse_experiment_envelope",
]
