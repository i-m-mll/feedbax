"""The one authored-experiment dialect Feedbax owns.

``feedbax.experiment_envelope.v1`` is the only envelope dialect there is. A
project does not define an envelope family, a layer, a lowerer, or a rule: it
authors documents in this dialect and Feedbax compiles them into the spec
families it already owns.

## Why one closed dialect rather than a per-project DSL

An extensible dialect makes every authored document mean whatever the installed
project says it means, which is exactly the property that makes a compiled
corpus unreadable a year later. The closed alternative works because the two
things a project really needs are already generic:

* **structure** — inheriting a row, naming it, seeding it, recording what it
  replaces, and stating an ordered patch layer over a content-pinned base. All
  of that is :class:`~feedbax.contracts.run_matrix.MatrixCompositionDelta` and
  :class:`~feedbax.contracts.manifest.OverridePatch`, which Feedbax already owns;
* **vocabulary** — dotted paths, values, component and recipe ids, input-role
  strings, and prose. All of that is *data inside* the structure, validated by
  the final output model and by whatever science plugin owns the payload.

So the dialect fixes the shape and stays silent about the words. Nothing here
names a task, an objective, a metric, or a project.

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
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, ValidationError, model_validator

from feedbax.contracts.checkpoint_initialization import CheckpointInitializationMode
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.figures import FigureCompositionDelta
from feedbax.contracts.manifest import StrictModel
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
    """

    from_: str = Field(alias="from")
    id: str
    seed: int | None = None
    replaces: RowReplacement | None = None
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TrainingRowAuthoring":
        if not self.from_.strip():
            raise ValueError("a training row inherits a nonempty parent row id")
        if not self.id.strip():
            raise ValueError("a training row states its own nonempty id")
        return self


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


class TrainingLayerAuthoring(DialectModel):
    """Rows inherited from the parent matrix, plus tags and checkpoint sources."""

    rows: list[TrainingRowAuthoring] = Field(default_factory=list)
    tags: TagsDelta | None = None
    checkpoint_initialization: list[CheckpointInitializationAuthoring] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _validate(self) -> "TrainingLayerAuthoring":
        if not self.rows and self.tags is None:
            raise ValueError("a training layer authors rows, tags, or both")
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


class EvaluationLayerAuthoring(DialectModel):
    """The subject an evaluation evaluates, and the recipe parameters it runs."""

    subject: AuthoredReference
    subject_id: str
    recipe: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    delta: MatrixCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "EvaluationLayerAuthoring":
        if not self.subject_id.strip():
            raise ValueError("an evaluation names its subject id")
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


class AnalysisLayerAuthoring(DialectModel):
    """A run or a bundle, its typed subjects, and its parameters."""

    target: Literal["run", "bundle"] = "run"
    subjects: list[AnalysisSubjectAuthoring] = Field(default_factory=list)
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
        return self


# -- figure ----------------------------------------------------------------


class FigureInputAuthoring(DialectModel):
    """One figure runtime input, addressed by its input role."""

    input_role: str
    ref: AuthoredReference

    @model_validator(mode="after")
    def _validate(self) -> "FigureInputAuthoring":
        if not self.input_role.strip():
            raise ValueError("a figure input states a nonempty input role")
        return self


class FigureLayerAuthoring(DialectModel):
    """Figure inputs bound by role, plus a native figure composition delta."""

    inputs: list[FigureInputAuthoring] = Field(default_factory=list)
    delta: FigureCompositionDelta | None = None

    @model_validator(mode="after")
    def _validate(self) -> "FigureLayerAuthoring":
        roles = [item.input_role for item in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError("figure input roles must be unique")
        if not self.inputs and self.delta is None:
            raise ValueError("a figure layer binds inputs, states a delta, or both")
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
    """

    layer: ExperimentEnvelopeLayer
    family: str
    schema_id: str
    schema_version: str
    model_ref: tuple[str, str]

    def model(self) -> Any:
        """Import and return the Feedbax output model this layer compiles into."""
        from importlib import import_module

        module, attribute = self.model_ref
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
REPORT_OUTPUT = LayerOutputContract(
    ExperimentEnvelopeLayer.REPORT,
    "report",
    "feedbax.spec.report.ordered_figure",
    "feedbax.spec.report.ordered_figure.v3",
    ("feedbax.analysis.reports", "OrderedFigureReportParams"),
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
    "TRAINING_OUTPUT",
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
    "LayerOutputContract",
    "NotApplicableAuthoring",
    "ReceiptReference",
    "ReportBindingAuthoring",
    "ReportLayerAuthoring",
    "RowReplacement",
    "TagsDelta",
    "TrainingLayerAuthoring",
    "TrainingRowAuthoring",
    "UpstreamEnvelopeReference",
    "layer_of_document",
    "output_contract_of_document",
    "parse_experiment_envelope",
]
