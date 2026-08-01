"""Generic envelope compile orchestration.

The engine owns the *shape* of a compile and none of its science. It reads the
authored bytes under the project's budget, resolves the one parent and its
content-pinned lineage, refuses an assertion that guards something the envelope
itself changes, refuses an authored leaf that merely echoes what it inherits,
hands the resolved context to the project's layer compiler, proves the result is
what the project says it must be, and emits the compiled document plus its
compile lock.

Everything project-specific enters through two typed surfaces and no other:

* the :class:`~feedbax.contracts.project_extension.ProjectExtensionDeclaration`
  from the plugin bootstrap, which names the layers, the compiler contract, and
  the authoring budget;
* :class:`ProjectCompilerHooks`, the callables the project supplies for the
  decisions only it can make — which layer a document belongs to, what an
  envelope asserts, and how one layer actually compiles.

Compilation is a pure function of tracked content. It allocates nothing, writes
nothing outside an explicitly requested output directory, and touches no network.
Everything it emits is therefore a compile *plan*: it may quote an authenticated
reference a previous run produced, and it may never author one.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    build_compile_lock,
)
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
    envelope_schema_of,
)
from feedbax.contracts.project_extension import (
    EnvelopeFamilyStatus,
    ProjectExtensionDeclaration,
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
    envelope_suffix: str = ".envelope.json"

    def __post_init__(self) -> None:
        for name in ("envelope_directory", "output_directory", "envelope_suffix"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"EnvelopeLayout {name} must be nonempty")

    def alias_ref(self, alias: str) -> str:
        """Return the repo-relative envelope path one alias names."""
        return str(PurePosixPath(self.envelope_directory) / f"{alias}{self.envelope_suffix}")


@dataclass(frozen=True)
class AuthoredAssertion:
    """One inherited precondition an envelope states before its delta applies."""

    path: str
    equals: Any


@dataclass(frozen=True)
class ResolvedParent:
    """The one experiment parent, resolved and content-pinned."""

    kind: str
    ref: str
    pinned: PinnedDocument
    lineage: Lineage
    layer: str

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
    """Everything the engine resolved before one layer compiles.

    Attributes:
        envelope: The parsed authored envelope.
        envelope_ref: Its repo-relative path.
        envelope_schema: Its declared schema string.
        layer: The declared layer label this envelope authors.
        binding: The declaration's binding for that label.
        parent: The resolved, content-pinned parent, if the layer has one.
        repo_root: Root every reference resolves against.
        layout: Where envelopes and outputs live.
        budgets: The project's per-layer budget policy.
        declaration: The project's extension declaration.
        compile_upstream: Compile another envelope by alias, cycle-checked. This
            is how a layer resolves a cross-layer reference into the two facts
            that exist before anything runs.
    """

    envelope: Mapping[str, Any]
    envelope_ref: str
    envelope_schema: str
    layer: str
    binding: Any
    parent: ResolvedParent | None
    repo_root: Path
    layout: EnvelopeLayout
    budgets: AuthoringBudgets
    declaration: ProjectExtensionDeclaration
    compile_upstream: Callable[[str, str], "EnvelopeCompileOutcome"]

    @property
    def lineage(self) -> Lineage:
        """Return the parent's lineage, or an empty one for a root document."""
        return self.parent.lineage if self.parent is not None else Lineage(())

    def upstream_pin(self, alias: str, *, field: str) -> dict[str, Any]:
        """Compile one upstream envelope and return its two pre-run facts.

        The upstream envelope's own hash and the content hash of the document it
        compiles to are exactly what exists before anything runs. Nothing about a
        run — its id, its checkpoints, its transaction — is invented here.
        """
        upstream = self.compile_upstream(alias, field)
        return {
            "name": upstream.name,
            "envelope": dict(upstream.compile_lock["envelope"]),
            "compiled_document": dict(upstream.compile_lock["compiled_document"]),
        }


@dataclass(frozen=True)
class CompiledLayer:
    """What one layer's compiler produced, and what the lock should record.

    Attributes:
        name: The compiled output's name.
        family: The output family the document belongs to.
        document: The compiled document itself.
        resolved_deltas: What the compiler resolved, keyed by project concern.
        references: Cross-document references this compile resolved.
        identity_contributions: Compile-time facts beyond the document that make
            two otherwise-identical plans different executions.
        overridden_paths: Paths this envelope decides, mapped to the authored
            field that decides them. An assertion may not guard one of these.
        issue: Optional tracking reference for the authoring change.
    """

    name: str
    family: str
    document: Any
    resolved_deltas: Mapping[str, Any] = dataclass_field(default_factory=dict)
    references: Sequence[Mapping[str, Any]] = ()
    identity_contributions: Mapping[str, Any] = dataclass_field(default_factory=dict)
    overridden_paths: Mapping[str, str] = dataclass_field(default_factory=dict)
    issue: str | None = None


@dataclass(frozen=True)
class EnvelopeCompileOutcome:
    """The two outputs of one compile: a document and its compile lock."""

    name: str
    family: str
    layer: str
    document: Any
    compile_lock: dict[str, Any]


@dataclass(frozen=True)
class ProjectCompilerHooks:
    """The decisions only the project can make, as explicit callables.

    Attributes:
        layer_of: Which declared layer a parsed document belongs to, or ``None``
            when the document belongs to no layer of this project.
        compile_layer: How one resolved layer compiles.
        authored_assertions: The inherited preconditions an envelope states.
        parent_ref_of: The base an envelope names, or ``None`` for a root
            document that inherits from nothing.
        overridden_paths: Paths the envelope decides, when the project can say so
            before its layer compiles. Supplying this makes an illegal assertion
            a rejection before any compile work happens; omitting it defers to
            the compiled layer's own report, which reaches the same verdict.
        source_refs: Extra documents a parent draws from, beyond its own chain.
        validate_compiled: Proof that a compiled document really is a member of
            the family it claims, raising a rejection when it is not.
    """

    layer_of: Callable[[Mapping[str, Any]], str | None]
    compile_layer: Callable[[LayerCompileContext], CompiledLayer]
    authored_assertions: Callable[[Mapping[str, Any]], Sequence[AuthoredAssertion]] = (
        lambda _envelope: ()
    )
    parent_ref_of: Callable[[Mapping[str, Any]], str | None] = (
        lambda envelope: envelope.get("base") if isinstance(envelope.get("base"), str) else None
    )
    overridden_paths: Callable[[LayerCompileContext], Mapping[str, str]] | None = None
    source_refs: Callable[[Mapping[str, Any]], Iterable[str]] | None = None
    validate_compiled: Callable[[Any, str, str], None] | None = None


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
    assertions: Sequence[AuthoredAssertion],
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


class EnvelopeKernel:
    """The generic engine, bound to one project's declaration and hooks."""

    def __init__(
        self,
        *,
        declaration: ProjectExtensionDeclaration,
        layout: EnvelopeLayout,
        budgets: AuthoringBudgets,
        hooks: ProjectCompilerHooks,
        implementation: CompilerImplementation,
    ) -> None:
        self.declaration = declaration
        self.layout = layout
        self.budgets = budgets
        self.hooks = hooks
        self.implementation = implementation
        self.contract = CompilerContract.from_declaration(declaration)

    # -- envelope reading ------------------------------------------------

    def read_envelope(self, raw: bytes, *, envelope_ref: str) -> dict[str, Any]:
        """Read one authored envelope under its layer's budget."""
        return read_authored_document(
            raw, self.budgets, field=envelope_ref, layer_of=self.hooks.layer_of
        )

    def require_accepted_family(self, envelope: Mapping[str, Any], *, field: str) -> str:
        """Return the envelope's schema, refusing a retired or unclaimed family."""
        schema = envelope_schema_of(envelope)
        for family in self.declaration.envelope_families:
            if family.schema_version != schema:
                continue
            if EnvelopeFamilyStatus(family.status) is EnvelopeFamilyStatus.RETIRED:
                _reject(
                    ExperimentEnvelopeRejectionCategory.RETIRED_BASE_FAMILY,
                    f"{field}#schema",
                    f"{schema!r} is retired; project {self.declaration.project!r} now "
                    f"authors {family.superseded_by!r}",
                    correct_home="migrate the document once and commit the result; there is "
                    "no dual acceptance and no silent upgrade at read time",
                )
            return schema
        accepted = ", ".join(
            item.schema_version for item in self.declaration.accepted_families
        ) or "none"
        _reject(
            ExperimentEnvelopeRejectionCategory.UNSUPPORTED_SCHEMA_VERSION,
            f"{field}#schema",
            f"project {self.declaration.project!r} does not author {schema!r}; "
            f"accepted families: {accepted}",
        )

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
        expected_layer: str,
        field: str = "envelope.base",
    ) -> ResolvedParent:
        """Resolve the single parent: an envelope alias or a frozen document.

        The parent must belong to ``expected_layer``. The layer is read from the
        *resolved document* — the alias's compiled layer, or the frozen
        document's own declaration — never from the base string, so an alias
        chain cannot smuggle a cross-layer parent past the check.
        """
        self.refuse_compiled_output_base(base, field)
        if base.endswith(".json"):
            parent = self._resolve_frozen_parent(repo_root, base, field)
        else:
            parent = self._resolve_alias_parent(repo_root, base, stack, field)
        if parent.layer != expected_layer:
            _reject(
                ExperimentEnvelopeRejectionCategory.CROSS_FAMILY_BASE,
                field,
                f"a {expected_layer!r} envelope resolves {base!r} to {parent.ref}, which is a "
                f"{parent.layer!r} document; an envelope inherits from its own layer only",
                correct_home=f"the {parent.layer} document is the base of a {parent.layer} "
                f"envelope; a {expected_layer} envelope reaches it by name, as a cross-layer "
                "reference, not as a parent",
            )
        return parent

    def _resolve_frozen_parent(
        self, repo_root: Path, base: str, field: str
    ) -> ResolvedParent:
        pinned = load_pinned(repo_root, base, skips=("rows",))
        if pinned is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{base!r} is not a readable repo-relative JSON document",
            )
        layer = self.hooks.layer_of(pinned.document)
        if layer is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_BASE,
                field,
                f"{base!r} belongs to no layer project {self.declaration.project!r} declares, "
                "so it is not an experiment parent",
                correct_home="a base is a document of the same layer as the envelope that "
                "names it; an unrelated document is read through its own layer's envelope",
            )
        return ResolvedParent(
            "frozen_document",
            base,
            pinned,
            build_lineage(repo_root, pinned, source_refs=self.hooks.source_refs),
            layer,
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
        return ResolvedParent(
            "envelope_alias",
            alias_ref,
            pinned,
            build_lineage(repo_root, pinned, source_refs=self.hooks.source_refs),
            outcome.layer,
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
        schema = self.require_accepted_family(envelope, field=envelope_ref)
        layer = self.hooks.layer_of(envelope)
        if layer is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.MISSING_FIELD,
                envelope_ref,
                f"the envelope authors no layer project {self.declaration.project!r} "
                f"declares; declared layers: {list(self.declaration.labels('layer'))}",
            )
        binding = self.declaration.binding("layer", layer)
        if binding is None:
            _reject(
                ExperimentEnvelopeRejectionCategory.UNRESOLVED_EXTENSION_LABEL,
                envelope_ref,
                f"project {self.declaration.project!r} declares no layer named {layer!r}",
            )

        budget = self.budgets.for_layer(layer)
        assertions = tuple(self.hooks.authored_assertions(envelope))
        enforce_assertion_budget(len(assertions), budget, field=f"{envelope_ref}#assert")

        parent_ref = self.hooks.parent_ref_of(envelope)
        parent = (
            None
            if parent_ref is None
            else self.resolve_parent(
                repo_root, parent_ref, _stack, expected_layer=layer
            )
        )

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
            envelope_schema=schema,
            layer=layer,
            binding=binding,
            parent=parent,
            repo_root=repo_root,
            layout=self.layout,
            budgets=self.budgets,
            declaration=self.declaration,
            compile_upstream=compile_upstream,
        )

        early_overridden = (
            None
            if self.hooks.overridden_paths is None
            else dict(self.hooks.overridden_paths(context))
        )
        if early_overridden is not None:
            verify_assertions(assertions, context.lineage, early_overridden)

        compiled = self.hooks.compile_layer(context)
        assertion_records = verify_assertions(
            assertions,
            context.lineage,
            early_overridden
            if early_overridden is not None
            else dict(compiled.overridden_paths),
        )
        if self.hooks.validate_compiled is not None:
            self.hooks.validate_compiled(compiled.document, compiled.family, envelope_ref)

        lock = build_compile_lock(
            CompileLockInputs(
                envelope_ref=envelope_ref,
                envelope_document=envelope,
                envelope_schema=schema,
                name=compiled.name,
                family=compiled.family,
                compiled_document=compiled.document,
                contract=self.contract,
                implementation=self.implementation,
                base=None if parent is None else parent.lock_record(),
                lineage_pins=() if parent is None else parent.lineage.pins(),
                resolved_deltas=compiled.resolved_deltas,
                references=compiled.references,
                assertions=assertion_records,
                identity_contributions=compiled.identity_contributions,
                issue=compiled.issue,
            )
        )
        return EnvelopeCompileOutcome(
            name=compiled.name,
            family=compiled.family,
            layer=layer,
            document=compiled.document,
            compile_lock=lock,
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

    def write_outputs(
        self, outcome: EnvelopeCompileOutcome, out_dir: Path
    ) -> dict[str, Path]:
        """Write both outputs deterministically; re-running rewrites identical bytes."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = self.output_paths(outcome, out_dir)
        paths["compile_lock"].write_text(
            emit_text(outcome.compile_lock), encoding="utf-8"
        )
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
    Which suffixes are protected is the project's policy; that the rule exists is
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
    "AuthoredAssertion",
    "CompiledLayer",
    "EnvelopeCompileOutcome",
    "EnvelopeKernel",
    "EnvelopeLayout",
    "LayerCompileContext",
    "ProjectCompilerHooks",
    "ResolvedParent",
    "check_echo",
    "check_no_co_created_protected_document",
    "compiled_document_pin",
    "reject_echo",
    "scalar_equal",
    "verify_assertions",
]
