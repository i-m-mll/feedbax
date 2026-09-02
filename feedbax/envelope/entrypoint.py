"""The one built-in compiler for the one experiment-envelope dialect.

Every supported ``feedbax.experiment_envelope`` version is compiled by Feedbax
itself, not by any project. That is the whole point of the closed dialect: a
project cannot claim the schema, cannot supply a second compiler for it, and
cannot change what a compiled document means. Dispatch is direct — there is no
registry, no registration record, and no injectable callable between an
authored envelope and this function. An envelope outside the enumerated set is refused by
:func:`feedbax.contracts.experiment_envelope.require_builtin_envelope_schema`.

A project reaches this compiler by registering its data declaration. The caller
resolves which declaration owns an envelope by matching the envelope's
repo-relative path against declared envelope directories, and hands the result
in on the request. Nothing about the compile is decided by a project name.
"""

from __future__ import annotations

from feedbax.contracts.authoring_budget import (
    AuthoringBudgets,
    load_authoring_budget_document,
)
from feedbax.contracts.experiment_compile_lock import CompilerImplementation
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeCompileRequest,
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerError,
    ExperimentEnvelopeParentAuthority,
)
from feedbax.contracts.experiment_envelope_dialect import ExperimentEnvelopeLayer
from feedbax.contracts.project_experiment import ProjectExperimentDeclaration
from feedbax.envelope.compile import EnvelopeKernel

#: The owner string the single built-in registration declares.
EXPERIMENT_ENVELOPE_COMPILER_OWNER = "feedbax.envelope.entrypoint"

#: The physical provenance recorded on every lock this compiler emits.
EXPERIMENT_ENVELOPE_IMPLEMENTATION = CompilerImplementation(
    code_unit=EXPERIMENT_ENVELOPE_COMPILER_OWNER,
    packages=("feedbax",),
)

#: Every layer a budget document must state caps for. The dialect's layers are
#: fixed, so a budget that omits one is incomplete rather than permissive.
DECLARED_LAYERS: tuple[str, ...] = tuple(layer.value for layer in ExperimentEnvelopeLayer)


def load_project_budgets(declaration: ProjectExperimentDeclaration) -> AuthoringBudgets:
    """Load the authoring budgets one project's declaration points at."""
    resource = declaration.authoring_budget
    field = f"{resource.resource_id}#{resource.document_name}"
    raw = (resource.root / resource.document_name).read_bytes()
    document = load_authoring_budget_document(raw, field=field)
    return AuthoringBudgets.from_document(document, field=field, declared_layers=DECLARED_LAYERS)


def kernel_for(
    declaration: ProjectExperimentDeclaration,
    *,
    parent_authorities: tuple[ExperimentEnvelopeParentAuthority, ...] = (),
) -> EnvelopeKernel:
    """Return the one compiler bound to one project's data declaration."""
    return EnvelopeKernel(
        declaration=declaration,
        budgets=load_project_budgets(declaration),
        implementation=EXPERIMENT_ENVELOPE_IMPLEMENTATION,
        parent_authorities=parent_authorities,
    )


def compile_experiment_envelope(
    request: ExperimentEnvelopeCompileRequest,
) -> ExperimentEnvelopeCompileResult:
    """Compile one authored envelope and write its two outputs.

    The corpus is checked for colliding output addresses before this compile
    runs. One envelope cannot see that another one claims its output address,
    and by the time the outputs are written the collision has already happened,
    so the whole envelope directory is the unit the check runs over even when
    the unit of work is a single envelope.
    """
    declaration = request.project_declaration
    if declaration is None:
        raise ExperimentEnvelopeCompilerError(
            "compiling an experiment envelope needs the declaration of the project whose "
            "envelope directory holds it; resolve it before dispatch"
        )
    kernel = kernel_for(declaration, parent_authorities=request.parent_authorities)
    kernel.refuse_duplicate_output_addresses(request.repo_root, out_dir=request.out_dir)
    outcome = kernel.compile_envelope_file(request.envelope_path, repo_root=request.repo_root)
    paths = kernel.write_outputs(outcome, request.out_dir)
    return ExperimentEnvelopeCompileResult(
        # The version the compiled envelope declared, which a supported older
        # document keeps: the compile held it to that grammar, so naming the
        # current constant here would report a grammar it never compiled under.
        envelope_schema=outcome.envelope_schema,
        name=outcome.name,
        family=outcome.family,
        compile_lock_path=paths["compile_lock"].name,
        document_path=paths["document"].name,
    )


__all__ = [
    "DECLARED_LAYERS",
    "EXPERIMENT_ENVELOPE_COMPILER_OWNER",
    "EXPERIMENT_ENVELOPE_IMPLEMENTATION",
    "compile_experiment_envelope",
    "kernel_for",
    "load_project_budgets",
]
