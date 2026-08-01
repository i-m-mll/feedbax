"""The built-in compiler registration for the one experiment-envelope dialect.

``feedbax.experiment_envelope.v1`` is claimed by Feedbax itself, not by any
project. That is the whole point of the closed dialect: a project cannot claim
the schema, cannot supply a second compiler for it, and cannot change what a
compiled document means. The compiler registry still mediates dispatch — so the
exit-code contract and the collision refusal both keep working — but there is
exactly one registration in it and Feedbax owns it.

A project reaches this compiler by registering its data declaration. The caller
resolves which declaration owns an envelope by matching the envelope's
repo-relative path against declared envelope directories, and hands the result
in on the request. Nothing about the compile is decided by a project name.
"""

from __future__ import annotations

from feedbax.contracts.authoring_budget import AuthoringBudgets, load_authoring_budget_document
from feedbax.contracts.experiment_compile_lock import CompilerImplementation
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeCompileRequest,
    ExperimentEnvelopeCompileResult,
    ExperimentEnvelopeCompilerError,
    ExperimentEnvelopeCompilerRegistration,
    ExperimentEnvelopeCompilerRegistry,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    ExperimentEnvelopeLayer,
)
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
    return AuthoringBudgets.from_document(
        document, field=field, declared_layers=DECLARED_LAYERS
    )


def kernel_for(declaration: ProjectExperimentDeclaration) -> EnvelopeKernel:
    """Return the one compiler bound to one project's data declaration."""
    return EnvelopeKernel(
        declaration=declaration,
        budgets=load_project_budgets(declaration),
        implementation=EXPERIMENT_ENVELOPE_IMPLEMENTATION,
    )


def compile_experiment_envelope(
    request: ExperimentEnvelopeCompileRequest,
) -> ExperimentEnvelopeCompileResult:
    """Compile one authored envelope and write its two outputs."""
    declaration = request.project_declaration
    if declaration is None:
        raise ExperimentEnvelopeCompilerError(
            "compiling an experiment envelope needs the declaration of the project whose "
            "envelope directory holds it; resolve it before dispatch"
        )
    kernel = kernel_for(declaration)
    outcome = kernel.compile_envelope_file(request.envelope_path, repo_root=request.repo_root)
    paths = kernel.write_outputs(outcome, request.out_dir)
    return ExperimentEnvelopeCompileResult(
        envelope_schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
        name=outcome.name,
        family=outcome.family,
        compile_lock_path=paths["compile_lock"].name,
        document_path=paths["document"].name,
    )


BUILTIN_EXPERIMENT_ENVELOPE_COMPILER = ExperimentEnvelopeCompilerRegistration(
    envelope_schema=EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    owner=EXPERIMENT_ENVELOPE_COMPILER_OWNER,
    compile=compile_experiment_envelope,
)


def register_builtin_experiment_envelope_compiler(
    registry: ExperimentEnvelopeCompilerRegistry,
) -> None:
    """Claim the one dialect for Feedbax's own compiler before plugins register."""
    registry.register(BUILTIN_EXPERIMENT_ENVELOPE_COMPILER)


__all__ = [
    "BUILTIN_EXPERIMENT_ENVELOPE_COMPILER",
    "DECLARED_LAYERS",
    "EXPERIMENT_ENVELOPE_COMPILER_OWNER",
    "EXPERIMENT_ENVELOPE_IMPLEMENTATION",
    "compile_experiment_envelope",
    "kernel_for",
    "load_project_budgets",
    "register_builtin_experiment_envelope_compiler",
]
