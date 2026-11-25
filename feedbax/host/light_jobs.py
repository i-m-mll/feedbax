"""Validate managed light requests and accept their staged Store products."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, field_validator

from feedbax.analysis.report_document import PLOTLY_SCRIPT_NAME, REPORT_HTML_NAME, render as render_report
from feedbax.contracts.artifact_custody import (
    ContentObjectRef,
    ContentRetentionTarget,
    ContentTreeRef,
)
from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import (
    Bindings,
    BoundMember,
    Product,
    Products,
    RequestObject,
    success_content_ref,
)
from feedbax.contracts.execution_target import AttemptTarget, LOCAL_TARGET
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, PlotlyFigureProduct
from feedbax.contracts.model_input import INITIAL_MODEL_MEDIA_TYPE, TASK_INPUT_MEDIA_TYPE, DocumentInput, read_document_input
from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE
from feedbax.contracts.model_checkpoint import MODEL_TREE_MEDIA_TYPE, SemanticModelEncoding
from feedbax.contracts.operation import DatasetPort, ModelPort, OperationDefinition, TaskPort
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import BatchMember, ContentRef, LastMember, ProductRef, RequestId, is_synthesized_model_product
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    NumericalSettings,
    RenderRequestSpec,
    ReportRequestSpec,
    SynthesisRequestSpec,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.synthesis import SynthesisMethodExecution
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import read_active_roots
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.declarations import DistributionSource, ResolvedDeclaration, missing_declaration

if TYPE_CHECKING:
    from feedbax.analysis.operations import RegisteredOperation
    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.synthesis.methods import RegisteredSynthesisMethod


type LightDeclaration = ResolvedDeclaration[RegisteredOperation] | ResolvedDeclaration[RegisteredSynthesisMethod]
type LightDefinition = OperationDefinition | SynthesisMethodExecution


class LightDelivery(StrictModel):
    project_id: str
    manifest_root: Path
    request: RequestObject

    @field_validator("manifest_root")
    @classmethod
    def _absolute_manifest_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Light delivery requires an absolute manifest root")
        return value


class LightCompletionBindings(StrictModel):
    bindings: tuple[dict, ...]
    origins: dict[RequestId, ContentObjectRef]


_PRODUCTS = TypeAdapter(Products)
_ORIGINS = TypeAdapter(dict[RequestId, ContentObjectRef])
_PROVIDER = TypeAdapter(DistributionSource)


@cache
def operation_application() -> ApplicationRegistryBundle:
    from feedbax.application.composition import load_application

    return load_application().bundle


def resolve_analysis_operation(
    spec: AnalysisRequestSpec, registry: ApplicationRegistryBundle,
) -> ResolvedDeclaration[RegisteredOperation]:
    try:
        operation = registry.operations.resolve(spec.analysis_type)
    except KeyError as error:
        raise ValueError(missing_declaration("operation", spec.analysis_type)) from error
    definition = operation.value.definition
    if operation.key != spec.analysis_type:
        raise ValueError(f"Use the canonical operation key {operation.key!r} in a managed request")
    for port in definition.inputs.values():
        if isinstance(port, TaskPort) and port.task_key is not None:
            task = registry.python_tasks.resolve(port.task_key)
            if task is None or task.parameters is not operation.value.task_parameters:
                raise ValueError(f"Operation {definition.key!r} requires the declared settings for {port.task_key!r}")
    return operation


def _typed_operation_input(port, members) -> bool:
    if isinstance(port, ModelPort):
        if len(members) != 1:
            return False
        reference = members[0].reference
        return (
            isinstance(reference, ProductRef) and reference.request_id.startswith("train:")
            and reference.role == "checkpoint" and isinstance(reference.select, (LastMember, BatchMember))
        ) or is_synthesized_model_product(reference) or (isinstance(reference, ContentRef) and not reference.tree and reference.media_type in {
            INITIAL_MODEL_MEDIA_TYPE, IMPORTED_MODEL_MEDIA_TYPE,
        })
    if isinstance(port, TaskPort):
        if len(members) != 1:
            return False
        reference = members[0].reference
        return isinstance(reference, ContentRef) and not reference.tree and reference.media_type == TASK_INPUT_MEDIA_TYPE
    return all(
        isinstance(member.reference, ProductRef) and (
            member.reference.request_id.startswith("analyze:")
            or member.reference.request_id.startswith("evaluate:") and member.reference.role == "states"
        )
        for member in members
    )


def validate_light_request(
    request: DecodedRequest, *, registry: ApplicationRegistryBundle | None = None,
    read_document: Callable[[ContentRef], DocumentInput] | None = None,
) -> tuple[str, LightDeclaration | None]:
    """Check declaration and binding metadata before managed dataset delivery."""
    from feedbax.analysis.operation_inputs import require_task_source

    spec = request.spec
    operation = None
    if isinstance(spec, AnalysisRequestSpec):
        registry = registry or operation_application()
        operation = resolve_analysis_operation(spec, registry)
    elif isinstance(spec, SynthesisRequestSpec):
        registry = registry or operation_application()
        operation = registry.synthesis_methods.declaration_record(spec.method)
        if operation.key != spec.method:
            raise ValueError(f"Use the canonical synthesis method key {operation.key!r} in a managed request.")
        settings = operation.value.declaration.settings_model.model_validate(spec.settings)
        if settings.model_dump(mode="json") != spec.settings:
            raise ValueError("Synthesis settings differ from their validated values.")
    output = validate_light_requirements(request, operation.value.definition if operation is not None else None)
    if isinstance(spec, AnalysisRequestSpec) and operation is not None and read_document is not None:
        assert registry is not None
        for role, port in operation.value.definition.inputs.items():
            if isinstance(port, TaskPort):
                reference = request.inputs[role][0].reference
                assert isinstance(reference, ContentRef)
                installed = registry.python_tasks.resolve(port.task_key) if port.task_key is not None else None
                require_task_source(role, read_document(reference), installed.key if installed else port.task_key)
    return output, operation


def validate_light_requirements(request: DecodedRequest, definition: LightDefinition | None) -> str:
    """Validate admitted ports from declaration metadata and authored input references."""
    spec = request.spec
    if isinstance(spec, SynthesisRequestSpec):
        if not isinstance(definition, SynthesisMethodExecution) or definition.key != spec.method:
            raise ValueError("Managed Synthesize requires metadata for its canonical method.")
        if definition.numerical.floating_point_bits != 64 or spec.numerical_settings != NumericalSettings(
            floating_point_bits=64,
        ):
            raise ValueError("Managed Synthesize requires the declared 64-bit environment.")
        if request.inputs:
            raise ValueError("Synthesize reads its resolved trial documents.")
        return "model"
    if isinstance(definition, SynthesisMethodExecution) or definition is not None and not isinstance(spec, AnalysisRequestSpec):
        raise ValueError("Only managed Analyze accepts operation metadata.")
    if isinstance(spec, AnalysisRequestSpec):
        if definition is None or definition.key != spec.analysis_type:
            raise ValueError("Managed Analyze requires metadata for its canonical operation.")
        roles = definition.inputs.keys()
        (output,) = definition.outputs
    elif isinstance(spec, RenderRequestSpec):
        roles = spec.document.data
        output = PLOTLY_FIGURE_ROLE
    elif isinstance(spec, ReportRequestSpec):
        roles = spec.document.figure_roles
        output = "report"
    else:
        raise ValueError("Expected a registered operation, dataset Figure, or Report document")
    if spec.numerical_settings != NumericalSettings():
        raise ValueError("Managed operation, Figure and Report requests require default numerical settings")

    if request.inputs.keys() != set(roles):
        raise ValueError("Managed light inputs must match the declared roles")
    for role, members in request.inputs.items():
        if not members:
            raise ValueError(f"Managed light input {role!r} requires a member")
        if definition is not None:
            port = definition.inputs[role]
            if not _typed_operation_input(port, members):
                raise ValueError(f"Operation input {role!r} differs from its declared {port.kind} port")
        if len(members) > 1 and (
            isinstance(spec, ReportRequestSpec) or not all(
                isinstance(member.reference, ProductRef) and (
                    member.reference.request_id.startswith("analyze:")
                    or member.reference.role == "states" and member.reference.request_id.startswith("evaluate:")
                )
                for member in members
            )
        ):
            raise ValueError(f"Multiple members of {role!r} must supply saved Evaluate states or operation datasets")
        for member in members:
            reference = member.reference
            if isinstance(reference, ProductRef):
                continue
            if isinstance(spec, AnalysisRequestSpec) and isinstance(reference, ContentRef):
                continue
            if isinstance(spec, ReportRequestSpec) and isinstance(reference, ContentRef):
                PlotlyFigureProduct(content=reference)
                continue
            raise ValueError(f"Input {member.key!r} requires a typed producer product")
    return output


def dataset_member(member: BoundMember | Product) -> DatasetProductRef:
    """Read the shared retained Dataset product encoding used by Evaluate and Analyze."""
    descriptor = member.descriptor
    if (
        member.content.media_type != DATASET_MEDIA_TYPE
        or not isinstance(descriptor, dict)
        or descriptor.keys() != {"descriptor_sha256"}
    ):
        raise ValueError(
            "Dataset input or output requires typed content and an array descriptor digest"
        )
    return DatasetProductRef.model_validate({"content": member.content, **descriptor})


def report_artifacts(
    spec: ReportRequestSpec,
    bindings: Bindings,
) -> tuple[dict[str, bytes], dict[str, ContentObjectRef]]:
    documents = {}
    script_digest = None
    if spec.document.figure_roles:
        from plotly.offline import get_plotlyjs  # pyright: ignore[reportMissingTypeStubs]

        script = get_plotlyjs().encode()
        documents[PLOTLY_SCRIPT_NAME] = script
        script_digest = sha256(script).hexdigest()
    report = render_report(
        spec.document,
        {role: PlotlyFigureProduct(content=member.content) for role, (member,) in bindings.items()},
        plotly_script_sha256=script_digest,
    )
    documents.update({REPORT_HTML_NAME: report.html.encode(), "report.md": report.markdown.encode()})
    figures = {}
    for path, content in report.figures.items():
        ref = success_content_ref(content)
        assert isinstance(ref, ContentObjectRef)  # PlotlyFigureProduct requires a blob.
        figures[path] = ref
    return documents, figures


def light_staging_id(job_id: str) -> str:
    return f"light-staging:{job_id}"


def light_staging_target(ref: ContentObjectRef | ContentTreeRef) -> ContentRetentionTarget:
    if isinstance(ref, ContentTreeRef):
        return ContentRetentionTarget(ref_kind="tree", tree=ref)
    return ContentRetentionTarget(ref_kind="object", object=ref)


def _verify_dataset_output(
    store: ImmutableArtifactBlobProvider, product: Product, port: DatasetPort
) -> None:
    # Opening a dataset loads JAX; the Host admits light jobs and must not pay for it.
    from feedbax.analysis.operation_inputs import uses_replicate_axis, verify_dataset
    from feedbax.dataset_replicates import replicate_ids, select_replicate
    from feedbax.persistence.dataset import open_stored_dataset

    with open_stored_dataset(store, dataset_member(product)) as dataset:
        if not uses_replicate_axis(port) and replicate_ids(dataset) is not None:
            dataset = select_replicate(dataset, 0)
        verify_dataset(dataset, port)


def _verify_staged_product(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    product: Product,
    definition: LightDefinition | None,
) -> None:
    spec = request.spec
    ref = success_content_ref(product.content)
    if isinstance(definition, SynthesisMethodExecution) and product.name == "model":
        from feedbax.persistence.model_tree import read_model_tree

        if not isinstance(ref, ContentTreeRef) or product.content.media_type != MODEL_TREE_MEDIA_TYPE:
            raise ValueError("Synthesize must publish a semantic model tree.")
        model = read_model_tree(store, ref, SemanticModelEncoding.model_validate(product.descriptor))
        if tuple(replica.descriptor.id for replica in model.replicas) != ("0",):
            raise ValueError("Synthesize must publish one model replica.")
        return
    if definition is not None:
        _verify_dataset_output(store, product, definition.outputs[product.name])
        return
    if product.descriptor is not None:
        raise ValueError(
            "Managed Figure and Report products require content without a secondary descriptor"
        )
    if isinstance(spec, RenderRequestSpec) and isinstance(ref, ContentObjectRef):
        PlotlyFigureProduct(content=product.content)
        figure = strict_json_loads(store.get_object(ref))
        if not isinstance(figure, dict) or not isinstance(figure.get("data"), list):
            raise ValueError("Managed Figure product must contain Plotly data")
        return
    if (
        isinstance(spec, ReportRequestSpec)
        and isinstance(ref, ContentTreeRef)
        and product.content.media_type is None
    ):
        documents, figures = report_artifacts(spec, bindings)
        tree = store.get_tree(ref)
        expected = {path: (sha256(raw).hexdigest(), len(raw)) for path, raw in documents.items()}
        expected.update({path: (ref.sha256, ref.size_bytes) for path, ref in figures.items()})
        actual = {
            file.relative_path: (file.object.sha256, file.object.size_bytes) for file in tree.files
        }
        directories = tuple(sorted({PurePosixPath(path).parent.as_posix() for path in figures}))
        if actual != expected or tree.directories != directories:
            raise ValueError(
                "Managed Report tree differs from the document and its exact bound figures"
            )
        return
    raise ValueError("Managed product differs from the authored output type")


def publish_light_success(
    payload: Mapping[str, object],
    job_id: str,
    request_id: str,
    origins: Mapping[str, object],
    products: object,
    *,
    checkpoint_bindings: tuple[dict, ...] = (),
    provider: object = None,
    target: AttemptTarget = LOCAL_TARGET,
    execution_evidence: ContentObjectRef | None = None,
) -> dict:
    """Bind the same exact inputs independently and commit the Host-accepted attempt."""
    from feedbax.host.model_checkpoint_input import bind_managed_inputs

    delivery = LightDelivery.model_validate(payload)
    if delivery.request.id != request_id:
        raise ValueError("Light delivery differs from the Host request ID")
    store = ImmutableArtifactBlobProvider(delivery.manifest_root)
    result = _PRODUCTS.validate_python(products)
    exact_origins = _ORIGINS.validate_python(origins)
    root_id = light_staging_id(job_id)
    with store._exclusive_lock():
        view = SuccessView(store)
        request = view.read_request(delivery.request)
        role, operation = validate_light_request(request, read_document=lambda content: read_document_input(content, store))
        if (operation is None) != (provider is None):
            raise ValueError("Managed operation success requires its executed provider provenance")
        provider_trace = _PROVIDER.validate_python(provider) if provider is not None else None
        definition = operation.value.definition if operation is not None else None
        if isinstance(definition, SynthesisMethodExecution) and provider_trace != operation.source:
            raise ValueError("Synthesis provider differs from the installed method declaration.")
        expected = {role, *definition.outputs} if isinstance(definition, SynthesisMethodExecution) else {role}
        if result.keys() != expected or any(len(items) != 1 or items[0].name != name for name, items in result.items()):
            raise ValueError(f"Managed {request.kind} success requires one member for each product in {sorted(expected)!r}.")
        order = (role, *sorted(expected - {role}))
        ordered = tuple(result[name][0] for name in order)
        with bind_managed_inputs(
            view, delivery.request, origins=exact_origins, checkpoint_bindings=checkpoint_bindings, project=delivery.project_id,
        ) as (_, bindings):
            if view.attempt_ref(job_id) is None:
                (root,) = read_active_roots(store, (root_id,))
                if root.targets != tuple(light_staging_target(success_content_ref(product.content)) for product in ordered):
                    raise ValueError("Managed product differs from the exact staged attempt output")
                for product in ordered:
                    _verify_staged_product(store, request, bindings, product, definition)
            success = view.publish_bound(
                delivery.request, attempt=job_id, products=result, bindings=bindings,
                provider=provider_trace, target=target,
                execution_evidence=execution_evidence,
            )
        store.release_root(root_id)
    return success.ref.model_dump(mode="json")
