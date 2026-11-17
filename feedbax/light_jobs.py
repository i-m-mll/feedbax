"""Validate managed light requests and accept their staged Store products."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, field_validator

from feedbax.analysis.operations import RegisteredOperation
from feedbax.analysis.operation_runtime import uses_replicate_axis, verify_dataset
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
from feedbax.contracts.model_input import INITIAL_MODEL_MEDIA_TYPE, TASK_INPUT_MEDIA_TYPE
from feedbax.contracts.operation import DatasetPort, ModelPort, TaskPort
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import BatchMember, ContentRef, LastMember, ProductRef, RequestId
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    NumericalSettings,
    RenderRequestSpec,
    ReportRequestSpec,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.dataset_replicates import replicate_ids, select_replicate
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import read_active_roots
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.declarations import DistributionSource, missing_declaration

if TYPE_CHECKING:
    from feedbax.plugins.application import ApplicationRegistryBundle


class LightDelivery(StrictModel):
    project_id: str
    manifest_root: Path
    request: RequestObject
    provider: DistributionSource | None = None

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


@cache
def operation_application() -> ApplicationRegistryBundle:
    from feedbax.plugins.composition import load_application

    return load_application().bundle


def resolve_analysis_operation(
    spec: AnalysisRequestSpec, registry: ApplicationRegistryBundle,
) -> RegisteredOperation:
    try:
        operation = registry.operations.resolve(spec.analysis_type)
    except KeyError as error:
        raise ValueError(missing_declaration("operation", spec.analysis_type)) from error
    definition = operation.definition
    if spec.computation != definition.computation():
        raise ValueError(f"Operation {definition.key!r} differs from the authored computation revision")
    numerical = definition.numerical
    if spec.numerical_settings != NumericalSettings(floating_point_bits=numerical.floating_point_bits if numerical else None):
        raise ValueError(f"Operation {definition.key!r} differs from the authored numerical settings")
    for port in definition.inputs.values():
        if isinstance(port, TaskPort) and port.task_key is not None:
            task = registry.python_tasks.resolve(port.task_key)
            if task is None or task.parameters is not operation.task_parameters:
                raise ValueError(f"Operation {definition.key!r} requires the declared settings for {port.task_key!r}")
    operation.parameter_model.model_validate(spec.params)
    return operation


def _typed_operation_input(port, members) -> bool:
    if isinstance(port, ModelPort):
        if len(members) != 1:
            return False
        reference = members[0].reference
        return (
            isinstance(reference, ProductRef) and reference.request_id.startswith("train:")
            and reference.role == "checkpoint" and isinstance(reference.select, (LastMember, BatchMember))
        ) or (isinstance(reference, ContentRef) and not reference.tree and reference.media_type == INITIAL_MODEL_MEDIA_TYPE)
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
) -> tuple[str, RegisteredOperation | None]:
    """Check managed dataset delivery support and return its declared output role."""
    spec = request.spec
    operation = None
    if isinstance(spec, AnalysisRequestSpec):
        operation = resolve_analysis_operation(spec, registry or operation_application())
        roles = operation.definition.inputs.keys()
        (output,) = operation.definition.outputs
    elif isinstance(spec, RenderRequestSpec) and spec.document is not None:
        roles = spec.document.data
        output = PLOTLY_FIGURE_ROLE
    elif isinstance(spec, ReportRequestSpec) and spec.document is not None:
        roles = spec.document.figure_roles
        output = "report"
    else:
        raise ValueError("Expected a registered operation, dataset Figure, or Report document")
    if not isinstance(spec, AnalysisRequestSpec) and spec.numerical_settings != NumericalSettings():
        raise ValueError("Figure and Report jobs require default numerical settings")

    if request.inputs.keys() != set(roles):
        raise ValueError("Managed light inputs must match the declared roles")
    for role, members in request.inputs.items():
        if not members:
            raise ValueError(f"Managed light input {role!r} requires a member")
        if operation is not None and not _typed_operation_input(operation.definition.inputs[role], members):
            raise ValueError(f"Operation input {role!r} differs from its declared {operation.definition.inputs[role].kind} port")
        if len(members) > 1 and (
            isinstance(spec, ReportRequestSpec) or not all(
                isinstance(member.reference, ProductRef)
                and member.reference.role == "states"
                and member.reference.request_id.startswith("evaluate:")
                for member in members
            )
        ):
            raise ValueError(f"Multiple members of {role!r} must supply saved Evaluate states")
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
    return output, operation


def verify_delivery_provider(delivery: LightDelivery, operation: RegisteredOperation | None) -> None:
    expected = operation.source if operation is not None else None
    if delivery.provider != expected:
        raise ValueError("Managed delivery provider differs from the admitted installed declaration")


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
    if spec.document is None:
        raise ValueError("Managed Report requires a document")
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
    with open_stored_dataset(store, dataset_member(product)) as dataset:
        if not uses_replicate_axis(port) and replicate_ids(dataset) is not None:
            dataset = select_replicate(dataset, 0)
        verify_dataset(dataset, port)


def _verify_staged_product(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    product: Product,
    operation: RegisteredOperation | None,
) -> None:
    spec = request.spec
    ref = success_content_ref(product.content)
    if operation is not None:
        _verify_dataset_output(store, product, operation.definition.outputs[product.name])
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
    target: AttemptTarget = LOCAL_TARGET,
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
        role, operation = validate_light_request(request)
        verify_delivery_provider(delivery, operation)
        if set(result) != {role} or len(result[role]) != 1 or result[role][0].name != role:
            raise ValueError(f"Managed {request.kind} success requires one {role!r} product")
        (product,) = result[role]
        with bind_managed_inputs(
            view, delivery.request, origins=exact_origins, checkpoint_bindings=checkpoint_bindings, project=delivery.project_id,
        ) as (_, bindings):
            if view.attempt_ref(job_id) is None:
                (root,) = read_active_roots(store, (root_id,))
                if root.targets != (light_staging_target(success_content_ref(product.content)),):
                    raise ValueError("Managed product differs from the exact staged attempt output")
                _verify_staged_product(store, request, bindings, product, operation)
            success = view.publish_bound(
                delivery.request, attempt=job_id, products=result, bindings=bindings,
                provider=delivery.provider, target=target,
            )
        store.release_root(root_id)
    return success.ref.model_dump(mode="json")
