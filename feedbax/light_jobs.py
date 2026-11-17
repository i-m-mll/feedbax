"""Validate managed light requests and accept their staged Store products."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath

from pydantic import TypeAdapter, field_validator

from feedbax.analysis.operations import RegisteredOperation
from feedbax.plugins.declarations import ResolvedDeclaration
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
from feedbax.contracts.operation import DatasetPort, require_dataset_fields
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import ContentRef, ProductRef, RequestId
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    RenderRequestSpec,
    ReportRequestSpec,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_retention import read_active_roots
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView


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
    bindings: tuple[()]
    origins: dict[RequestId, ContentObjectRef]


_PRODUCTS = TypeAdapter(Products)
_ORIGINS = TypeAdapter(dict[RequestId, ContentObjectRef])


def light_operation(key: str) -> ResolvedDeclaration[RegisteredOperation]:
    from feedbax.plugins.composition import load_application
    from feedbax.plugins.declarations import missing_declaration

    try:
        return load_application().bundle.operations.resolve(key)
    except KeyError as error:
        raise ValueError(missing_declaration("operation", key)) from error


def validate_light_request(request: DecodedRequest) -> str:
    """Check managed execution support and return the declared output role."""
    spec = request.spec
    if spec.numerical_settings.model_dump(exclude_none=True):
        raise ValueError("Managed light requests require the default numerical settings")
    if isinstance(spec, AnalysisRequestSpec):
        operation = light_operation(spec.analysis_type).value
        operation.parameter_model.model_validate(spec.params)
        roles = operation.definition.inputs.keys()
        (output,) = operation.definition.outputs
    elif isinstance(spec, RenderRequestSpec) and spec.document is not None:
        roles = spec.document.data
        output = PLOTLY_FIGURE_ROLE
    elif isinstance(spec, ReportRequestSpec) and spec.document is not None:
        roles = spec.document.figure_roles
        output = "report"
    else:
        raise ValueError("Expected an Analysis, dataset Figure, or Report document")

    if request.inputs.keys() != set(roles):
        raise ValueError("Managed light inputs must match the declared roles")
    for role, members in request.inputs.items():
        if not members:
            raise ValueError(f"Managed light input {role!r} requires a member")
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
        descriptor = dataset.descriptor
        if descriptor.schema_id not in port.schema_ids:
            raise ValueError(
                f"Managed Analysis output requires one of the schemas {port.schema_ids}"
            )
        require_dataset_fields(descriptor.variables, port.variables)
        require_dataset_fields(descriptor.coords, port.coordinates)


def _verify_staged_product(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    product: Product,
) -> None:
    spec = request.spec
    ref = success_content_ref(product.content)
    if isinstance(spec, AnalysisRequestSpec):
        _verify_dataset_output(
            store, product, light_operation(spec.analysis_type).value.definition.outputs[product.name]
        )
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
    target: AttemptTarget = LOCAL_TARGET,
) -> dict:
    """Bind the same exact inputs independently and commit the Host-accepted attempt."""
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
        role = validate_light_request(request)
        if set(result) != {role} or len(result[role]) != 1 or result[role][0].name != role:
            raise ValueError(f"Managed {request.kind} success requires one {role!r} product")
        (product,) = result[role]
        _, bindings = view.bind(delivery.request, origins=exact_origins)
        if view.attempt_ref(job_id) is None:
            (root,) = read_active_roots(store, (root_id,))
            if root.targets != (light_staging_target(success_content_ref(product.content)),):
                raise ValueError("Light product differs from the exact staged attempt output")
            _verify_staged_product(store, request, bindings, product)
        success = view.publish_bound(
            delivery.request, attempt=job_id, products=result, bindings=bindings, target=target,
        )
        store.release_root(root_id)
    return success.ref.model_dump(mode="json")
