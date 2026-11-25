"""Compute a managed light delivery and submit its staged products to the Host."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from feedbax.host.operation_runtime import run_operation
from feedbax.contracts.artifact_custody import ContentRootRecord
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import Bindings, BoundMember, Product, Products, success_content_ref
from feedbax.contracts.figure_document import PLOTLY_MEDIA_TYPE
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.request_identity import DecodedRequest
from feedbax.contracts.request_refs import ContentRef
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    RenderRequestSpec,
    ReportRequestSpec,
)
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_figure import FigureRun
from feedbax.contracts.trial_clock import TrialMeanOptions, TrialMeanSelection
from feedbax.dataset import Dataset
from feedbax.contracts.member_stack import labelled_members
from feedbax.dataset_sweep import dataset_entries, stack_dataset_members
from feedbax.persistence.completion import read_completion_bindings
from feedbax.execution.trial_clock import TrialClockRecord
from feedbax.host.io import append
from feedbax.host.model_checkpoint_input import bind_managed_inputs
from feedbax.host.light_jobs import (
    LightCompletionBindings,
    LightDelivery,
    dataset_member,
    light_staging_id,
    light_staging_target,
    report_artifacts,
    operation_application,
    validate_light_request,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.evaluation_trial_records import RecordedTrialLayout, read_trial_clock
from feedbax.plot.dataset_figure import render as render_figure
from feedbax.plot.figure_trial_clock import FigureRecordedDataset

if TYPE_CHECKING:
    from feedbax.analysis.operations import RegisteredOperation
    from feedbax.plugins.declarations import ResolvedDeclaration


_MAX_CACHED_DATASET_BYTES = 4 * 1024 * 1024


@lru_cache(maxsize=2)
def _decoded_dataset(root: Path, digest: str, size: int, descriptor_sha256: str) -> Dataset:
    ref = DatasetProductRef(
        content=ContentRef(sha256=digest, size=size, media_type=DATASET_MEDIA_TYPE),
        descriptor_sha256=descriptor_sha256,
    )
    with open_stored_dataset(ImmutableArtifactBlobProvider(root), ref) as source:
        descriptor = source.descriptor
        arrays = {name: np.array(source.data(name), copy=True) for name in descriptor.tensor_names}
    for array in arrays.values():
        array.flags.writeable = False
    return Dataset.bind(
        descriptor,
        tensor_shapes={name: array.shape for name, array in arrays.items()},
        load_tensor=lambda name, index: arrays[name][index],
    )


def _dataset_inputs(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    stack: ExitStack,
) -> dict[str, Dataset]:
    """Each role's dataset: its labelled members, and the labelled entries of those that are stacks, stacked by their
    labels; or its one unlabelled member."""
    datasets = {}
    for role, members in bindings.items():
        labelled = {member.key: member for member in labelled_members(request.inputs[role])}
        inputs = []
        for member in members:
            ref = dataset_member(member)
            # The Store binds and verifies the exact object before every cache lookup.
            dataset = (
                _decoded_dataset(
                    store.root, ref.content.sha256, ref.content.size, ref.descriptor_sha256
                )
                if ref.content.size <= _MAX_CACHED_DATASET_BYTES
                else stack.enter_context(open_stored_dataset(store, ref))
            )
            inputs.append((labelled[member.key], dataset))
        if len(inputs) == 1 and not inputs[0][0].coordinates:
            ((_, datasets[role]),) = inputs
        else:
            datasets[role] = stack_dataset_members(
                [entry for member, dataset in inputs for entry in dataset_entries(member, dataset)]
            )
    return datasets


def _figure_recorded_dataset(dataset: Dataset, member_keys: frozenset[str]) -> FigureRecordedDataset:
    descriptor = dataset.descriptor

    def read_trial(trial: str, indices: tuple[int, ...]) -> tuple[TrialClockRecord, TrialMeanOptions]:
        layout = RecordedTrialLayout.model_validate(descriptor.attrs["trial_record"])
        return read_trial_clock(dataset, trial, indices, layout), layout.mean_options()

    return FigureRecordedDataset(dataset, member_keys, descriptor.variables["epoch_start"].dims[:-1], read_trial)


def _recorded_figure_trials(
    success: SuccessView, roles: set[str], bindings: Bindings, stack: ExitStack,
) -> dict[str, list[FigureRecordedDataset]]:
    """Find the exact source clocks, retaining the member keys of stacked Analyze inputs."""
    opened: dict[str, Dataset] = {}

    def recorded(member: BoundMember, keys: frozenset[str]) -> Iterator[tuple[str, frozenset[str]]]:
        ref = dataset_member(member)
        digest = ref.content.sha256
        dataset = opened.get(digest)
        if dataset is None:
            dataset = stack.enter_context(open_stored_dataset(success.store, ref))
            opened[digest] = dataset
        keys = keys | {member.key}
        descriptor = dataset.descriptor
        if "trial_record" in descriptor.attrs and "epoch_start" in descriptor.variables:
            yield digest, keys
        elif member.selection is not None:
            producer = success.read_receipt(member.selection.receipt)
            for members in producer.inputs.values():
                for parent in members:
                    if parent.content.media_type == DATASET_MEDIA_TYPE:
                        yield from recorded(parent, keys)

    sources = {}
    for role in roles:
        by_digest: dict[str, frozenset[str]] = {}
        for member in bindings[role]:
            for digest, keys in recorded(member, frozenset()):
                by_digest[digest] = by_digest.get(digest, frozenset()) | keys
        sources[role] = [_figure_recorded_dataset(opened[digest], keys) for digest, keys in by_digest.items()]
    return sources


def render_saved_figure(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    *,
    mean: TrialMeanSelection | None = None,
) -> bytes:
    """Draw directly from exact bound datasets, with no publication or job submission."""
    spec = request.spec
    if not isinstance(spec, RenderRequestSpec):
        raise ValueError("Expected a saved figure request.")
    with ExitStack() as stack:
        datasets = _dataset_inputs(store, request, bindings, stack)
        success = SuccessView(store)
        recorded = None
        if mean is not None:
            roles = {layer.role for layer in spec.document.layers if "trial" in layer.group}
            recorded = _recorded_figure_trials(success, roles, bindings, stack)
        figure = render_figure(
            spec.document, datasets, names=spec.names, mean=mean, recorded_trials=recorded,
            runs=figure_source_runs(success, bindings),
        )
        document = figure.to_json()
        assert document is not None
        return document.encode()


def _retain_staged_object(
    store: ImmutableArtifactBlobProvider, content: ContentRef, attempt: str
) -> None:
    store.register_root(
        ContentRootRecord(
            root_id=light_staging_id(attempt),
            root_kind="run",
            targets=(light_staging_target(success_content_ref(content)),),
        )
    )


def figure_source_runs(
    view: SuccessView, bindings: Bindings,
) -> dict[str, tuple[FigureRun, ...]]:
    """Follow the figure's bound datasets to the runs that produced their source data."""
    producers: dict[str, tuple[FigureRun, ...]] = {}

    def lineage(member: BoundMember) -> tuple[FigureRun, ...]:
        if member.selection is None or member.content.media_type != DATASET_MEDIA_TYPE:
            return ()
        ref = member.selection.receipt
        if ref.sha256 not in producers:
            receipt = view.read_receipt(ref)
            sources = tuple(
                run for group in receipt.inputs.values() for parent in group for run in lineage(parent)
            )
            producers[ref.sha256] = sources or (
                FigureRun(job_id=receipt.attempt, request_id=receipt.request.id),
            )
        return producers[ref.sha256]

    return {
        role: tuple(sources.values()) for role, members in bindings.items()
        if (sources := {run.job_id: run for member in members for run in lineage(member)})
    }


def _compute(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    attempt: str,
    role: str,
    operation: ResolvedDeclaration[RegisteredOperation] | None,
) -> Product:
    spec = request.spec
    if operation is not None:
        assert isinstance(spec, AnalysisRequestSpec)
        registry = operation_application()
        parameters = operation.value.parameter_model.model_validate(spec.params)
        if parameters.model_dump(mode="json") != spec.params:
            raise ValueError(f"Operation {operation.key!r} settings differ from their validated values")
        dataset_bindings = {
            name: bindings[name]
            for name, port in operation.value.definition.inputs.items()
            if isinstance(port, DatasetPort)
        }
        with ExitStack() as stack:
            dataset = run_operation(
                operation, parameters, bindings, _dataset_inputs(store, request, dataset_bindings, stack),
                registry=registry, artifacts=store,
            )
            with store._exclusive_lock():
                ref = publish_dataset(store, dataset)
                content = ref.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE})
                _retain_staged_object(store, content, attempt)
        return Product(
            name=role, content=content, descriptor={"descriptor_sha256": ref.descriptor_sha256}
        )
    if isinstance(spec, RenderRequestSpec):
        raw = render_saved_figure(store, request, bindings)
        with store._exclusive_lock():
            ref = store.store_object(raw)
            content = ContentRef(
                sha256=ref.sha256, size=ref.size_bytes, media_type=PLOTLY_MEDIA_TYPE
            )
            _retain_staged_object(store, content, attempt)
        return Product(name=role, content=content)
    if isinstance(spec, ReportRequestSpec):
        documents, members = report_artifacts(spec, bindings)
        with store._exclusive_lock():
            members.update({path: store.store_object(raw) for path, raw in documents.items()})
            tree, _ = store.publish_rooted_tree(
                members, root_id=light_staging_id(attempt), root_kind="run"
            )
        return Product(
            name=role,
            content=ContentRef(
                sha256=tree.descriptor.sha256, size=tree.descriptor.size_bytes, tree=True
            ),
        )
    raise ValueError("Expected a managed light request")


def check_light_environment(
    store: ImmutableArtifactBlobProvider,
    request: DecodedRequest,
    bindings: Bindings,
    products: Products,
    attempt: str,
) -> bool:
    """Replay the ordinary operation without publishing a successful receipt."""
    from feedbax.validity.light_products import light_product_matches

    role, operation = validate_light_request(request)
    (retained,) = products[role]
    candidate = _compute(store, request, bindings, attempt, role, operation)
    try:
        return light_product_matches(store, retained, candidate)
    finally:
        store.release_root(light_staging_id(attempt))


def main(_argv: list[str]) -> int:
    attempt = os.environ["FEEDBAX_HOST_JOB_ID"]
    log = Path(os.environ["FEEDBAX_HOST_EVENT_LOG"])
    payload = Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes()
    delivery = LightDelivery.model_validate(strict_json_loads(payload))
    exact = LightCompletionBindings.model_validate(read_completion_bindings())
    store = ImmutableArtifactBlobProvider(delivery.manifest_root)
    view = SuccessView(store)
    request = view.read_request(delivery.request)
    role, operation = validate_light_request(request)
    with bind_managed_inputs(
        view, delivery.request, origins=exact.origins, checkpoint_bindings=exact.bindings, project=delivery.project_id,
    ) as (_, bindings, _):
        product = _compute(store, request, bindings, attempt, role, operation)
    result = {"outcome": "succeeded", "products": {role: [product.model_dump(mode="json")]}}
    if operation is not None:
        result["provider"] = asdict(operation.source)
    append(log, "result", result, sync=True)
    return 0
