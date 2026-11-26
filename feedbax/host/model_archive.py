"""Exchange native model slots independently of a Train transaction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import re
from stat import S_IFDIR, S_IFMT, S_IFREG
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import BadZipFile, ZIP_STORED, ZipFile, ZipInfo

import jax

from feedbax.compiler.graph import GraphCompilationError, GraphDocument, realize_graph, resolve_graph
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.artifact_custody import (
    ArtifactBlobCustodyError, ContentLeaseRecord, ContentRetentionTarget, ContentTree, ContentTreeFile, ContentTreeRef,
)
from feedbax.contracts.canonical_json import CanonicalJsonError, canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec, WireSpec
from feedbax.contracts.loaded_model import (
    ImportedModelSource, ModelFitProblem, ModelFitReport, ModelGraphFitProblem, encode_imported_model_document,
)
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_GRAPH_PATH, MODEL_TREE_MANIFEST_PATH, SemanticModelEncoding, model_replica_array_path,
)
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_values import ModelOwnerProblem, model_owner_problems, scalar_source_parameters
from feedbax.contracts.project import diff
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.execution.trial_resolution import declared_model_input_prototypes
from feedbax.host.model_checkpoint import (
    ModelCheckpointError, ModelSlotPrecision, SemanticModelSlot, effective_model_graph,
    imported_array_dtype_fits, model_slot_fit_problems,
)
from feedbax.persistence.array_container import ArrayContainerError, DecodedArrayContainer, decode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import ModelTree, ModelTreeLayoutError, read_model_tree
from feedbax.runtime.graph import Graph


_MEMBER_NAME = re.compile(r"replicas/(0|[1-9][0-9]*)\.safetensors|assets/[0-9a-f]{64}")
_MEMBER_DIRECTORIES = {"assets", "replicas"}


@dataclass(frozen=True)
class UnreadableModelArchive:
    path: str
    detail: str


class _ArchiveReadError(ValueError):
    def __init__(self, path: str, detail: str):
        self.problem = UnreadableModelArchive(path, detail)
        super().__init__(detail)


@contextmanager
def stage_model_archive(
    raw: bytes, model: ModelDocument, store: ImmutableArtifactBlobProvider, *, components: ComponentRegistry,
) -> Iterator[ImportedModelSource | ModelFitReport | UnreadableModelArchive]:
    """Hold both accepted content references while the caller publishes and roots its document."""
    with TemporaryDirectory(prefix="feedbax-model-") as directory:
        staging = ImmutableArtifactBlobProvider(Path(directory).resolve())
        try:
            tree, encoding = _read_archive_members(raw, staging)
            weights = _read_tree(staging, tree, encoding)
            decoded = tuple(_read_replica(weights, staging, index) for index in range(len(weights.replicas)))
            report = _fit_model(model, weights, decoded, staging, components)
        except _ArchiveReadError as error:
            yield error.problem
            return
        if report is not None:
            yield report
            return

        with store._exclusive_lock():
            for ref in weights.members.values():
                store.store_object(staging.get_object(ref), expected=ref)
            stored_tree = store.store_tree(staging.get_tree(tree))
            if stored_tree != tree:
                raise RuntimeError("Model archive descriptor changed during immutable staging")
            model_ref = store.store_object(encode_imported_model_document(model))
            source = ImportedModelSource(kind="imported", model=model_ref, weights=tree, encoding=encoding)
            lease = _lease_model(store, tree, model=ContentRetentionTarget(ref_kind="object", object=model_ref))
    try:
        yield source
    finally:
        store.release_lease(lease.lease_id)


def export_model_archive(
    store: ImmutableArtifactBlobProvider, tree: ContentTreeRef, encoding: SemanticModelEncoding,
) -> bytes | UnreadableModelArchive:
    """Wrap one exact native model checkpoint tree in a single ZIP file."""
    try:
        lease = _lease_model(store, tree)
    except (ArtifactBlobCustodyError, ValueError) as error:
        return UnreadableModelArchive("/model", str(error))
    try:
        model = _read_tree(store, tree, encoding)
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
            for path, ref in sorted(model.members.items()):
                try:
                    raw = store.get_object(ref)
                except (OSError, ArtifactBlobCustodyError) as error:
                    raise _ArchiveReadError("/" + path, str(error)) from error
                archive.writestr(ZipInfo(path), raw)
        return buffer.getvalue()
    except _ArchiveReadError as error:
        return error.problem
    finally:
        store.release_lease(lease.lease_id)


def _lease_model(
    store: ImmutableArtifactBlobProvider, tree: ContentTreeRef, *, model: ContentRetentionTarget | None = None,
) -> ContentLeaseRecord:
    targets = (ContentRetentionTarget(ref_kind="tree", tree=tree),)
    return store.acquire_lease(ContentLeaseRecord(
        lease_id=f"model-archive:{uuid4().hex}",
        owner="model archive",
        targets=targets if model is None else (*targets, model),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    ))


def _read_archive_members(
    raw: bytes, store: ImmutableArtifactBlobProvider,
) -> tuple[ContentTreeRef, SemanticModelEncoding]:
    files = {}
    seen = set()
    try:
        with ZipFile(BytesIO(raw)) as archive:
            for member in archive.infolist():
                path = member.filename
                directory = member.is_dir()
                kind = S_IFMT(member.external_attr >> 16)
                supported = (
                    path.rstrip("/") in _MEMBER_DIRECTORIES and path.count("/") == 1
                    if directory else path in {MODEL_TREE_GRAPH_PATH, MODEL_TREE_MANIFEST_PATH}
                    or _MEMBER_NAME.fullmatch(path) is not None
                )
                if (
                    not supported or member.orig_filename != path or path in seen
                    or member.flag_bits & 1 or member.compress_type != ZIP_STORED
                    or member.file_size != member.compress_size
                    or kind not in {0, S_IFDIR if directory else S_IFREG}
                    or directory and member.file_size != 0
                ):
                    raise _ArchiveReadError("/archive", f"Unsupported or repeated model archive member {path!r}")
                seen.add(path)
                if not directory:
                    try:
                        content = archive.read(member)
                    except (BadZipFile, EOFError, OSError) as error:
                        raise _ArchiveReadError("/" + path, str(error)) from error
                    files[path] = store.store_object(content)
    except (BadZipFile, EOFError, UnicodeDecodeError) as error:
        raise _ArchiveReadError("/archive", str(error)) from error

    manifest = files.get(MODEL_TREE_MANIFEST_PATH)
    if manifest is None:
        raise _ArchiveReadError("/model.json", "Model archive has no replica manifest")
    file_paths = sorted(files)
    descriptor = ContentTree(
        directories=tuple(sorted({path.split("/")[0] for path in file_paths if "/" in path})),
        files=tuple(ContentTreeFile(relative_path=path, object=files[path]) for path in file_paths),
    )
    return store.store_tree(descriptor), SemanticModelEncoding(descriptor_sha256=manifest.sha256)


def _read_tree(
    store: ImmutableArtifactBlobProvider, tree: ContentTreeRef, encoding: SemanticModelEncoding,
) -> ModelTree:
    try:
        return read_model_tree(store, tree, encoding)
    except ModelTreeLayoutError as error:
        raise _ArchiveReadError(error.path, error.detail) from error
    except (ArtifactBlobCustodyError, OSError, ValueError) as error:
        raise _ArchiveReadError("/model", str(error)) from error


def _read_replica(
    weights: ModelTree, staging: ImmutableArtifactBlobProvider, index: int,
) -> DecodedArrayContainer:
    replica = weights.replicas[index]
    path = model_replica_array_path(index)
    try:
        return decode_array_container(
            staging.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
        )
    except (ArrayContainerError, ArtifactBlobCustodyError, OSError) as error:
        raise _ArchiveReadError("/" + path, str(error)) from error


def _comparable_graph(graph: GraphSpec, components: ComponentRegistry) -> GraphSpec:
    return graph.model_copy(update={
        "nodes": {
            name: node.model_copy(update={
                "params": {
                    key: value for key, value in node.params.items()
                    if key not in sources
                },
            }) if (sources := scalar_source_parameters(node, components)) else node
            for name, node in graph.nodes.items()
        },
        "wires": sorted(graph.wires, key=WireSpec.authored_id),
        "subgraphs": {
            name: _comparable_graph(interior, components) if isinstance(interior, GraphSpec) else interior
            for name, interior in (graph.subgraphs or {}).items()
        } or None,
    })


@dataclass(frozen=True)
class _CompiledArchiveFit:
    graph: GraphSpec
    template: Graph
    owners: tuple[ModelOwnerProblem, ...]


def read_fitted_imported_replicas(
    model: ModelDocument, slot: SemanticModelSlot,
) -> tuple[DecodedArrayContainer, ...]:
    """Authenticate the slots against the pinned authored Model before any adaptation."""
    decoded = tuple(slot.read_replica(index) for index in range(len(slot.replicas)))
    try:
        report = _fit_model(model, slot, decoded, slot.provider, slot.components)
    except _ArchiveReadError as error:
        raise ModelCheckpointError(error.problem.path, error.problem.detail) from error
    if report is not None:
        problem = report.problems[0]
        raise ModelCheckpointError(problem.address, problem.detail)
    return decoded


def _fit_model(
    model: ModelDocument, weights: ModelTree | SemanticModelSlot, decoded: Sequence[DecodedArrayContainer],
    staging: ImmutableArtifactBlobProvider, components: ComponentRegistry,
) -> ModelFitReport | None:
    try:
        supplied = canonical_json_v2_bytes(_comparable_graph(weights.graph, components).model_dump(mode="json"))
    except CanonicalJsonError as error:
        raise _ArchiveReadError("/graph.json", str(error)) from error
    for index, replica in enumerate(decoded):
        if not isinstance(replica.descriptor.metadata, dict):
            raise _ArchiveReadError("/" + model_replica_array_path(index), "Replica has no named value declarations")

    current = _compile_fit_model(model, weights.graph, staging, components)
    problems = _fit_problems(current, weights, decoded, supplied, components, precision="exact")
    if problems and jax.config.x64_enabled and isinstance(current, _CompiledArchiveFit):
        # Archives carry their resolved declarations, so prove the original F32 layout
        # before checking whether it can populate the caller's F64 layout.
        with jax.enable_x64(False):
            source = _compile_fit_model(model, weights.graph, staging, components)
            source_problems = _fit_problems(source, weights, decoded, supplied, components, precision="exact")
        if not source_problems:
            problems = _fit_problems(current, weights, decoded, supplied, components, precision="imported")
        elif len(source_problems) < len(problems):
            problems = source_problems
    return ModelFitReport(problems=problems) if problems else None


def _compile_fit_model(
    model: ModelDocument, supplied: GraphSpec, staging: ImmutableArtifactBlobProvider, components: ComponentRegistry,
) -> _CompiledArchiveFit | list[ModelFitProblem]:
    lowered = None
    try:
        # Archived bindings select exposed optional inputs. All prototypes and
        # component configurations come from the selected authored Model.
        lowered = lower_model_document(model, components, supplied_inputs=frozenset(supplied.input_bindings))
        prototypes = declared_model_input_prototypes(model)
        resolved = resolve_graph(
            GraphDocument(graph=lowered.graph), components,
            {
                target: prototypes[name]
                for name, target in lowered.graph.input_bindings.items()
                if name in prototypes
            },
        )
        owners = tuple(model_owner_problems(resolved.graph, "checkpoint", components))
        template = realize_graph(resolved, artifact_provider=staging).graph
        return _CompiledArchiveFit(effective_model_graph(resolved.graph, template), template, owners)
    except ModelCompilationProblem as error:
        return [
            ModelGraphFitProblem(kind="unresolved_model", address=item.address.to_text(), detail=item.reason)
            for item in error.problems
        ]
    except GraphCompilationError as error:
        problems: list[ModelFitProblem] = []
        for diagnostic in error.record.diagnostics:
            path = diagnostic.source_anchor.authored_path
            address = lowered.source_address(path) if lowered is not None else None
            problems.append(ModelGraphFitProblem(
                kind="unresolved_model", address=address.to_text() if address else path,
                detail=diagnostic.observed_condition,
            ))
        return problems


def _fit_problems(
    fitted: _CompiledArchiveFit | list[ModelFitProblem], weights: ModelTree | SemanticModelSlot,
    decoded: Sequence[DecodedArrayContainer], supplied: bytes, components: ComponentRegistry,
    *, precision: ModelSlotPrecision,
) -> list[ModelFitProblem]:
    if isinstance(fitted, list):
        return fitted
    graph, template = fitted.graph, fitted.template
    problems: list[ModelFitProblem] = [
        ModelGraphFitProblem(kind="unsupported_owner", address=pointer(problem.path), detail=problem.reason)
        for problem in sorted(fitted.owners, key=lambda item: item.path)
    ]
    if precision == "exact":
        expected = canonical_json_v2_bytes(_comparable_graph(graph, components).model_dump(mode="json"))
        if expected != supplied:
            path = next(iter(diff(strict_json_loads(expected), strict_json_loads(supplied))))
            problems.append(ModelGraphFitProblem(
                kind="graph_mismatch", address="/graph" + path,
                detail="Archive graph differs from the graph resolved from the selected Model",
            ))
    ports = template.input_arrays.keys() | weights.input_arrays.keys()
    for name in sorted(ports):
        expected = template.input_arrays.get(name)
        saved = weights.input_arrays.get(name)
        if expected is not None and saved is not None:
            shape, dtype = expected
            if saved.shape == shape and (
                saved.dtype == dtype or precision == "imported" and imported_array_dtype_fits(saved.dtype, dtype)
            ):
                continue
        problems.append(ModelGraphFitProblem(
            kind="graph_mismatch", address=pointer(("model", "input_arrays", name)),
            detail="Archive input array differs from the input resolved from the selected Model",
        ))
    unsupported = tuple(problem.occurrence for problem in fitted.owners)
    for replica, arrays in zip(weights.replicas, decoded, strict=True):
        assert isinstance(arrays.descriptor.metadata, dict)
        try:
            problems.extend(model_slot_fit_problems(
                graph, template, replica.descriptor.id, arrays.tensors, arrays.descriptor.metadata,
                unsupported=unsupported, precision=precision,
            ))
        except ModelCheckpointError as error:
            problems.append(ModelGraphFitProblem(kind="unsupported_owner", address=error.path, detail=str(error)))
            break
    return problems
