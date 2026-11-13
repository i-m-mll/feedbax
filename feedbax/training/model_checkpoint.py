"""Immutable semantic model slots for the qualified Studio component owners."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    CONTENT_TREE_REF_PREFIX,
    ArtifactBlobCustodyError,
    ContentObjectRef,
    ContentTreeRef,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_ARRAY_PATH as _ARRAY,
    MODEL_TREE_GRAPH_PATH as _GRAPH,
    MODEL_TREE_MEDIA_TYPE as MODEL_TREE_MEDIA_TYPE,
    SemanticModelEncoding,
)
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.model_occurrence import graph_leaf_specs
from feedbax.persistence.array_container import (
    ArrayContainerError,
    decode_array_container,
    encode_array_container,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
    content_object_relative_path,
    decode_content_tree_descriptor,
)
from feedbax.persistence.model_tree import (
    ModelTreeLayoutError,
    model_tree_asset_path,
    model_tree_assets,
)
from feedbax.runtime.graph import Graph
from feedbax.runtime.model_values import MODEL_OWNERS, SPATIAL_OWNER
from feedbax.training.checkpoint_errors import CheckpointCustodyError

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.contracts.checkpoints import CheckpointSlotBlobRef


MODEL_TREE_SERIALIZER = "feedbax.semantic-model-tree.v1"
_MODEL_STORE = Path("semantic-model")


class ModelCheckpointError(CheckpointCustodyError):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


def _slot_path(ref: ContentObjectRef) -> Path:
    return _MODEL_STORE / content_object_relative_path(ref.sha256)


def _slot_tree(slot: CheckpointSlotBlobRef) -> ContentTreeRef:
    if slot.slot != "model" or slot.role != "model" or slot.media_type != MODEL_TREE_MEDIA_TYPE:
        raise ModelCheckpointError("/v1/model", "invalid semantic model slot")
    descriptor = ContentObjectRef(
        object_id=f"{CONTENT_OBJECT_REF_PREFIX}{slot.sha256}",
        sha256=slot.sha256,
        size_bytes=slot.size_bytes,
    )
    if slot.relative_path != _slot_path(descriptor).as_posix():
        raise ModelCheckpointError("/v1/model", "semantic model tree path disagrees with its digest")
    return ContentTreeRef(
        tree_id=f"{CONTENT_TREE_REF_PREFIX}{slot.sha256}",
        descriptor=descriptor,
    )


def effective_model_graph(authored: GraphSpec, model: Graph) -> GraphSpec:
    """Include the model interface exposed by the prepared graph."""
    return authored.model_copy(update={
        "nodes": {
            name: spec.model_copy(update={"revision": model.component_bindings[name].revision})
            for name, spec in authored.nodes.items()
        },
        "input_ports": list(model.input_ports),
        "input_bindings": dict(model.input_bindings),
        "subgraphs": {
            name: effective_model_graph(interior, model.nodes[name])
            if isinstance(interior, GraphSpec) else interior
            for name, interior in (authored.subgraphs or {}).items()
        } or None,
    })


@dataclass(frozen=True)
class SemanticModelWriter:
    """The candidate writer is selected only for a fully declared Studio graph."""

    graph: GraphSpec
    assets: ArtifactBlobProvider | None

    @classmethod
    def for_graph(
        cls, graph: GraphSpec, assets: ArtifactBlobProvider | None,
    ) -> SemanticModelWriter | None:
        if not graph.nodes:
            return None
        for path, node in graph_leaf_specs(graph):
            interior = (graph.subgraphs or {}).get(path[0]) if len(path) == 1 else None
            owner = MODEL_OWNERS.get(node.type)
            if (
                owner is not None and owner.fixed_configuration and node.params.get("fixed") is True
                and node.revision in {None, owner.revision}
            ):
                continue
            if owner is None or not owner.checkpointable or node.revision not in {None, owner.revision}:
                return None
            if node.type != SPATIAL_OWNER and interior is None:
                continue
            if (
                node.type == SPATIAL_OWNER
                and isinstance(interior, AcausalGraphSpec)
                and interior.mujoco_asset is not None
                and interior.solver.solver_type == "mujoco"
                and interior.solver.mjx_backend == "jax"
                and node.params.get("trainable") is not True
            ):
                continue
            return None
        return cls(graph, assets)

    def publish(self, transaction: Path, model: Graph) -> tuple[Path, ContentObjectRef, dict]:
        store = ImmutableArtifactBlobProvider(transaction.absolute() / _MODEL_STORE)
        arrays, declarations = _describe(self.graph, model)
        graph = effective_model_graph(self.graph, model)
        try:
            encoded = encode_array_container(arrays, metadata=declarations)
        except ArrayContainerError as exc:
            raise ModelCheckpointError("/v1/model", str(exc)) from exc
        members = {
            _GRAPH: store.store_object(canonical_json_v2_bytes(graph.model_dump(mode="json"))),
            _ARRAY: store.store_object(encoded.data),
        }
        for path, asset in model_asset_refs(graph):
            if self.assets is None:
                raise ModelCheckpointError(path, "missing model asset provider")
            raw = _read(self.assets, asset, path)
            if store.store_object(raw) != asset:
                raise ModelCheckpointError(path, "model asset bytes differ from the authored reference")
            members[model_tree_asset_path(asset)] = asset
        tree, _ = store.publish_rooted_tree(
            members, root_id="checkpoint:model", root_kind="logical-ref",
        )
        encoding = SemanticModelEncoding(array_descriptor_sha256=encoded.descriptor_sha256)
        return _slot_path(tree.descriptor), tree.descriptor, encoding.model_dump(mode="json")


def semantic_model_member_files(
    slot: CheckpointSlotBlobRef, raw_tree: bytes,
) -> dict[Path, ContentObjectRef]:
    """List the member closure from an already hash-checked slot descriptor."""
    tree = decode_content_tree_descriptor(raw_tree, _slot_tree(slot))
    return {_slot_path(file.object): file.object for file in tree.files}


@dataclass(frozen=True)
class ModelRestoreTarget:
    """A resolved variant and its authorized scalar source-value overrides."""

    graph: GraphSpec
    template: Graph
    source_overrides: Mapping[str, float]


@dataclass(frozen=True)
class SemanticModelSlot:
    provider: ImmutableArtifactBlobProvider
    graph: GraphSpec
    arrays: ContentObjectRef
    encoding: SemanticModelEncoding
    tree: ContentTreeRef

    @classmethod
    def open(cls, transaction: Path, slot: CheckpointSlotBlobRef) -> SemanticModelSlot:
        provider = ImmutableArtifactBlobProvider(transaction.absolute() / _MODEL_STORE)
        try:
            tree = _slot_tree(slot)
            members = {file.relative_path: file.object for file in provider.get_tree(tree).files}
            graph = GraphSpec.model_validate_json(_read(provider, members[_GRAPH], "/graph"))
            if SemanticModelWriter.for_graph(graph, provider) is None:
                raise ModelCheckpointError("/graph", "graph has an undeclared semantic model owner")
            for path, asset in model_tree_assets(graph, members):
                _read(provider, asset, path)
            return cls(
                provider, graph, members[_ARRAY],
                SemanticModelEncoding.model_validate(slot.metadata["semantic_model"]), tree,
            )
        except ModelCheckpointError:
            raise
        except ModelTreeLayoutError as exc:
            raise ModelCheckpointError(exc.path, exc.detail) from exc
        except (ValueError, KeyError, OSError) as exc:
            raise ModelCheckpointError("/v1/model", str(exc)) from exc

    def template(self) -> Graph:
        from feedbax.compiler import GraphDocument, compile_graph
        from feedbax.component_registry import ComponentRegistry

        return compile_graph(
            GraphDocument(graph=self.graph), ComponentRegistry(),
            artifact_provider=self.provider,
        ).graph

    def restore(
        self, template: Graph | None = None, *, target: ModelRestoreTarget | None = None,
    ) -> Graph:
        """Verify the source slot, then bind compatible retained values to the target."""
        if template is None:
            template = self.template()
        try:
            decoded = decode_array_container(
                _read(self.provider, self.arrays, "/v1/model"),
                expected_descriptor_sha256=self.encoding.array_descriptor_sha256,
            )
        except ArrayContainerError as exc:
            raise ModelCheckpointError("/v1/model", str(exc)) from exc
        expected, declarations = _describe(self.graph, template)
        actual = decoded.descriptor.metadata
        if not isinstance(actual, dict):
            raise ModelCheckpointError("/v1/model", "missing value declarations")
        _verify_values(expected, declarations, decoded.tensors, actual)

        retained = frozenset(path for path, _ in graph_leaf_specs(self.graph))
        overrides: Mapping[str, float] = {}
        if target is not None:
            destination, destination_declarations = _describe(
                target.graph, target.template, retained=retained,
            )
            _verify_values(destination, destination_declarations, decoded.tensors, actual)
            template, overrides = target.template, target.source_overrides

        applied = set()

        def restore_occurrences(current: Graph, scope: tuple[str, ...]) -> Graph:
            nodes = dict(current.nodes)
            for name, component in nodes.items():
                path = (*scope, name)
                if isinstance(component, Graph):
                    nodes[name] = restore_occurrences(component, path)
                    continue
                if path not in retained:
                    continue
                binding = current.component_bindings[name]
                source = {}
                learned = []
                owner = MODEL_OWNERS[binding.type_id]
                for value in owner.values if not owner.fixed_configuration else ():
                    address = semantic_key("model", "value", path, value.symbol)
                    if address not in decoded.tensors:
                        continue
                    data = decoded.tensors[address]
                    if value.kind == "source":
                        if address in overrides:
                            applied.add(address)
                        source[value.symbol] = overrides.get(address, float(data))
                    else:
                        learned.append((value.field, jnp.asarray(data)))
                if source:
                    component = type(component).from_config(component.configuration_values() | source)
                if learned:
                    paths, tensors = zip(*learned, strict=True)
                    component = eqx.tree_at(
                        lambda owner: tuple(attrgetter(path)(owner) for path in paths),
                        component, tensors,
                    )
                nodes[name] = component
            return eqx.tree_at(lambda graph: graph.nodes, current, nodes)

        restored = restore_occurrences(template, ())
        if unexpected := overrides.keys() - applied:
            raise ModelCheckpointError(min(unexpected), "override has no declared scalar source destination")
        return restored.prepare()


def _verify_values(expected, declarations, tensors, saved_declarations) -> None:
    keys = expected.keys()
    unexpected = (keys ^ saved_declarations.keys()) | (keys ^ tensors.keys())
    if unexpected:
        raise ModelCheckpointError(min(unexpected), "missing or unexpected authoritative value")
    for path in sorted(keys):
        current, restored = expected[path], tensors[path]
        if saved_declarations[path] != declarations[path]:
            raise ModelCheckpointError(path, "owner, revision or value schema differs")
        if restored.shape != current.shape or restored.dtype != current.dtype:
            raise ModelCheckpointError(
                path, f"authoritative value shape or dtype differs: "
                f"checkpoint {restored.shape} {restored.dtype}, destination {current.shape} {current.dtype}",
            )


def _describe(
    graph: GraphSpec, model: Graph, *, retained: Set[tuple[str, ...]] | None = None,
) -> tuple[dict, dict]:
    def verify_occurrences(spec: GraphSpec, runtime: Graph, scope: tuple[str, ...]) -> None:
        if runtime.nodes.keys() != spec.nodes.keys() or runtime.nodes.keys() != runtime.component_bindings.keys():
            raise ModelCheckpointError(
                "/v1/model/" + "/".join(scope), "effective graph and realized owner occurrences differ",
            )
        for name, interior in (spec.subgraphs or {}).items():
            if isinstance(interior, GraphSpec):
                node = runtime.nodes[name]
                authored = spec.nodes[name]
                binding = runtime.component_bindings[name]
                if not isinstance(node, Graph):
                    raise ModelCheckpointError("/v1/model/" + "/".join((*scope, name)), "composite is missing")
                if binding.type_id != authored.type or authored.revision not in {None, binding.revision}:
                    raise ModelCheckpointError(
                        "/v1/model/" + "/".join((*scope, name)), "composite declaration differs",
                    )
                verify_occurrences(interior, node, (*scope, name))

    verify_occurrences(graph, model, ())
    arrays, declarations = {}, {}
    specs = dict(graph_leaf_specs(graph))
    if retained is None:
        retained = specs.keys()
    if missing := retained - specs.keys():
        path = min(missing)
        raise ModelCheckpointError(
            semantic_key("model", "value", path[:-1], path[-1]),
            "retained owner has no destination",
        )
    for path in sorted(retained):
        scope = model
        for name in path[:-1]:
            if not isinstance(child := scope.nodes.get(name), Graph):
                raise ModelCheckpointError("/v1/model", "effective graph and realized owner occurrences differ")
            scope = child
        name = path[-1]
        owner_path = semantic_key("model", "value", path[:-1], name)
        component = scope.nodes.get(name)
        binding = scope.component_bindings.get(name)
        if component is None or binding is None:
            raise ModelCheckpointError(owner_path, "effective graph and realized owner occurrences differ")
        owner = MODEL_OWNERS.get(binding.type_id)
        if (
            owner is not None and owner.fixed_configuration and binding.revision == owner.revision
            and specs[path].type == binding.type_id and specs[path].params.get("fixed") is True
        ):
            continue
        if (
            owner is None or not owner.checkpointable or specs[path].type != binding.type_id
            or binding.revision != owner.revision
        ):
            raise ModelCheckpointError(owner_path, "unsupported runtime owner identity or revision")
        for value in owner.values:
            address = semantic_key("model", "value", path, value.symbol)
            leaf = attrgetter(value.field)(component)
            if leaf is None and value.kind == "learned":
                continue
            if type(leaf) is float:
                representation = "python-float"
            elif isinstance(leaf, jax.Array) and not leaf.weak_type:
                representation = "jax-array"
            else:
                raise ModelCheckpointError(address, "unsupported authoritative value representation")
            arrays[address] = np.asarray(leaf)
            declarations[address] = {
                "owner_type": binding.type_id,
                "owner_revision": binding.revision,
                "value_schema": f"{binding.type_id}/{value.symbol}/v1",
                "representation": representation,
            }
    return arrays, declarations


def _read(provider: ArtifactBlobProvider, ref: ContentObjectRef, path: str) -> bytes:
    try:
        return provider.get_object(ref)
    except (ArtifactBlobCustodyError, OSError) as exc:
        raise ModelCheckpointError(path, f"cannot read {ref.object_id}: {exc}") from exc
