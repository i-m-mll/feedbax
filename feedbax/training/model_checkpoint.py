"""Immutable semantic model slots for the qualified Studio component owners."""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.artifact_custody import (
    CONTENT_OBJECT_REF_PREFIX,
    CONTENT_TREE_REF_PREFIX,
    ArtifactBlobCustodyError,
    ContentObjectRef,
    ContentTreeRef,
)
from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec
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
from feedbax.runtime.graph import Graph
from feedbax.training.checkpoint_errors import CheckpointCustodyError

if TYPE_CHECKING:
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.contracts.checkpoints import CheckpointSlotBlobRef


MODEL_TREE_MEDIA_TYPE = "application/vnd.feedbax.semantic-model-tree+json"
MODEL_TREE_SERIALIZER = "feedbax.semantic-model-tree.v1"
_MODEL_STORE = Path("semantic-model")
_GRAPH = "graph.json"
_ARRAY = "model.safetensors"


class ModelCheckpointError(CheckpointCustodyError):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class _Value:
    symbol: str
    field: str
    kind: Literal["source", "learned"]


@dataclass(frozen=True)
class _Owner:
    revision: int
    values: tuple[_Value, ...]


def _learned(*values: tuple[str, str]) -> tuple[_Value, ...]:
    return tuple(_Value(symbol, field, "learned") for symbol, field in values)


def _source(*names: str) -> tuple[_Value, ...]:
    return tuple(_Value(name, name, "source") for name in names)


_SPATIAL = "feedbax.AcausalSystem"
_OWNERS = {
    "feedbax.GRU": _Owner(1, _learned(
        ("input_kernel", "cell.weight_ih"),
        ("recurrent_kernel", "cell.weight_hh"),
        ("gate_bias", "cell.bias"),
        ("candidate_bias", "cell.bias_n"),
    )),
    "feedbax.Linear": _Owner(2, _learned(("kernel", "layer.weight"), ("bias", "layer.bias"))),
    "feedbax.PointMass": _Owner(1, (
        _Value("mass", "plant.skeleton.mass", "source"),
        _Value("damping", "plant.skeleton.damping", "source"),
        *_source("dt"),
    )),
    "feedbax.RigidTendonHillMuscleThelen": _Owner(2, _source(
        "max_isometric_force", "optimal_muscle_length", "tendon_slack_length",
        "vmax_factor", "min_activation", "tau_activation", "tau_deactivation", "dt",
    )),
    _SPATIAL: _Owner(1, ()),
}


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _asset_path(ref: ContentObjectRef) -> str:
    return f"assets/{ref.sha256}"


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
    })


class _Encoding(StrictModel):
    array_descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
        for name, node in graph.nodes.items():
            owner = _OWNERS.get(node.type)
            if owner is None or node.revision not in {None, owner.revision}:
                return None
            interior = (graph.subgraphs or {}).get(name)
            if node.type != _SPATIAL and interior is None:
                continue
            if (
                node.type == _SPATIAL
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
            members[_asset_path(asset)] = asset
        tree, _ = store.publish_rooted_tree(
            members, root_id="checkpoint:model", root_kind="logical-ref",
        )
        encoding = _Encoding(array_descriptor_sha256=encoded.descriptor_sha256)
        return _slot_path(tree.descriptor), tree.descriptor, encoding.model_dump(mode="json")


def semantic_model_member_files(
    slot: CheckpointSlotBlobRef, raw_tree: bytes,
) -> dict[Path, ContentObjectRef]:
    """List the member closure from an already hash-checked slot descriptor."""
    tree = decode_content_tree_descriptor(raw_tree, _slot_tree(slot))
    return {_slot_path(file.object): file.object for file in tree.files}


@dataclass(frozen=True)
class SemanticModelSlot:
    provider: ImmutableArtifactBlobProvider
    graph: GraphSpec
    arrays: ContentObjectRef
    encoding: _Encoding

    @classmethod
    def open(cls, transaction: Path, slot: CheckpointSlotBlobRef) -> SemanticModelSlot:
        provider = ImmutableArtifactBlobProvider(transaction.absolute() / _MODEL_STORE)
        try:
            tree = provider.get_tree(_slot_tree(slot))
            members = {file.relative_path: file.object for file in tree.files}
            graph = GraphSpec.model_validate_json(_read(provider, members[_GRAPH], "/graph"))
            if SemanticModelWriter.for_graph(graph, provider) is None:
                raise ModelCheckpointError("/graph", "graph has an undeclared semantic model owner")
            assets = dict(model_asset_refs(graph))
            expected = {_GRAPH, _ARRAY, *(_asset_path(asset) for asset in assets.values())}
            if members.keys() != expected:
                raise ModelCheckpointError("/v1/model", "model tree members disagree with the effective graph")
            for path, asset in assets.items():
                if members[_asset_path(asset)] != asset:
                    raise ModelCheckpointError(path, "model tree pins a different asset")
                _read(provider, asset, path)
            return cls(provider, graph, members[_ARRAY], _Encoding.model_validate(
                slot.metadata["semantic_model"]
            ))
        except ModelCheckpointError:
            raise
        except (ValueError, KeyError, OSError) as exc:
            raise ModelCheckpointError("/v1/model", str(exc)) from exc

    def template(self) -> Graph:
        from feedbax.compiler import GraphDocument, compile_graph
        from feedbax.component_registry import ComponentRegistry

        return compile_graph(
            GraphDocument(graph=self.graph), ComponentRegistry(),
            artifact_provider=self.provider,
        ).graph

    def restore(self, template: Graph | None = None) -> Graph:
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
        keys = expected.keys()
        unexpected = (keys ^ actual.keys()) | (keys ^ decoded.tensors.keys())
        if unexpected:
            raise ModelCheckpointError(min(unexpected), "missing or unexpected authoritative value")
        for path, value in expected.items():
            restored = decoded.tensors[path]
            if actual[path] != declarations[path]:
                raise ModelCheckpointError(path, "owner, revision or value schema differs")
            if restored.shape != value.shape or restored.dtype != value.dtype:
                raise ModelCheckpointError(path, "authoritative value shape or dtype differs")

        nodes = dict(template.nodes)
        for name, component in template.nodes.items():
            binding = template.component_bindings[name]
            values = _OWNERS[binding.type_id].values
            source = {}
            learned = []
            for value in values:
                address = f"/v1/model/value/{_pointer_token(name)}/{value.symbol}"
                if address in decoded.tensors:
                    data = decoded.tensors[address]
                    if value.kind == "source":
                        source[value.symbol] = float(data)
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
        return eqx.tree_at(lambda graph: graph.nodes, template, nodes).prepare()


def _describe(graph: GraphSpec, model: Graph) -> tuple[dict, dict]:
    if model.nodes.keys() != graph.nodes.keys() or model.component_bindings.keys() != graph.nodes.keys():
        raise ModelCheckpointError("/v1/model", "effective graph and realized owner occurrences differ")
    arrays, declarations = {}, {}
    for name, component in model.nodes.items():
        binding = model.component_bindings[name]
        owner_path = f"/v1/model/value/{_pointer_token(name)}"
        owner = _OWNERS.get(binding.type_id)
        if (
            owner is None or graph.nodes[name].type != binding.type_id
            or binding.revision != owner.revision
        ):
            raise ModelCheckpointError(owner_path, "unsupported runtime owner identity or revision")
        for value in owner.values:
            path = f"{owner_path}/{value.symbol}"
            leaf = attrgetter(value.field)(component)
            if leaf is None and value.kind == "learned":
                continue
            if type(leaf) is float:
                representation = "python-float"
            elif isinstance(leaf, jax.Array) and not leaf.weak_type:
                representation = "jax-array"
            else:
                raise ModelCheckpointError(path, "unsupported authoritative value representation")
            arrays[path] = np.asarray(leaf)
            declarations[path] = {
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
