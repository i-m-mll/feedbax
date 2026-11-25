"""Immutable semantic model slots for the qualified Studio component owners."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler.model_assets import model_asset_refs
from feedbax.contracts.artifact_custody import (
    ArtifactBlobCustodyError,
    ContentObjectRef,
    ContentRetentionTarget,
    ContentTreeRef,
)
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.loaded_model import ModelSlotFitProblem
from feedbax.contracts.model_address import FieldElement, ModelAddress
from feedbax.contracts.method_checkpoint import MethodCheckpointModel
from feedbax.contracts.model_checkpoint import (
    MODEL_TREE_GRAPH_PATH as _GRAPH,
    MODEL_TREE_MANIFEST_PATH as _MANIFEST,
    MODEL_TREE_MEDIA_TYPE as MODEL_TREE_MEDIA_TYPE,
    ModelReplica,
    ModelReplicaManifest,
    SemanticModelEncoding,
    model_replica_array_path,
)
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.model_occurrence import graph_leaf_occurrences, graph_leaf_specs
from feedbax.persistence.array_container import (
    ArrayContainerError,
    decode_array_container,
    encode_array_container,
)
from feedbax.persistence.artifact_custody import (
    ImmutableArtifactBlobProvider,
)
from feedbax.persistence.model_tree import (
    MODEL_CHECKPOINT_STORE as _MODEL_STORE,
    ModelTreeLayoutError,
    ModelTreeReplica,
    model_tree_asset_path,
    read_model_tree,
)
from feedbax.runtime.graph import Graph
from feedbax.contracts.model_values import model_owner_problem
from feedbax.persistence.checkpoint_errors import CheckpointCustodyError

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.compiler.model_document import LoweredModelDocument




class ModelCheckpointError(CheckpointCustodyError):
    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


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
        cls, graph: GraphSpec, assets: ArtifactBlobProvider | None, components: ComponentRegistry,
    ) -> SemanticModelWriter | None:
        if not graph.nodes or model_owner_problem(graph, "checkpoint", components) is not None:
            return None
        return cls(graph, assets)

    def publish(
        self, transaction: Path, models: Sequence[Graph], *,
        templates: Sequence[Graph], replica_ids: tuple[str, ...],
    ) -> MethodCheckpointModel:
        store = ImmutableArtifactBlobProvider(transaction.absolute() / _MODEL_STORE)
        return self.publish_to(
            store, models, templates=templates, replica_ids=replica_ids,
            root_id="checkpoint:model", root_kind="logical-ref",
        )

    def publish_to(
        self, store: ImmutableArtifactBlobProvider, models: Sequence[Graph], *,
        templates: Sequence[Graph], replica_ids: tuple[str, ...], root_id: str,
        root_kind: Literal["logical-ref", "run"], additional_targets: tuple[ContentRetentionTarget, ...] = (),
    ) -> MethodCheckpointModel:
        if not len(models) == len(templates) == len(replica_ids) or not replica_ids:
            raise ModelCheckpointError("/model", "models must match a nonempty ordered list of unique replicas")
        graph = effective_model_graph(self.graph, models[0])
        members = {
            _GRAPH: store.store_object(canonical_json_v2_bytes(graph.model_dump(mode="json"))),
        }
        replicas = []
        for index, (replica_id, model, template) in enumerate(zip(replica_ids, models, templates, strict=True)):
            path = model_replica_array_path(index)
            if effective_model_graph(self.graph, model) != graph:
                raise ModelCheckpointError(path, "replicas do not share one effective graph")
            arrays, declarations = _describe(graph, model)
            _verify_exact_model(model, _restore_model(graph, template, arrays, declarations))
            try:
                encoded = encode_array_container(
                    {address: np.asarray(array) for address, array in arrays.items()},
                    metadata=declarations,
                )
            except ArrayContainerError as exc:
                raise ModelCheckpointError(path, str(exc)) from exc
            members[path] = store.store_object(encoded.data)
            replicas.append(ModelReplica(id=replica_id, array_descriptor_sha256=encoded.descriptor_sha256))
        manifest = ModelReplicaManifest(replicas=tuple(replicas))
        members[_MANIFEST] = store.store_object(canonical_json_v2_bytes(manifest.model_dump(mode="json")))
        for path, asset in model_asset_refs(graph):
            if self.assets is None:
                raise ModelCheckpointError(path, "missing model asset provider")
            raw = _read(self.assets, asset, path)
            if store.store_object(raw) != asset:
                raise ModelCheckpointError(path, "model asset bytes differ from the authored reference")
            members[model_tree_asset_path(asset)] = asset
        tree, _ = store.publish_rooted_tree(
            members, root_id=root_id, root_kind=root_kind, additional_targets=additional_targets,
        )
        encoding = SemanticModelEncoding(descriptor_sha256=members[_MANIFEST].sha256)
        return MethodCheckpointModel(tree=tree, encoding=encoding)


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
    replicas: tuple[ModelTreeReplica, ...]
    encoding: SemanticModelEncoding
    tree: ContentTreeRef
    components: ComponentRegistry

    @classmethod
    def open_store(
        cls, provider: ImmutableArtifactBlobProvider, tree: ContentTreeRef, encoding: SemanticModelEncoding,
        *, components: ComponentRegistry,
    ) -> SemanticModelSlot:
        try:
            model = read_model_tree(provider, tree, encoding)
            if SemanticModelWriter.for_graph(model.graph, provider, components) is None:
                raise ModelCheckpointError("/graph", "graph has an undeclared semantic model owner")
            for path, asset in model.assets:
                _read(provider, asset, path)
            return cls(provider, model.graph, model.replicas, encoding, tree, components)
        except ModelCheckpointError:
            raise
        except ModelTreeLayoutError as exc:
            raise ModelCheckpointError(exc.path, exc.detail) from exc
        except (ValueError, KeyError, OSError) as exc:
            raise ModelCheckpointError("/v1/model", str(exc)) from exc

    @property
    def replica_ids(self) -> tuple[str, ...]:
        return tuple(replica.descriptor.id for replica in self.replicas)

    def template(self) -> Graph:
        from feedbax.compiler import GraphDocument, compile_graph

        return compile_graph(
            GraphDocument(graph=self.graph), self.components,
            artifact_provider=self.provider,
        ).graph

    def restore(
        self, template: Graph | None = None, *,
        target: ModelRestoreTarget | None = None, replica_index: int = 0, synthesized: bool = False,
    ) -> Graph:
        """Verify the source slot, then bind compatible retained values to the target."""
        if template is None:
            template = self.template()
        if not 0 <= replica_index < len(self.replicas):
            raise ModelCheckpointError("/model/replicas", "unknown replica index")
        replica = self.replicas[replica_index]
        path = model_replica_array_path(replica_index)
        try:
            decoded = decode_array_container(
                _read(self.provider, replica.arrays, path),
                expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
            )
        except ArrayContainerError as exc:
            raise ModelCheckpointError(path, str(exc)) from exc
        return _restore_model(
            self.graph, template, decoded.tensors, decoded.descriptor.metadata, target, synthesized=synthesized,
        )

    def restore_all(
        self, templates: Sequence[Graph], *,
        targets: Sequence[ModelRestoreTarget] | None = None, synthesized: bool = False,
    ) -> tuple[Graph, ...]:
        """Restore every replica using templates in checkpoint order."""
        if len(templates) != len(self.replicas) or (targets is not None and len(targets) != len(templates)):
            raise ModelCheckpointError("/model/replicas", "templates do not match the checkpoint replica count")
        return tuple(
            self.restore(
                template, target=None if targets is None else targets[index], replica_index=index, synthesized=synthesized,
            )
            for index, template in enumerate(templates)
        )


def _restore_model(
    graph: GraphSpec, template: Graph, tensors: Mapping[str, jax.Array | np.ndarray],
    actual: object, target: ModelRestoreTarget | None = None, *, synthesized: bool = False,
) -> Graph:
    expected, declarations = _describe(graph, template)
    if not isinstance(actual, dict):
        raise ModelCheckpointError("/v1/model", "missing value declarations")
    if synthesized:
        tensors = _synthesized_precision(expected, tensors, declarations)
    _verify_values(expected, declarations, tensors, actual)

    source_leaves = dict(graph_leaf_specs(graph))
    retained = frozenset(source_leaves)
    overrides: Mapping[str, float] = {}
    if target is not None:
        from feedbax.contracts.model_attachment import is_generated_attachment_node

        destination_leaves = dict(graph_leaf_specs(target.graph))
        removed = {
            path for path in retained - destination_leaves.keys()
            if is_generated_attachment_node(path[-1], source_leaves[path].type)
            and not any(
                value.kind == "parameter"
                for value in template.get_node(".".join(path[:-1])).component_bindings[path[-1]].model_facts.values
            )
        }
        retained -= removed
        destination, destination_declarations = _describe(
            target.graph, target.template, retained=retained,
        )
        _verify_values(destination, destination_declarations, tensors, actual)
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
            owner = binding.model_facts
            node = destination_leaves[path] if target is not None else source_leaves[path]
            for value in owner.values if not owner.is_fixed_configuration(node) else ():
                address = semantic_key("model", "value", path, value.symbol)
                if address not in tensors:
                    continue
                data = tensors[address]
                if value.kind == "source":
                    if address in overrides:
                        applied.add(address)
                    source[value.symbol] = overrides.get(address, float(data))
                else:
                    learned.append((value.field, jnp.asarray(data)))
            if source:
                component = type(component).from_config(component.configuration_values() | source)
            if learned:
                paths, replacements = zip(*learned, strict=True)
                component = eqx.tree_at(
                    lambda owner: tuple(attrgetter(path)(owner) for path in paths),
                    component, replacements,
                )
            nodes[name] = component
        return eqx.tree_at(lambda graph: graph.nodes, current, nodes)

    restored = restore_occurrences(template, ())
    if unexpected := overrides.keys() - applied:
        raise ModelCheckpointError(min(unexpected), "override has no declared scalar source destination")
    return restored.prepare()


def _synthesized_precision(expected, tensors, declarations):
    """Round a 64-bit synthesized array to the consumer's declared 32-bit slot."""
    adapted = dict(tensors)
    for address in expected.keys() & tensors.keys():
        current, saved = expected[address], tensors[address]
        if (
            declarations[address]["representation"] != "jax-array" or saved.shape != current.shape
            or (saved.dtype, current.dtype) != (np.dtype("float64"), np.dtype("float32"))
        ):
            continue
        with np.errstate(over="ignore"):
            target = np.asarray(saved, dtype=np.float32)
        if not np.isfinite(target).all():
            raise ModelCheckpointError(address, "Synthesized parameter is outside the consumer's finite float32 range")
        adapted[address] = target
    return adapted


def _verify_exact_model(original: Graph, restored: Graph) -> None:
    before, structure = jax.tree.flatten_with_path(original)
    after, restored_structure = jax.tree.flatten(restored)
    if structure != restored_structure:
        raise ModelCheckpointError("/model", "model state structure differs from semantic reconstruction")
    for (path, left), right in zip(before, after, strict=True):
        if left is right:
            continue
        if eqx.is_array(left) and eqx.is_array(right):
            equal = (
                getattr(left, "weak_type", False) == getattr(right, "weak_type", False)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and _array_bytes(left) == _array_bytes(right)
            )
        else:
            equal = bool(eqx.tree_equal(left, right, typematch=True))
        if not equal:
            raise ModelCheckpointError(
                "/model" + jax.tree_util.keystr(path),
                "state differs from semantic reconstruction; declare the value or keep it in method extra state",
            )


def _array_bytes(value: jax.Array | np.ndarray) -> bytes:
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value).tobytes(order="C")


def _verify_values(expected, declarations, tensors, saved_declarations) -> None:
    if problem := next(_value_differences(expected, declarations, tensors, saved_declarations), None):
        _, path, reason = problem
        raise ModelCheckpointError(path, reason)


def _value_differences(
    expected: Mapping[str, jax.Array | np.ndarray],
    declarations: Mapping[str, object],
    tensors: Mapping[str, jax.Array | np.ndarray],
    saved_declarations: Mapping[str, object],
) -> Iterator[tuple[Literal["missing_slot", "unexpected_slot", "mismatched_slot"], str, str]]:
    present = saved_declarations.keys() | tensors.keys()
    for path in sorted(expected.keys() - tensors.keys()):
        yield "missing_slot", path, "missing authoritative tensor"
    for path in sorted(present - expected.keys()):
        yield "unexpected_slot", path, "unexpected authoritative value"
    for path in sorted(expected.keys() & tensors.keys()):
        current, restored = expected[path], tensors[path]
        differences = []
        if saved_declarations.get(path) != declarations[path]:
            differences.append("owner, revision, value schema or representation differs")
        if restored.shape != current.shape or restored.dtype != current.dtype:
            differences.append(
                f"expected shape {current.shape} and dtype {current.dtype}, "
                f"found shape {restored.shape} and dtype {restored.dtype}",
            )
        if differences:
            yield "mismatched_slot", path, ", ".join(differences)


def model_slot_fit_problems(
    graph: GraphSpec, template: Graph, replica: str, tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, object], *, unsupported: Iterable[tuple[str, ...]],
) -> Iterator[ModelSlotFitProblem]:
    unsupported = frozenset(unsupported)
    excluded = tuple(semantic_key("model", "value", path[:-1], path[-1]) + "/" for path in unsupported)
    retained = {path for path, _ in graph_leaf_specs(graph)} - unsupported
    expected, declarations = _describe(graph, template, retained=retained)
    for kind, address, detail in _value_differences(
        expected, declarations,
        {path: value for path, value in tensors.items() if not path.startswith(excluded)},
        {path: value for path, value in metadata.items() if not path.startswith(excluded)},
    ):
        yield ModelSlotFitProblem(kind=kind, replica=replica, address=address or "/", detail=detail)


@dataclass(frozen=True)
class ModelParameterLayout:
    """A model's addressable parameters and the semantic model slots that hold them."""

    graph: GraphSpec
    template: Graph
    addresses: Mapping[str, str]
    arrays: Mapping[str, jax.Array | np.ndarray]
    declarations: Mapping[str, object]

    @classmethod
    def from_model(cls, lowered: LoweredModelDocument, template: Graph) -> ModelParameterLayout:
        graph = effective_model_graph(lowered.graph, template)
        arrays, declarations = _describe(graph, template)
        addresses = {}
        for path, node in graph_leaf_specs(graph):
            owner = template.get_node(path[:-1]) if path[:-1] else template
            facts = owner.component_bindings[path[-1]].model_facts
            if facts.is_fixed_configuration(node):
                continue
            origin = lowered.source_address(
                "/graph/" + "/".join((*[token for name in path[:-1] for token in ("subgraphs", name)], "nodes", path[-1])),
            )
            for value in facts.values:
                slot = semantic_key("model", "value", path, value.symbol)
                if value.kind != "parameter" or slot not in arrays:
                    continue
                if origin is None or origin.element.kind != "component":
                    raise ModelCheckpointError(slot, "Parameter has no authored Model address")
                address = ModelAddress(composites=origin.composites, element=FieldElement(
                    kind="field", parent=origin.element, path=("parameters", value.symbol),
                )).to_text()
                if address in addresses:
                    raise ModelCheckpointError(slot, "Multiple parameters resolve to the same Model address")
                addresses[address] = slot
        return cls(graph, template, addresses, arrays, declarations)

    def fitting(self, parameters: Mapping[str, jax.Array]) -> tuple[ModelSlotFitProblem, ...]:
        expected = {address: self.arrays[slot] for address, slot in self.addresses.items()}
        declarations = {address: self.declarations[slot] for address, slot in self.addresses.items()}
        return tuple(
            ModelSlotFitProblem(kind=kind, replica="0", address=address, detail=detail)
            for kind, address, detail in _value_differences(
                expected, declarations, parameters, {address: declarations.get(address) for address in parameters},
            )
        )

    def apply(self, parameters: Mapping[str, jax.Array]) -> Graph:
        arrays = dict(self.arrays)
        arrays.update({self.addresses[address]: array for address, array in parameters.items()})
        return _restore_model(self.graph, self.template, arrays, self.declarations)


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
    specs = {path: (node, physical) for path, node, physical in graph_leaf_occurrences(graph)}
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
        node, physical = specs[path]
        owner = binding.model_facts
        if (
            owner is None or node.type != binding.type_id or node.revision not in (None, binding.revision)
            or not owner.supports_checkpoint(node, physical)
            or (physical is None and (binding.owner_class is None or not isinstance(component, binding.owner_class)))
        ):
            raise ModelCheckpointError(owner_path, "unsupported runtime owner identity or revision")
        if owner.is_fixed_configuration(node):
            continue
        for value in owner.values:
            address = semantic_key("model", "value", path, value.symbol)
            leaf = attrgetter(value.field)(component)
            if leaf is None and value.kind == "parameter":
                continue
            if type(leaf) is float:
                representation = "python-float"
            elif isinstance(leaf, jax.Array) and not leaf.weak_type:
                representation = "jax-array"
            else:
                raise ModelCheckpointError(address, "unsupported authoritative value representation")
            arrays[address] = np.asarray(leaf) if representation == "python-float" else leaf
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
