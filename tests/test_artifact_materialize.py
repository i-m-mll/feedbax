from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.nn import LeakyRNNCell
from feedbax.artifact_materialize import (
    UnsupportedArtifactSchemaError,
    materialize_array_store,
    materialize_model_artifact,
    template_array_roles,
)
from feedbax.artifact_schema import (
    ArrayStoreValidationError,
    array_store_ref,
    read_npz_array_store,
    write_npz_array_store,
)
from feedbax.manifest import ModelArtifactManifest, spec_payload


class Nested(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray


class TinyModel(eqx.Module):
    nodes: dict[str, Nested]


class PublicCellWrapper(eqx.Module):
    _cell: LeakyRNNCell

    @property
    def cell(self) -> LeakyRNNCell:
        return self._cell

    def __call__(self, input, state):
        return self._cell(input, state, jax.random.key(0))


class RecurrentModel(eqx.Module):
    hidden: PublicCellWrapper


def _template() -> TinyModel:
    return TinyModel(
        nodes={
            "net": Nested(
                weight=jnp.zeros((2, 3), dtype=jnp.float32),
                bias=jnp.zeros((2,), dtype=jnp.float32),
            )
        }
    )


def test_template_array_roles_follow_jax_key_paths() -> None:
    assert template_array_roles(_template()) == [
        "model.nodes.net.bias",
        "model.nodes.net.weight",
    ]


def test_template_array_roles_use_public_property_for_private_cell_field() -> None:
    model = RecurrentModel(
        hidden=PublicCellWrapper(
            _cell=LeakyRNNCell(2, 3, key=jax.random.key(0)),
        ),
    )

    assert template_array_roles(model) == [
        "model.hidden.cell.bias",
        "model.hidden.cell.weight_hh",
        "model.hidden.cell.weight_ih",
    ]


def test_materialize_array_store_replaces_template_arrays(tmp_path: Path) -> None:
    path = tmp_path / "model.arrays.npz"
    arrays = {
        "model.nodes.net.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
    }
    write_npz_array_store(path, arrays, store_role="params")
    model = materialize_array_store(_template(), read_npz_array_store(path))

    np.testing.assert_array_equal(
        np.asarray(model.nodes["net"].weight), arrays["model.nodes.net.weight"]
    )
    np.testing.assert_array_equal(
        np.asarray(model.nodes["net"].bias), arrays["model.nodes.net.bias"]
    )


def test_materialize_array_store_uses_public_property_role_for_private_cell_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.arrays.npz"
    arrays = {
        "model.hidden.cell.weight_hh": np.arange(9, dtype=np.float32).reshape(3, 3),
        "model.hidden.cell.weight_ih": np.arange(6, dtype=np.float32).reshape(3, 2),
        "model.hidden.cell.bias": np.ones((3,), dtype=np.float32),
    }
    write_npz_array_store(path, arrays, store_role="params")
    template = RecurrentModel(
        hidden=PublicCellWrapper(
            _cell=LeakyRNNCell(2, 3, key=jax.random.key(0)),
        ),
    )

    model = materialize_array_store(template, read_npz_array_store(path))

    np.testing.assert_array_equal(
        np.asarray(model.hidden.cell.weight_hh), arrays["model.hidden.cell.weight_hh"]
    )
    np.testing.assert_array_equal(
        np.asarray(model.hidden.cell.weight_ih), arrays["model.hidden.cell.weight_ih"]
    )
    np.testing.assert_array_equal(
        np.asarray(model.hidden.cell.bias), arrays["model.hidden.cell.bias"]
    )


def test_materialize_array_store_rejects_role_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.arrays.npz"
    write_npz_array_store(
        path,
        {"model.nodes.net.weight": np.ones((2, 3), dtype=np.float32)},
        store_role="params",
    )

    with pytest.raises(ArrayStoreValidationError, match="missing"):
        materialize_array_store(_template(), read_npz_array_store(path))


def test_materialize_array_store_rejects_unexpected_roles(tmp_path: Path) -> None:
    path = tmp_path / "model.arrays.npz"
    write_npz_array_store(
        path,
        {
            "model.nodes.net.weight": np.ones((2, 3), dtype=np.float32),
            "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
            "model.nodes.unexpected.weight": np.ones((1,), dtype=np.float32),
        },
        store_role="params",
    )

    with pytest.raises(ArrayStoreValidationError, match="unexpected"):
        materialize_array_store(_template(), read_npz_array_store(path))


def test_materialize_array_store_rejects_shape_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.arrays.npz"
    write_npz_array_store(
        path,
        {
            "model.nodes.net.weight": np.ones((3, 2), dtype=np.float32),
            "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
        },
        store_role="params",
    )

    with pytest.raises(ArrayStoreValidationError, match="shape mismatch"):
        materialize_array_store(_template(), read_npz_array_store(path))


def test_materialize_array_store_rejects_dtype_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.arrays.npz"
    write_npz_array_store(
        path,
        {
            "model.nodes.net.weight": np.ones((2, 3), dtype=np.int32),
            "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
        },
        store_role="params",
    )

    with pytest.raises(ArrayStoreValidationError, match="dtype mismatch"):
        materialize_array_store(_template(), read_npz_array_store(path))


def test_materialize_model_artifact_resolves_manifest_store(tmp_path: Path) -> None:
    arrays_path = tmp_path / "model.arrays.npz"
    arrays = {
        "model.nodes.net.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
    }
    payload = write_npz_array_store(arrays_path, arrays, store_role="params")
    manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:test",
        graph_spec=spec_payload("GraphSpec", {"nodes": {}, "wires": []}),
        parameter_store=array_store_ref(arrays_path, payload),
    )
    manifest_path = tmp_path / "model.artifact.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

    model = materialize_model_artifact(manifest_path, _template())

    np.testing.assert_array_equal(
        np.asarray(model.nodes["net"].weight), arrays["model.nodes.net.weight"]
    )
    np.testing.assert_array_equal(
        np.asarray(model.nodes["net"].bias), arrays["model.nodes.net.bias"]
    )


def test_materialize_model_artifact_rejects_unsupported_manifest_schema(
    tmp_path: Path,
) -> None:
    arrays_path = tmp_path / "model.arrays.npz"
    payload = write_npz_array_store(
        arrays_path,
        {
            "model.nodes.net.weight": np.ones((2, 3), dtype=np.float32),
            "model.nodes.net.bias": np.ones((2,), dtype=np.float32),
        },
        store_role="params",
    )
    manifest = ModelArtifactManifest(
        id="feedbax-model-artifact:test",
        schema_version="feedbax.manifest.v0",
        graph_spec=spec_payload("GraphSpec", {"nodes": {}, "wires": []}),
        parameter_store=array_store_ref(arrays_path, payload),
    )

    with pytest.raises(UnsupportedArtifactSchemaError, match="registered manifest migration"):
        materialize_model_artifact(manifest, _template(), root=tmp_path)
