"""Microbenchmark checkpoint custody/adoption hot paths."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from feedbax.contracts.checkpoints import LeafManifestEntry
from feedbax.training.legacy_checkpoint_adoption import (
    _array_paths,
    _coerce_array,
    _get_by_path,
    adopt_tree_from_legacy_stream,
    read_raw_np_save_stream,
)


def _entries(leaf_count: int) -> list[LeafManifestEntry]:
    return [
        LeafManifestEntry(
            tree_path=f"/leaf_{index:04d}",
            kind="array",
            shape=(4,),
            dtype="float32",
        )
        for index in range(leaf_count)
    ]


def _template(leaf_count: int) -> dict[str, Any]:
    return {
        f"leaf_{index:04d}": jnp.zeros((4,), dtype=jnp.float32)
        for index in range(leaf_count)
    }


def _write_stream(path: Path, leaf_count: int) -> None:
    with path.open("wb") as stream:
        for index in range(leaf_count):
            np.save(
                stream,
                np.full((4,), float(index), dtype=np.float32),
                allow_pickle=False,
            )


def _legacy_per_leaf_tree_at(
    stream_path: Path,
    manifest_entries: Sequence[LeafManifestEntry],
    current_template: Any,
) -> Any:
    records = read_raw_np_save_stream(stream_path, manifest_entries)
    current_arrays = _array_paths(current_template)
    adopted = current_template
    for entry, record in zip(manifest_entries, records, strict=True):
        path_entries = current_arrays[entry.tree_path]
        old_leaf = _get_by_path(adopted, path_entries)
        replacement = _coerce_array(record.value, old_leaf)
        adopted = eqx.tree_at(
            lambda tree, p=path_entries: _get_by_path(tree, p),
            adopted,
            replacement,
        )
    return adopted


def _time_call(func, *args) -> float:
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


def _median(values: Sequence[float]) -> float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    entries = _entries(args.leaves)
    template = _template(args.leaves)
    with tempfile.TemporaryDirectory(prefix="feedbax-custody-hot-path-bench-") as tmp:
        stream = Path(tmp) / "model.eqx"
        _write_stream(stream, args.leaves)

        _legacy_per_leaf_tree_at(stream, entries, template)
        adopt_tree_from_legacy_stream(stream, entries, template)

        legacy_times = [
            _time_call(_legacy_per_leaf_tree_at, stream, entries, template)
            for _ in range(args.repeat)
        ]
        batched_times = [
            _time_call(adopt_tree_from_legacy_stream, stream, entries, template)
            for _ in range(args.repeat)
        ]

    legacy_median = _median(legacy_times)
    batched_median = _median(batched_times)
    speedup = legacy_median / batched_median if batched_median else float("inf")
    print(f"synthetic_array_leaves={args.leaves}")
    print(f"repeats={args.repeat}")
    print(f"legacy_per_leaf_tree_at_median_s={legacy_median:.6f}")
    print(f"batched_tree_at_median_s={batched_median:.6f}")
    print(f"adoption_speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
