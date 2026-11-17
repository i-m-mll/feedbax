"""Replay production sweep expansion against the pinned conversion proof."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from functools import cache
from itertools import groupby, product
import json
from pathlib import Path
import tarfile
from time import perf_counter
from typing import Any

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v1_bytes
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.contracts.matrix_core import (
    MatrixAxisValueGenerator,
    apply_override_patches,
    expand_axis_value_generator,
)
from feedbax.contracts.migrations import migrate_graph_spec
from feedbax.contracts.project_sweep import (
    ExpandedRow,
    NamedAxis,
    SetValue,
    Sweep,
    count_sweep,
    expand_sweep,
    inherit_sweep,
    sweep_patches,
)

CONVERSION_SHA256 = "1619e5ca9de16d4bf14daf43e38c0ca9b3bb87af8f5ac4b6ff3d5f9f84a05cab"
EVALUATIONS = (
    "sisu/tier-a-acceptance-evaluation",
    "reaching-bank-stab0-velocity-evaluation",
    "reaching-bank-stab10-velocity-evaluation",
)
TRAINING = (
    "inner-solve-budget-saturation",
    "mapped-fork-continuation",
    "learning-rate-rewarm-continuation",
    "sisu/target2x-target4x-floor",
)


def select(value: Any, path: str) -> Any:
    for segment in path.split("."):
        value = value[int(segment)] if isinstance(value, list) else value[segment]
    return value


def prototype_record_merge(parent: dict, child: dict) -> dict:
    """Reconstitute the proof's partial patch records before parsing current grammar."""
    result = dict(parent)
    for key, value in child.items():
        previous = parent.get(key)
        if value is None:
            result.pop(key, None)
        elif all(isinstance(item, dict) and "document" not in item for item in (previous, value)):
            result[key] = prototype_record_merge(previous, value)
        else:
            result[key] = value
    return result


def complete_axis_patches(sweep: dict, inherited: Sweep | None) -> dict:
    if inherited is None:
        return sweep
    axes = {}
    for name, axis in sweep.get("axes", {}).items():
        previous_axis = inherited.axes.get(name)
        if not isinstance(previous_axis, NamedAxis) or axis is None:
            axes[name] = axis
            continue
        values = {}
        for key, value in axis["values"].items():
            previous = previous_axis.values.get(key)
            if previous is None or value is None:
                values[key] = value
                continue
            patches = {}
            for path, patch in value.get("patches", {}).items():
                prior = previous.patches.get(path)
                patches[path] = (
                    prototype_record_merge(prior.model_dump(mode="json"), patch)
                    if prior is not None and patch is not None
                    else patch
                )
            values[key] = {**value, "patches": patches}
        axes[name] = {**axis, "values": values}
    return {**sweep, "axes": axes}


class ConversionFixture:
    """Read the prototype's historical alias, row patch and document spellings."""

    def __init__(self, project: dict):
        self.project = project

    def inline(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.inline(item) for item in value]
        if isinstance(value, dict):
            if "document" in value and value.keys() <= {"document", "pointer"}:
                target = self.project["documents"][value["document"]]["payload"]
                if "pointer" in value:
                    target = select(target, value["pointer"])
                return self.inline(target)
            return {key: self.inline(item) for key, item in value.items()}
        return value

    @cache
    def step(self, name: str) -> tuple[dict, Sweep | None, dict]:
        step = self.project["steps"][name]
        base = step["base"]
        inherited, inputs = None, {}
        if "step" in base:
            payload, inherited, inputs = self.step(base["step"])
        elif "run" in base:
            payload = self.project["stored_runs"][base["run"]]
        elif "inline" in base:
            payload = base["inline"]
        else:
            payload = self.project["documents"][base["document"]]["payload"]
            if "pointer" in base:
                payload = select(payload, base["pointer"])
        for layer in step.get("deltas", ()):
            payload = apply_override_patches(payload, layer)
        sweep = step.get("sweep")
        if sweep is not None:
            sweep = complete_axis_patches(sweep, inherited)
        if sweep is not None and "rows" in sweep:
            rows = {}
            for row_id, row in sweep["rows"].items():
                patches = {
                    item["path"]: {key: value for key, value in item.items() if key != "path"}
                    for item in row["patches"]
                }
                assert len(patches) == len(row["patches"])
                rows[row_id] = {**row, "patches": patches}
            sweep = {**sweep, "rows": rows}
        return (
            payload,
            inherit_sweep(inherited, Sweep.model_validate(sweep) if sweep is not None else None),
            prototype_record_merge(inputs, step.get("inputs", {})),
        )


def training_rows(
    fixture: ConversionFixture, name: str
) -> Iterator[tuple[str, tuple[int | None, dict]]]:
    base, sweep, _ = fixture.step(name)
    for row in expand_sweep(sweep):
        seed = row.patches.get("$row_seed")
        assert seed is None or isinstance(seed, SetValue) and isinstance(seed.value, int)
        writes = {path: patch for path, patch in row.patches.items() if path != "$row_seed"}
        payload = fixture.inline(apply_override_patches(base, sweep_patches(writes)))
        if "inline" in payload.get("graph", {}):
            payload["graph"]["inline"] = migrate_graph_spec(
                payload["graph"]["inline"], path="graph.inline"
            ).payload
        yield row.id, (seed.value if seed is not None else None, payload)


def evaluation_rows(
    fixture: ConversionFixture, name: str
) -> tuple[tuple[ExpandedRow, ...], Iterator[tuple[str, dict]]]:
    base, sweep, inputs = fixture.step(name)
    base = dict(base)
    grid = {
        axis: expand_axis_value_generator(MatrixAxisValueGenerator.model_validate(generator))
        for axis, generator in base.pop("condition_grid").items()
    }
    rows = expand_sweep(sweep)
    assert len(rows) == count_sweep(sweep)
    legacy_inputs = [
        {"path": path, **patch}
        for reference in inputs.values()
        for path, patch in reference.get("legacy_binding", {}).items()
    ]

    def materialize() -> Iterator[tuple[str, dict]]:
        # Compare in the old arm/grid/condition order while keeping the grid
        # outside the production sweep, where it is one payload dimension.
        for arm, arm_rows in groupby(rows, key=lambda row: row.coordinates["arm"].value_id):
            runs = tuple(arm_rows)
            for values in product(*grid.values()):
                grid_coords = dict(zip(grid, values, strict=True))
                grid_writes = [patch for value in values for patch in value.deltas]
                for row in runs:
                    ids = {
                        axis: coordinate.value_id for axis, coordinate in row.coordinates.items()
                    }
                    ids.update({axis: value.id for axis, value in grid_coords.items()})
                    row_id = "--".join(
                        f"{axis}-{ids[axis]}"
                        for axis in ("arm", "target", "slot", "cond")
                        if axis in ids
                    )
                    writes = sweep_patches(row.patches) + grid_writes
                    if arm == "trained":
                        writes.extend(legacy_inputs)
                    payload = fixture.inline(apply_override_patches(base, writes))
                    yield row_id, EvaluationRunSpec.model_validate(payload).model_dump(mode="json")

    return rows, materialize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversion_archive", type=Path)
    args = parser.parse_args()
    assert sha256_bytes(args.conversion_archive.read_bytes()) == CONVERSION_SHA256
    with tarfile.open(args.conversion_archive, "r:gz") as archive:
        fixture = ConversionFixture(json.load(archive.extractfile("project.json")))
        expected = json.load(archive.extractfile("summary.json"))
    for name in EVALUATIONS:
        runs, expanded = evaluation_rows(fixture, name)
        legacy_rows = list(expanded)
        assert len(runs) == expected[name]["new_runs"], name
        assert len(legacy_rows) == expected[name]["legacy_rows"], name
        assert (
            sha256_bytes(canonical_json_v1_bytes(legacy_rows))
            == expected[name]["expanded_rows_sha256"]
        ), name
        print(f"{name}: {len(runs)} runs, {len(legacy_rows)} matching legacy rows")
    for name in TRAINING:
        rows = list(training_rows(fixture, name))
        assert [row_id for row_id, _ in rows] == expected[name]["rows"], name
        assert (
            sha256_bytes(canonical_json_v1_bytes(rows)) == expected[name]["expanded_rows_sha256"]
        ), name
        print(f"{name}: {len(rows)} matching legacy rows and payloads")

    for width in (10, 100):
        sweep = Sweep.model_validate(
            {
                "axes": {
                    "a": {"path": "params.a", "values": list(range(width))},
                    "b": {"path": "params.b", "values": list(range(width))},
                }
            }
        )
        start = perf_counter()
        count = len(expand_sweep(sweep))
        elapsed_ms = (perf_counter() - start) * 1000
        print(f"Expanded {count} rows in {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
