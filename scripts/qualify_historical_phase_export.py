"""Export the pinned nominal checkpoint's numerical leaves for phase qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np

from feedbax.contracts.manifest import ParentRef
from feedbax.persistence import ImmutableArtifactBlobProvider
from feedbax.training import materialize_checkpoint_custody_archive, resolve_checkpoint_custody_ref


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def export(archive: Path, source: dict, output: Path) -> None:
    """Authenticate original custody and export all numerical leaves without objects."""
    raw = archive.read_bytes()
    assert sha256(raw) == source["archive_sha256"]
    parent = ParentRef.model_validate(source["parent_ref"])
    with TemporaryDirectory(prefix="historical-phase-reader-") as scratch:
        root = Path(scratch).resolve()
        provider = ImmutableArtifactBlobProvider(root / "provider")
        artifact = provider.store_bytes(
            raw,
            role="training_checkpoint_custody_archive",
            logical_name=f"{parent.id}.checkpoint-custody.tar.gz",
            media_type="application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip",
            metadata={
                "schema_id": "feedbax.archive.training_checkpoint_custody",
                "schema_version": "feedbax.archive.training_checkpoint_custody.v1",
                "transaction_root_sha256": source["transaction_root_sha256"],
            },
        )
        materialized = materialize_checkpoint_custody_archive(
            provider, artifact, root / "source", expected_parent_ref=parent,
            expected_transaction_root_sha256=source["transaction_root_sha256"],
        )
        resolved = resolve_checkpoint_custody_ref(parent, allowed_root=materialized.destination)
        assert resolved.manifest_sha256 == parent.metadata["manifest_sha256"]
        assert resolved.manifest.transaction_id == parent.id
        assert resolved.manifest.completed_training_batches == source["completed_batches"]
        assert set(resolved.slots) == {"model", "optimizer", "prng", "batch_counter"}
        arrays, slots = {}, {}
        for name, tree in resolved.slots.items():
            leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
            records = []
            for index, (path, leaf) in enumerate(leaves):
                array = np.asarray(jax.device_get(leaf))
                if array.dtype.kind not in "bfiuc":
                    raise ValueError(f"nonportable numerical dtype at {name}{jax.tree_util.keystr(path)}")
                key = f"{name}_{index}"
                arrays[key] = array
                records.append({
                    "name": key,
                    "path": jax.tree_util.keystr(path),
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "sha256": sha256(array.tobytes(order="C")),
                    "nonzero": int(np.count_nonzero(array)),
                })
            slots[name] = records

    output.mkdir()
    values_path = output / "values.npz"
    np.savez(values_path, **arrays)
    with np.load(values_path, allow_pickle=False) as reloaded:
        assert set(reloaded.files) == set(arrays)
        for records in slots.values():
            for record in records:
                value = reloaded[record["name"]]
                assert value.dtype.str == record["dtype"] and list(value.shape) == record["shape"]
                assert sha256(value.tobytes(order="C")) == record["sha256"]
    assert sha256(archive.read_bytes()) == source["archive_sha256"]
    (output / "export.json").write_text(json.dumps({
        "source": source,
        "archive_size_bytes": len(raw),
        "values_sha256": sha256(values_path.read_bytes()),
        "slots": slots,
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_receipt.read_bytes())["source_checkpoint"]
    export(args.archive, source, args.output)


if __name__ == "__main__":
    main()
