"""Inspect a remote object cache and project authenticated run-local content trees."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.orchestration.input_materialization import (
    materialize_content_tree_views,
    publish_content_file,
)
from feedbax.persistence.artifact_custody import (
    ArtifactBlobContainmentError,
    ArtifactBlobIntegrityError,
    read_content_tree_descriptor,
    verify_content_file,
)

_OBJECT_REFS = TypeAdapter(list[ContentObjectRef])
_TREE_REFS = TypeAdapter(list[ContentTreeRef])


def _object_path(root: Path, ref: ContentObjectRef) -> Path:
    return root / ref.sha256[:2] / ref.sha256


def _inspect(object_root: Path, refs: Sequence[ContentObjectRef]) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"present": [], "missing": [], "conflicts": []}
    for ref in refs:
        try:
            verify_content_file(_object_path(object_root, ref), ref)
        except FileNotFoundError:
            report["missing"].append(ref.sha256)
        except (ArtifactBlobContainmentError, ArtifactBlobIntegrityError):
            report["conflicts"].append(ref.sha256)
        else:
            report["present"].append(ref.sha256)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    inspect = operations.add_parser("inspect")
    inspect.add_argument("cache_root", type=Path)
    inspect.add_argument("object_refs")
    publish = operations.add_parser("publish")
    publish.add_argument("cache_root", type=Path)
    publish.add_argument("upload_root", type=Path)
    publish.add_argument("object_refs")
    project = operations.add_parser("project")
    project.add_argument("cache_root", type=Path)
    project.add_argument("destination_root", type=Path)
    project.add_argument("tree_refs")
    args = parser.parse_args(argv)

    object_root = args.cache_root / "objects" / "sha256"
    if args.operation == "project":
        tree_refs = _TREE_REFS.validate_python(strict_json_loads(args.tree_refs, ref="tree_refs"))
        trees = [
            (ref, read_content_tree_descriptor(_object_path(object_root, ref.descriptor), ref))
            for ref in tree_refs
        ]
        materialize_content_tree_views(
            trees,
            destination_root=args.destination_root,
            source_for=lambda ref: _object_path(object_root, ref),
            tree_root_mode=0o755,
        )
        return

    object_refs = _OBJECT_REFS.validate_python(strict_json_loads(args.object_refs, ref="object_refs"))
    if args.operation == "inspect":
        print(json.dumps(_inspect(object_root, object_refs)))
        return
    for ref in object_refs:
        source = args.upload_root / ref.sha256
        publish_content_file(
            source, _object_path(object_root, ref), ref, verify_source_on_reuse=True
        )
        source.unlink()


if __name__ == "__main__":
    main()
