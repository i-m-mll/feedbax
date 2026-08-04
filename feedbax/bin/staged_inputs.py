"""The one staged-input surface every execution command-line exposes.

A staged execution descriptor plus explicitly bound roots is how a caller states
where the bytes an execution reads actually live. There is exactly one shape for
that statement — one versioned descriptor, and repeated ``NAME=ROOT`` bindings
for artifact providers, retained manifest stores, and checkpoint custody — and
this module is that shape, so a second command does not grow a second spelling
of the same declaration.

The binding flags require the descriptor and the descriptor alone is valid. That
asymmetry is the contract rather than a convenience: the descriptor is what names
the logical authorities an execution may use, and a root bound to a name the
descriptor never declared is a root nobody asked for. A descriptor with no roots
bound is a complete statement — every authority it declares must be bound, which
``resolve_staged_execution_context`` proves — so refusing it would refuse the
descriptor's own empty case.

Nothing here infers a root from anywhere else. A receipt root, an output
directory, and a repository root are each their own declaration with their own
meaning, and reusing one as a provider root would silently merge two custody
domains that a caller deliberately kept apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.analysis.execution_context import (
    StagedArtifactProviderRootBinding,
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContext,
    StagedManifestRootBinding,
    resolve_staged_execution_context,
)
from feedbax.contracts.staged_execution import StagedExecutionDescriptor


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Return a JSON object, refusing a member the document states twice.

    A repeated key is not a document with a last-writer-wins value; it is two
    incompatible statements, and silently keeping one of them would bind a
    declaration the author cannot see in their own file.
    """
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON document contains duplicate key {key!r}")
        payload[key] = value
    return payload


def load_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Load one JSON object from *path*, refusing duplicate keys and non-objects."""
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} document must be a JSON object")
    return payload


def binding_parts(value: str, *, option: str) -> tuple[str, str]:
    """Split one ``NAME=ROOT`` binding, refusing either half being empty."""
    name, separator, root = value.partition("=")
    if not separator or not name or not root:
        raise ValueError(f"{option} must use NAME=ROOT syntax")
    return name, root


def add_staged_input_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the staged execution descriptor and root-binding flags to *parser*."""
    parser.add_argument(
        "--execution-descriptor",
        type=Path,
        default=None,
        help="Versioned staged execution descriptor for explicit runtime bindings.",
    )
    parser.add_argument(
        "--artifact-provider",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help=(
            "Bind one authenticated-manifest artifact provider root; may be repeated. "
            "Requires --execution-descriptor."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help=(
            "Bind one retained Feedbax manifest-store root; may be repeated. "
            "Requires --execution-descriptor."
        ),
    )
    parser.add_argument(
        "--checkpoint-custody",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help=(
            "Bind one named checkpoint custody root; may be repeated. "
            "Requires --execution-descriptor."
        ),
    )
    return parser


def require_descriptor_for_bindings(args: argparse.Namespace) -> None:
    """Refuse bound roots that no descriptor declares, before anything executes."""
    if (args.artifact_provider or args.manifest_root or args.checkpoint_custody) and (
        args.execution_descriptor is None
    ):
        raise ValueError(
            "--artifact-provider, --manifest-root, and --checkpoint-custody "
            "require --execution-descriptor"
        )


def staged_execution_context(args: argparse.Namespace) -> StagedExecutionContext | None:
    """Resolve the staged execution context these arguments declare, or ``None``.

    ``None`` is the statement that no staged bindings were declared at all, and
    it is deliberately distinct from an empty resolved context: a command that
    gets ``None`` leaves its existing cold-start resolution untouched rather than
    routing it through a staged surface nobody asked for.
    """
    require_descriptor_for_bindings(args)
    if args.execution_descriptor is None:
        return None
    descriptor = StagedExecutionDescriptor.model_validate(
        load_json_object(args.execution_descriptor, label="--execution-descriptor")
    )
    return resolve_staged_execution_context(
        descriptor,
        artifact_provider_bindings=[
            StagedArtifactProviderRootBinding(
                *binding_parts(value, option="--artifact-provider")
            )
            for value in args.artifact_provider
        ],
        manifest_root_bindings=[
            StagedManifestRootBinding(*binding_parts(value, option="--manifest-root"))
            for value in args.manifest_root
        ],
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding(
                *binding_parts(value, option="--checkpoint-custody")
            )
            for value in args.checkpoint_custody
        ],
    )


__all__ = [
    "add_staged_input_arguments",
    "binding_parts",
    "load_json_object",
    "reject_duplicate_json_keys",
    "require_descriptor_for_bindings",
    "staged_execution_context",
]
