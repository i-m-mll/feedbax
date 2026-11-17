"""Inspect the declared project Store or transfer a local receipt packet."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feedbax.bin.project_commands import CommandOutput
    from feedbax.contracts.project_io import TextProject
    from feedbax.contracts.store_closure import StoreClosureRef
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider


_PACKET_REF = "packets/export"
_PROJECT_DIR_HELP = "Project declaration file or directory (default: search from current directory)."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="store_command", required=True)
    gc = commands.add_parser("gc", help="Report retained and potentially collectable Store bytes.")
    gc.add_argument("--dry-run", action="store_true", required=True)
    gc.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help=_PROJECT_DIR_HELP,
    )

    export = commands.add_parser(
        "export", help="Export an exact committed receipt to a same-volume packet."
    )
    export.add_argument("attempt", metavar="ATTEMPT", help="Committed receipt attempt ID.")
    export.add_argument(
        "--dest", dest="packet_dir", type=Path, required=True, help="Destination packet directory."
    )
    export.add_argument(
        "--product",
        metavar="ROLE",
        help="Select a product role (add --member for multi-member roles).",
    )
    export.add_argument("--member", metavar="NAME", help="Select a named member of that role.")
    export.add_argument(
        "--include-ancestry", action="store_true", help="Include exact bound inputs transitively."
    )
    receive = commands.add_parser(
        "import", help="Import a local receipt packet into a project Store."
    )
    receive.add_argument("packet_dir", type=Path, metavar="PACKET", help="Source packet directory.")
    receive.add_argument("--name", required=True, help="Destination ref name under packets/.")
    for command in (export, receive):
        command.add_argument("--project-dir", type=Path, default=Path("."), help=_PROJECT_DIR_HELP)


def _export_selection(
    store: ImmutableArtifactBlobProvider, args: argparse.Namespace
) -> StoreClosureRef:
    from feedbax.contracts.effective_success import (
        INDEX_REF,
        decode_success_index,
        read_committed_success,
    )
    from feedbax.contracts.request_refs import NamedMember, ProductRef
    from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef

    if args.member is not None and args.product is None:
        raise ValueError("--member requires --product")
    with store._exclusive_lock(create=False):
        index = decode_success_index(store.read_ref(INDEX_REF))
        if (ref := index.attempts.get(args.attempt)) is None:
            raise ValueError(f"No committed receipt for attempt {args.attempt!r}")
        receipt = read_committed_success(index, ref, store.get_object)
        if receipt.attempt != args.attempt:
            raise ValueError(f"Receipt differs from the indexed attempt {args.attempt!r}")

    product = None
    if args.product is not None:
        product = ProductRef(
            request_id=receipt.request.id,
            role=args.product,
            select=NamedMember(member=args.member) if args.member is not None else None,
        )
    return StoreClosureRef(
        roots=(ReceiptRoot(receipt=ref, product=product),),
        include_ancestry=args.include_ancestry,
    )


def _transfer_packet(text: TextProject, args: argparse.Namespace) -> int:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.store_sync import import_packet, sync_store

    store = ImmutableArtifactBlobProvider(text.store_path)
    packet = ImmutableArtifactBlobProvider(Path(os.path.abspath(args.packet_dir)))
    packet_path = packet.root.resolve()
    if packet_path.is_relative_to(store.root) or store.root.is_relative_to(packet_path):
        raise ValueError("Packet directory and project Store must be separate")

    if args.store_command == "export":
        destination, ref = packet, _PACKET_REF
        result = sync_store(store, destination, _export_selection(store, args), destination_ref=ref)
    else:
        destination, ref = store, f"packets/{args.name}"
        result = import_packet(packet, destination, packet_ref=_PACKET_REF, name=args.name)
    print(
        json.dumps(
            {
                "store": str(destination.root),
                "ref": ref,
                "cloned": result.cloned,
                "reused": result.reused,
                "closure": result.closure.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


def run(args: argparse.Namespace, output: CommandOutput) -> int:
    from feedbax.contracts.project_io import find_text_project, read_text_project_snapshot

    try:
        text = read_text_project_snapshot(find_text_project(args.project_dir))
        if args.store_command != "gc":
            return _transfer_packet(text, args)

        from feedbax.host import HostClient
        from feedbax.project_retention import inspect_project_retention

        report = inspect_project_retention(text, HostClient())
    except (ValueError, OSError) as error:
        output.failure(error, path=args.project_dir)
        return 2
    print(json.dumps({"dry_run": True, **asdict(report)}, indent=2))
    return 2 if report.objects.gaps else 0
